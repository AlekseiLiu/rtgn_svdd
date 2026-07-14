import torch
import pytest
import rtgn_svdd.training as training_module

from rtgn_svdd.data import EventStream
from rtgn_svdd.training import (
    TGNModel,
    TrainingConfig,
    create_optimizer,
    load_checkpoint,
    replay_splits,
    sample_negative_pairs,
    save_checkpoint,
    score_stream,
    train_epoch,
)


def _events() -> EventStream:
    return EventStream(
        src=torch.tensor([2, 7, 2], dtype=torch.long),
        dst=torch.tensor([7, 2, 7], dtype=torch.long),
        t=torch.tensor([1, 2, 3], dtype=torch.long),
        msg=torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
        is_attack=torch.zeros(3, dtype=torch.bool),
        is_injected=torch.zeros(3, dtype=torch.bool),
    )


def _model() -> TGNModel:
    torch.manual_seed(4)
    return TGNModel(
        num_nodes=8,
        msg_dim=1,
        memory_dim=8,
        time_dim=4,
        endpoint_dim=8,
        center=torch.zeros(16),
        variant="baseline",
        neighbor_size=2,
    )


def _robust_model() -> TGNModel:
    torch.manual_seed(4)
    return TGNModel(
        num_nodes=8,
        msg_dim=1,
        memory_dim=8,
        time_dim=4,
        endpoint_dim=8,
        center=torch.zeros(16),
        variant="robust",
        neighbor_size=2,
    )


def test_score_stream_maps_global_nodes_to_local_embeddings() -> None:
    model = _model()
    output = score_stream(model, _events(), batch_size=2)

    assert output.scores.s_mu.shape == (3,)
    assert output.event_mean.shape == (3, 16)
    assert not output.scores.s_mu.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_score_stream_offloads_completed_outputs_to_cpu() -> None:
    model = TGNModel(
        num_nodes=8,
        msg_dim=1,
        memory_dim=8,
        time_dim=4,
        endpoint_dim=8,
        center=torch.zeros(16),
        variant="robust",
        neighbor_size=2,
        device="cuda:0",
    )

    output = score_stream(model, _events(), batch_size=2)

    assert model.device.type == "cuda"
    assert output.event_mean.device.type == "cpu"
    assert output.event_log_variance is not None
    assert output.event_log_variance.device.type == "cpu"
    assert output.scores.s_mu.device.type == "cpu"
    assert output.scores.s_sigma is not None
    assert output.scores.s_sigma.device.type == "cpu"
    assert torch.isfinite(output.scores.s_mu).all()


def test_training_config_rejects_invalid_values() -> None:
    config = TrainingConfig(variant="robust", batch_size=2)
    assert config.as_dict()["negative_ratio"] == 0.3

    try:
        TrainingConfig(variant="unknown")  # type: ignore[arg-type]
    except ValueError as error:
        assert "variant" in str(error)
    else:
        raise AssertionError("invalid variant was accepted")


def test_injected_update_policy_uses_whole_batch_then_selected_subset() -> None:
    events = _events()
    mixed = EventStream(
        src=events.src,
        dst=events.dst,
        t=events.t,
        msg=events.msg,
        is_attack=events.is_attack,
        is_injected=torch.tensor([False, True, False]),
    )
    false_model = _model()
    false_output = score_stream(
        false_model, mixed, batch_size=2, update_injected_state=False
    )
    assert false_model.state_event_count == 2

    true_model = _model()
    true_output = score_stream(
        true_model, mixed, batch_size=2, update_injected_state=True
    )
    assert true_model.state_event_count == 3
    assert torch.allclose(
        false_output.scores.s_mu[:2], true_output.scores.s_mu[:2]
    )
    assert not torch.allclose(
        false_output.scores.s_mu[2:], true_output.scores.s_mu[2:]
    )


def test_negative_pairs_are_seeded_observed_and_non_self() -> None:
    nodes = torch.tensor([2, 7, 11])
    first = sample_negative_pairs(nodes, 20, generator=torch.Generator().manual_seed(8))
    second = sample_negative_pairs(nodes, 20, generator=torch.Generator().manual_seed(8))
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not bool(torch.any(first[0] == first[1]))
    assert set(torch.cat(first).tolist()) <= {2, 7, 11}


def test_baseline_epoch_is_finite_and_updates_parameters() -> None:
    model = _model()
    config = TrainingConfig(variant="baseline", batch_size=2)
    optimizer = create_optimizer(model, config)
    before = model.head.center.detach().clone()
    loss = train_epoch(
        model,
        _events(),
        optimizer,
        config,
        generator=torch.Generator().manual_seed(2),
    )
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.head.center)


def test_robust_epoch_does_not_insert_negative_pairs() -> None:
    model = _robust_model()
    config = TrainingConfig(variant="robust", batch_size=2, negative_ratio=0.5)
    optimizer = create_optimizer(model, config)
    loss = train_epoch(
        model,
        _events(),
        optimizer,
        config,
        generator=torch.Generator().manual_seed(2),
        training_nodes=torch.tensor([2, 7]),
    )
    assert torch.isfinite(torch.tensor(loss))
    assert model.state_event_count == 3


def test_checkpoint_round_trip_excludes_transient_state(tmp_path) -> None:
    events = _events()
    model = _model()
    config = TrainingConfig(variant="baseline", batch_size=2)
    optimizer = create_optimizer(model, config)
    model.reset_state()
    expected = score_stream(model, events, batch_size=2).scores.s_mu
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, config=config.as_dict(), epoch=3
    )
    payload = torch.load(path, weights_only=False)
    assert "memory" not in payload
    assert "last_update" not in payload
    assert "edge_history" not in payload
    with torch.no_grad():
        model.head.center.add_(10)
    metadata = load_checkpoint(path, model=model, optimizer=optimizer)
    assert metadata.epoch == 3
    assert model.state_event_count == 0
    actual = score_stream(model, events, batch_size=2).scores.s_mu
    assert torch.allclose(actual, expected)


def test_checkpoint_save_failure_preserves_existing_file_and_cleans_temp(
    tmp_path, monkeypatch
) -> None:
    model = _model()
    config = TrainingConfig(variant="baseline", batch_size=2)
    optimizer = create_optimizer(model, config)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, config={}, epoch=1)
    original = path.read_bytes()

    def fail_save(payload, destination):
        destination = tmp_path / destination.name
        destination.write_bytes(b"partial checkpoint")
        raise OSError("injected checkpoint write failure")

    monkeypatch.setattr(training_module.torch, "save", fail_save)
    with pytest.raises(OSError, match="injected checkpoint write failure"):
        save_checkpoint(path, model=model, optimizer=optimizer, config={}, epoch=2)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_replay_splits_resets_dirty_state_and_traverses_continuously() -> None:
    events = _events()
    train = events.slice(0, 1)
    validation = events.slice(1, 2)
    test = events.slice(2, 3)
    model = _model()
    score_stream(model, test, batch_size=1)

    first = replay_splits(model, train, validation, test, batch_size=1)
    isolated_model = _model()
    isolated_validation = score_stream(
        isolated_model, validation, batch_size=1
    ).scores.s_mu
    assert not torch.allclose(first[1].scores.s_mu, isolated_validation)

    score_stream(model, test, batch_size=1)
    second = replay_splits(model, train, validation, test, batch_size=1)
    for first_split, second_split in zip(first, second):
        assert torch.allclose(first_split.scores.s_mu, second_split.scores.s_mu)


def test_train_epoch_rejects_empty_stream() -> None:
    events = _events().slice(0, 0)
    model = _model()
    config = TrainingConfig(variant="baseline", batch_size=2)
    optimizer = create_optimizer(model, config)
    try:
        train_epoch(
            model,
            events,
            optimizer,
            config,
            generator=torch.Generator().manual_seed(2),
        )
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty training stream was accepted")

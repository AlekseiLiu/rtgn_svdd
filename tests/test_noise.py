import pytest
import torch

from rtgn_svdd.data import EventStream
from rtgn_svdd.noise import inject_test_noise


def make_test_events() -> EventStream:
    return EventStream(
        src=torch.tensor([0, 1, 2, 0]),
        dst=torch.tensor([1, 2, 0, 2]),
        t=torch.tensor([10, 11, 12, 13]),
        msg=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        is_attack=torch.tensor([False, True, False, False]),
        is_injected=torch.zeros(4, dtype=torch.bool),
        attack_name=("normal", "attack", "normal", "normal"),
    )


def test_zero_noise_returns_independent_events_and_manifest():
    original = make_test_events()

    result = inject_test_noise(
        original,
        node_ids=torch.tensor([0, 1, 2]),
        ratio=0.0,
        seed=7,
    )

    assert result.normal_denominator == 3
    assert result.injected_count == 0
    assert result.noise_ratio == 0.0
    assert result.noise_seed == 7
    assert result.node_population_policy == "dataset_observed_nodes"
    assert result.feature_distribution == "paper_covariance_5"
    assert result.allow_self_edges is False
    assert result.timestamp_tie_policy == "original_before_injected_stable"
    for field in ("src", "dst", "t", "msg", "is_attack", "is_injected"):
        returned = getattr(result.events, field)
        source = getattr(original, field)
        assert torch.equal(returned, source)
        assert returned.data_ptr() != source.data_ptr()
    assert result.events.attack_name == original.attack_name


def test_seeded_injection_has_exact_count_labels_endpoints_and_time_range():
    original = make_test_events()

    result = inject_test_noise(
        original,
        node_ids=torch.tensor([0, 1, 2]),
        ratio=1.0,
        seed=19,
    )

    injected = result.events.is_injected
    assert result.injected_count == 3
    assert injected.sum().item() == 3
    assert not result.events.is_attack[injected].any()
    assert set(result.events.src[injected].tolist()) <= {0, 1, 2}
    assert set(result.events.dst[injected].tolist()) <= {0, 1, 2}
    assert torch.all(result.events.src[injected] != result.events.dst[injected])
    assert result.events.t[injected].min().item() >= 10
    assert result.events.t[injected].max().item() <= 13
    assert [name for name, flag in zip(result.events.attack_name, injected) if flag] == [
        None,
        None,
        None,
    ]


def test_fixed_seed_is_exact_and_does_not_change_input_or_global_rng():
    original = make_test_events()
    original_values = {name: getattr(original, name).clone() for name in vars(original) if name != "attack_name"}
    global_state = torch.get_rng_state().clone()

    first = inject_test_noise(original, node_ids=torch.tensor([0, 1, 2]), ratio=1.0, seed=4)
    second = inject_test_noise(original, node_ids=torch.tensor([0, 1, 2]), ratio=1.0, seed=4)

    for field in ("src", "dst", "t", "msg", "is_attack", "is_injected"):
        assert torch.equal(getattr(first.events, field), getattr(second.events, field))
        assert torch.equal(getattr(original, field), original_values[field])
    assert first.events.attack_name == second.events.attack_name
    assert torch.equal(torch.get_rng_state(), global_state)


def test_different_seed_changes_generated_events():
    original = make_test_events()
    first = inject_test_noise(original, node_ids=torch.tensor([0, 1, 2]), ratio=2.0, seed=4)
    second = inject_test_noise(original, node_ids=torch.tensor([0, 1, 2]), ratio=2.0, seed=5)

    first_noise = first.events.msg[first.events.is_injected]
    second_noise = second.events.msg[second.events.is_injected]
    assert not torch.equal(first_noise, second_noise)


def test_named_feature_distributions_use_paper_and_historical_scales():
    original = make_test_events()
    paper = inject_test_noise(
        original,
        node_ids=torch.tensor([0, 1, 2]),
        ratio=1 / 3,
        seed=31,
        feature_distribution="paper_covariance_5",
    )
    historical = inject_test_noise(
        original,
        node_ids=torch.tensor([0, 1, 2]),
        ratio=1 / 3,
        seed=31,
        feature_distribution="historical_scale_5",
    )

    paper_msg = paper.events.msg[paper.events.is_injected]
    historical_msg = historical.events.msg[historical.events.is_injected]
    assert torch.allclose(historical_msg, paper_msg * (5.0 / (5.0**0.5)))


def test_equal_timestamps_keep_originals_before_injected_events_stably():
    original = make_test_events()
    original = EventStream(**{**vars(original), "t": torch.full((4,), 10, dtype=torch.long)})

    result = inject_test_noise(
        original,
        node_ids=torch.tensor([0, 1, 2]),
        ratio=1.0,
        seed=11,
    )

    assert result.events.is_injected.tolist() == [False, False, False, False, True, True, True]
    assert result.events.src[:4].tolist() == original.src.tolist()
    assert result.events.attack_name[:4] == original.attack_name


def test_self_edges_are_available_only_when_explicitly_enabled():
    result = inject_test_noise(
        make_test_events(),
        node_ids=torch.tensor([7]),
        ratio=1.0,
        seed=11,
        allow_self_edges=True,
    )

    injected = result.events.is_injected
    assert result.allow_self_edges is True
    assert result.events.src[injected].tolist() == [7, 7, 7]
    assert result.events.dst[injected].tolist() == [7, 7, 7]


def test_count_uses_original_normal_events_and_half_up_rounding():
    original = EventStream(
        src=torch.tensor([0, 0, 0, 0, 0, 0]),
        dst=torch.tensor([1, 1, 1, 1, 1, 1]),
        t=torch.arange(6, dtype=torch.long),
        msg=torch.ones(6, 1),
        is_attack=torch.tensor([False, False, False, False, False, True]),
        is_injected=torch.zeros(6, dtype=torch.bool),
    )

    result = inject_test_noise(
        original,
        node_ids=torch.tensor([0, 1]),
        ratio=0.5,
        seed=2,
    )

    assert result.normal_denominator == 5
    assert result.injected_count == 3


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"ratio": -0.1}, ValueError, "finite nonnegative"),
        ({"ratio": float("nan")}, ValueError, "finite nonnegative"),
        ({"node_ids": torch.tensor([0, 0])}, ValueError, "sorted and unique"),
        ({"node_ids": torch.tensor([1, 0])}, ValueError, "sorted and unique"),
        ({"node_ids": torch.tensor([0])}, ValueError, "at least two"),
        ({"node_ids": torch.tensor([0.0, 1.0])}, TypeError, "torch.long"),
        ({"feature_distribution": "unknown"}, ValueError, "feature_distribution"),
    ],
)
def test_invalid_noise_options_fail_clearly(changes, error, message):
    arguments = {
        "node_ids": torch.tensor([0, 1, 2]),
        "ratio": 1.0,
        "seed": 3,
    }
    arguments.update(changes)

    with pytest.raises(error, match=message):
        inject_test_noise(make_test_events(), **arguments)


def test_positive_ratio_requires_a_normal_event():
    original = make_test_events()
    original = EventStream(
        **{
            **vars(original),
            "is_attack": torch.ones(len(original), dtype=torch.bool),
        }
    )

    with pytest.raises(ValueError, match="without original normal events"):
        inject_test_noise(original, node_ids=torch.tensor([0, 1, 2]), ratio=0.1, seed=3)


def test_empty_test_stream_is_rejected():
    empty = make_test_events().slice(0, 0)

    with pytest.raises(ValueError, match="at least one event"):
        inject_test_noise(empty, node_ids=torch.tensor([0, 1]), ratio=0.0, seed=3)

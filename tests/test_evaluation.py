import json
from dataclasses import replace

import numpy as np
import pytest
import torch
import rtgn_svdd.evaluation as evaluation_module

from rtgn_svdd.data import EventStream
from rtgn_svdd.evaluation import (
    AggregateRecord,
    BASELINE_MU_PERCENTILES,
    EvaluationRecord,
    ScoreArtifact,
    aggregate_records,
    attack_targets,
    evaluate_baseline,
    evaluate_fixed_threshold,
    evaluate_paper_recovered,
    merge_baseline_threshold_choices,
    merge_fixed_threshold_choices,
    make_tuning_selection_artifact,
    read_aggregate_record,
    read_result_record,
    read_score_artifact,
    read_tuning_selection_artifact,
    tune_baseline_threshold,
    tune_fixed_threshold,
    write_aggregate_record,
    write_result_record,
    write_score_artifact,
    write_tuning_selection_artifact,
)


def _events(labels=(False, True, False, True), injected=(False, False, True, False)):
    n = len(labels)
    return EventStream(
        src=torch.arange(n), dst=torch.arange(n) + n, t=torch.arange(n),
        msg=torch.zeros((n, 1)), is_attack=torch.tensor(labels),
        is_injected=torch.tensor(injected),
    )


def _metadata(**updates):
    result = dict(model_variant="robust", day="wednesday", noise_ratio=0.05,
                  repeat=0, training_seed=3, noise_seed=7, checkpoint_id="ckpt-a",
                  update_injected_state=True, uncertainty_mode="per_dimension_log_variance")
    result.update(updates)
    return result


def test_targets_and_baseline_auc_are_attack_vs_rest():
    events = _events()
    assert attack_targets(events).tolist() == [False, True, False, True]
    record = evaluate_baseline(events, [0.0, 3.0, 1.0, 2.0], **_metadata(model_variant="baseline"))
    assert record.roc_auc == 1.0
    assert record.counts == {"attack": 2, "benign": 1, "injected": 1,
                             "normal_denominator": 1, "realized_injected": 1}


def test_paper_recovered_matches_strict_historical_rejection_and_average():
    events = _events()
    record = evaluate_paper_recovered(
        events, [0.0, 3.0, 1.0, 2.0], [5.0, 10.0, 30.0, 20.0], **_metadata()
    )
    assert record.thresholds["candidates"] == [10.0, 20.0]
    expected = [0.625, 1.0]
    assert record.thresholds["auc_by_threshold"] == expected
    assert record.roc_auc == pytest.approx(np.mean(expected))
    with pytest.raises(ValueError, match="no observed"):
        evaluate_paper_recovered(events, [0, 1, 2, 3], [5, 25, 30, 40], **_metadata())


def test_paper_recovered_rejects_scores_outside_the_model_contract():
    with pytest.raises(ValueError, match="s_mu must be nonnegative"):
        evaluate_paper_recovered(
            _events(), [-0.1, 3.0, 1.0, 2.0], [5.0, 10.0, 30.0, 20.0], **_metadata()
        )


def _naive_binary_auc(labels, scores):
    positives = scores[labels]
    negatives = scores[~labels]
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) /
                 comparisons.size)


def _naive_paper_recovered(labels, s_mu, s_sigma):
    candidates = np.unique(s_sigma[(s_sigma > 5.0) & (s_sigma < 25.0)])
    aucs = [
        _naive_binary_auc(labels, np.where(s_sigma > threshold, -100.0, s_mu))
        for threshold in candidates
    ]
    return candidates, aucs


def test_paper_recovered_optimized_sweep_matches_naive_randomized_cases():
    generator = np.random.default_rng(7301)
    sigma_values = np.array([4.0, 5.0, 6.0, 10.0, 24.0, 25.0, 26.0])
    for size in range(4, 24):
        for _ in range(8):
            labels = generator.integers(0, 2, size=size, dtype=np.int64).astype(bool)
            labels[0], labels[1] = False, True
            injected = np.zeros(size, dtype=bool)
            injected[2] = not labels[2]
            s_mu = generator.integers(0, 6, size=size).astype(float)
            s_sigma = generator.choice(sigma_values, size=size)
            s_sigma[0] = 10.0
            events = _events(
                labels=tuple(bool(value) for value in labels),
                injected=tuple(bool(value) for value in injected),
            )

            candidates, expected = _naive_paper_recovered(labels, s_mu, s_sigma)
            record = evaluate_paper_recovered(
                events, s_mu, s_sigma, **_metadata()
            )

            assert record.thresholds["candidates"] == candidates.tolist()
            assert record.thresholds["auc_by_threshold"] == pytest.approx(expected)
            assert record.roc_auc == pytest.approx(np.mean(expected))


def test_paper_recovered_handles_many_candidates_with_exact_known_auc():
    negative_count = 2_000
    positive_count = 1_000
    labels = np.concatenate((np.zeros(negative_count, dtype=bool),
                             np.ones(positive_count, dtype=bool)))
    events = _events(
        labels=tuple(bool(value) for value in labels),
        injected=(False,) * labels.size,
    )
    s_mu = np.concatenate((np.zeros(negative_count), np.ones(positive_count)))
    positive_sigma = np.linspace(5.01, 24.99, positive_count)
    s_sigma = np.concatenate((np.full(negative_count, 5.0), positive_sigma))

    record = evaluate_paper_recovered(events, s_mu, s_sigma, **_metadata())

    assert record.thresholds["candidates"] == positive_sigma.tolist()
    assert record.thresholds["auc_by_threshold"] == pytest.approx(
        np.arange(1, positive_count + 1) / positive_count
    )
    assert record.roc_auc == pytest.approx((positive_count + 1) / (2 * positive_count))


def test_baseline_tuning_selects_non_final_epoch_and_validation_threshold():
    events = _events(labels=(False, True, False, True), injected=(False,) * 4)
    early = {
        "checkpoint_id": "epoch_0001",
        "epoch": 1,
        "s_mu": [0.0, 90.0, 0.0, 90.0],
        "validation_s_mu": np.arange(100, dtype=float),
    }
    final = {
        "checkpoint_id": "epoch_0002",
        "epoch": 2,
        "s_mu": [90.0, 0.0, 90.0, 0.0],
        "validation_s_mu": np.arange(200, 300, dtype=float),
    }

    choice = tune_baseline_threshold([early, final], events)

    assert BASELINE_MU_PERCENTILES == tuple(range(60, 102, 2))
    assert choice.checkpoint_id == "epoch_0001"
    assert choice.epoch == 1
    assert choice.mu_percentile == 60
    assert choice.mu_threshold == pytest.approx(np.percentile(early["validation_s_mu"], 60))
    assert choice.f1 == 1.0
    assert len(choice.candidates) == 2 * len(BASELINE_MU_PERCENTILES)


def _selection_metadata(**updates):
    values = {
        "training_seed": 3,
        "tuning_noise_seed": 7,
        "selected_checkpoint_sha256": "a" * 64,
        "dataset_identity": {
            "day": "wednesday",
            "configured_path": "data/wednesday.csv",
            "sha256": "b" * 64,
        },
        "normal_denominator": 100,
        "realized_injected": 5,
        "feature_distribution": "paper_covariance_5",
        "update_injected_state": True,
        "uncertainty_mode": None,
        "source_revision": {"git_commit": "c" * 40, "dirty": False},
        "score_artifact": {"path": "scores/tuning/baseline.npz", "sha256": "d" * 64},
    }
    values.update(updates)
    return values


def _baseline_selection_artifact(**updates):
    events = _events(labels=(False, True, False, True), injected=(False,) * 4)
    choice = tune_baseline_threshold(
        [{"checkpoint_id": "epoch_0001", "epoch": 1,
          "s_mu": [0.0, 90.0, 0.0, 90.0],
          "validation_s_mu": np.arange(100, dtype=float)}],
        events,
    )
    return make_tuning_selection_artifact(choice, **_selection_metadata(**updates))


def test_baseline_tuning_artifact_is_derived_and_round_trips_canonically(tmp_path):
    events = _events(labels=(False, True, False, True), injected=(False,) * 4)
    choice = tune_baseline_threshold(
        [{"checkpoint_id": "epoch_0001", "epoch": 1,
          "s_mu": [0.0, 90.0, 0.0, 90.0],
          "validation_s_mu": np.arange(100, dtype=float)}],
        events,
    )

    artifact = make_tuning_selection_artifact(choice, **_selection_metadata())
    path = tmp_path / "nested" / "selection.json"
    write_tuning_selection_artifact(artifact, path)

    assert artifact.model_variant == "baseline"
    assert artifact.selection_policy == "wednesday_f1"
    assert artifact.selected_logical_checkpoint_id == choice.checkpoint_id
    assert artifact.selected_epoch == choice.epoch
    assert artifact.thresholds == {
        "mu_percentile": choice.mu_percentile,
        "mu_threshold": choice.mu_threshold,
    }
    assert artifact.candidate_grid == choice.candidates
    assert read_tuning_selection_artifact(path) == artifact
    assert path.read_text() == json.dumps(
        artifact.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
    ) + "\n"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("model_variant", "other", "model_variant"),
        ("selection_policy", "final_configured_epoch", "policy"),
        ("training_seed", True, "training_seed"),
        ("tuning_day", "tuesday", "Wednesday"),
        ("tuning_noise_ratio", float("nan"), "Wednesday"),
        ("tuning_noise_seed", -1, "tuning_noise_seed"),
        ("selected_epoch", 0, "selected_epoch"),
        ("selected_logical_checkpoint_id", "", "checkpoint ID"),
        ("selected_checkpoint_sha256", "a" * 63, "SHA-256"),
        ("f1", 1.1, "f1"),
        ("normal_denominator", -1, "normal_denominator"),
        ("realized_injected", True, "realized_injected"),
        ("feature_distribution", "other", "feature_distribution"),
        ("update_injected_state", 1, "update_injected_state"),
        ("uncertainty_mode", "per_dimension_log_variance", "uncertainty_mode"),
        ("source_revision", {"git_commit": "short", "dirty": False}, "source_revision"),
        ("score_artifact", {"path": "/absolute.npz", "sha256": "d" * 64}, "relative"),
    ],
)
def test_tuning_artifact_writer_rejects_invalid_scalar_contracts(
    tmp_path, field, value, message
):
    artifact = replace(_baseline_selection_artifact(), **{field: value})
    with pytest.raises(ValueError, match=message):
        write_tuning_selection_artifact(artifact, tmp_path / "invalid.json")


def _write_selection_payload(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("root_change", ["missing", "unknown"])
def test_tuning_artifact_reader_requires_exact_root_fields(tmp_path, root_change):
    payload = _baseline_selection_artifact().to_dict()
    if root_change == "missing":
        del payload["training_seed"]
    else:
        payload["unexpected"] = 1
    with pytest.raises(ValueError, match=root_change):
        read_tuning_selection_artifact(
            _write_selection_payload(tmp_path / "selection.json", payload)
        )


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ({"day": "tuesday", "configured_path": "data/wednesday.csv", "sha256": "b" * 64},
         "dataset_identity"),
        ({"day": "wednesday", "configured_path": "", "sha256": "b" * 64},
         "configured_path"),
        ({"day": "wednesday", "configured_path": "data/wednesday.csv", "sha256": "B" * 64},
         "SHA-256"),
    ],
)
def test_tuning_artifact_reader_validates_dataset_identity(tmp_path, identity, message):
    payload = _baseline_selection_artifact().to_dict()
    payload["dataset_identity"] = identity
    with pytest.raises(ValueError, match=message):
        read_tuning_selection_artifact(
            _write_selection_payload(tmp_path / "selection.json", payload)
        )


def test_tuning_artifact_reader_rejects_candidate_shape_and_grid_drift(tmp_path):
    payload = _baseline_selection_artifact().to_dict()
    payload["candidate_grid"][0]["unexpected"] = 1
    with pytest.raises(ValueError, match="candidate_grid row fields"):
        read_tuning_selection_artifact(
            _write_selection_payload(tmp_path / "shape.json", payload)
        )

    payload = _baseline_selection_artifact().to_dict()
    payload["thresholds"]["mu_percentile"] = 61
    payload["candidate_grid"][0]["mu_percentile"] = 61
    with pytest.raises(ValueError, match="mu_percentile"):
        read_tuning_selection_artifact(
            _write_selection_payload(tmp_path / "grid.json", payload)
        )


def test_tuning_artifact_reader_requires_selected_choice_in_candidate_grid(tmp_path):
    payload = _baseline_selection_artifact().to_dict()
    payload["candidate_grid"][0]["f1"] = 0.5
    with pytest.raises(ValueError, match="selected tuning choice"):
        read_tuning_selection_artifact(
            _write_selection_payload(tmp_path / "selection.json", payload)
        )


def test_tuning_artifact_reader_rejects_nonfinite_json_and_bad_score_digest(tmp_path):
    payload = _baseline_selection_artifact().to_dict()
    payload["thresholds"]["mu_threshold"] = float("nan")
    with pytest.raises(ValueError, match="mu_threshold"):
        read_tuning_selection_artifact(
            _write_selection_payload(tmp_path / "nan.json", payload)
        )

    payload = _baseline_selection_artifact().to_dict()
    payload["score_artifact"]["sha256"] = "D" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        read_tuning_selection_artifact(
            _write_selection_payload(tmp_path / "digest.json", payload)
        )


def test_tuning_artifact_atomic_failure_preserves_destination_and_cleans_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "selection.json"
    destination.write_text("original\n", encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(evaluation_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        write_tuning_selection_artifact(_baseline_selection_artifact(), destination)

    assert destination.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".selection.json.*.tmp")) == []


def test_baseline_tuning_and_stream_merge_use_recovered_tie_order():
    events = _events(labels=(False, True, False, True), injected=(False,) * 4)

    def candidate(checkpoint_id, epoch):
        return {
            "checkpoint_id": checkpoint_id,
            "epoch": epoch,
            "s_mu": np.zeros(4),
            "validation_s_mu": np.arange(100, dtype=float),
        }

    later = tune_baseline_threshold([candidate("later", 2)], events)
    lexical_late = tune_baseline_threshold([candidate("zeta", 1)], events)
    lexical_early = tune_baseline_threshold([candidate("alpha", 1)], events)
    merged = merge_baseline_threshold_choices([later, lexical_late, lexical_early])

    assert merged.checkpoint_id == "alpha"
    assert merged.epoch == 1
    assert merged.mu_percentile == 60
    assert len(merged.candidates) == 3 * len(BASELINE_MU_PERCENTILES)
    with pytest.raises(ValueError, match="Wednesday with 5% noise"):
        tune_baseline_threshold([candidate("wrong-day", 1)], events, day="tuesday")


def test_streamed_robust_choice_merge_equals_one_shot_and_preserves_rows():
    events = _events(labels=(False, True, False, True), injected=(False, False, True, False))
    early = {
        "checkpoint_id": "early",
        "epoch": 2,
        "s_mu": [0, 90, 0, 90],
        "s_sigma": [0, 0, 90, 0],
        "validation_s_mu": np.arange(100, dtype=float),
        "validation_s_sigma": np.arange(100, dtype=float),
    }
    late = {
        "checkpoint_id": "late",
        "epoch": 4,
        "s_mu": [0, 90, 0, 90],
        "s_sigma": [0, 0, 90, 0],
        "validation_s_mu": np.arange(200, 300, dtype=float),
        "validation_s_sigma": np.arange(200, 300, dtype=float),
    }

    one_shot = tune_fixed_threshold([early, late], events)
    merged = merge_fixed_threshold_choices([
        tune_fixed_threshold([early], events),
        tune_fixed_threshold([late], events),
    ])

    assert merged == one_shot
    assert len(merged.candidates) == 2 * 25


def test_robust_tuning_artifact_preserves_selected_threshold_pair_and_grid(tmp_path):
    events = _events(labels=(False, True, False, True), injected=(False, False, True, False))
    choice = tune_fixed_threshold(
        [{"checkpoint_id": "epoch_0002", "epoch": 2,
          "s_mu": [0, 90, 0, 90], "s_sigma": [0, 0, 90, 0],
          "validation_s_mu": np.arange(100, dtype=float),
          "validation_s_sigma": np.arange(100, dtype=float)}],
        events,
    )
    artifact = make_tuning_selection_artifact(
        choice,
        **_selection_metadata(
            uncertainty_mode="per_dimension_log_variance",
            score_artifact={"path": "scores/tuning/robust.npz", "sha256": "e" * 64},
        ),
    )
    path = tmp_path / "robust.json"
    write_tuning_selection_artifact(artifact, path)

    assert artifact.model_variant == "robust"
    assert artifact.thresholds == {
        "mu_percentile": choice.mu_percentile,
        "sigma_percentile": choice.sigma_percentile,
        "mu_threshold": choice.mu_threshold,
        "sigma_threshold": choice.sigma_threshold,
        "rejected_count": choice.rejected_count,
    }
    assert artifact.candidate_grid == choice.candidates
    assert read_tuning_selection_artifact(path) == artifact


def test_fixed_threshold_tuning_grid_tie_break_and_transfer():
    events = _events(labels=(False, True, False, True), injected=(False, False, True, False))
    validation_mu = np.arange(100, dtype=float)
    validation_sigma = np.arange(100, dtype=float)
    choice = tune_fixed_threshold(
        [
            {"checkpoint_id": "late", "epoch": 4, "s_mu": [0, 90, 0, 90],
             "s_sigma": [0, 0, 90, 0], "validation_s_mu": np.arange(200, 300),
             "validation_s_sigma": np.arange(200, 300)},
            {"checkpoint_id": "early", "epoch": 2, "s_mu": [0, 90, 0, 90],
             "s_sigma": [0, 0, 90, 0], "validation_s_mu": validation_mu,
             "validation_s_sigma": validation_sigma},
        ],
        events,
    )
    assert choice.checkpoint_id == "early"
    assert choice.mu_percentile in [86, 88, 90, 92, 94]
    assert choice.sigma_percentile in [75, 77, 79, 81, 83]
    reporting = evaluate_fixed_threshold(
        events, [0, 3, 1, 2], [0, 0, 90, 0], choice,
        **_metadata(day="tuesday", noise_ratio=0.1, checkpoint_id="early")
    )
    assert reporting.thresholds["mu_threshold"] == choice.mu_threshold
    assert reporting.thresholds["sigma_threshold"] == choice.sigma_threshold
    assert reporting.roc_auc == 1.0


def test_aggregation_retains_raw_values_and_uses_population_std():
    base = evaluate_baseline(_events(), [0, 3, 1, 2], **_metadata(model_variant="baseline", repeat=0, noise_seed=8))
    other = EvaluationRecord.from_dict({**base.to_dict(), "repeat": 1, "noise_seed": 9, "roc_auc": 0.5})
    aggregate = aggregate_records([base, other], expected_repeats=2)
    assert aggregate.values == [1.0, 0.5]
    assert aggregate.mean == 0.75
    assert aggregate.std == 0.25
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_records([base, base], expected_repeats=2)
    incompatible = EvaluationRecord.from_dict({**other.to_dict(), "manifest": {"revision": "other"}})
    with pytest.raises(ValueError, match="compatible"):
        aggregate_records([base, incompatible], expected_repeats=2)
    duplicate_seed = EvaluationRecord.from_dict(
        {**other.to_dict(), "noise_seed": base.noise_seed}
    )
    with pytest.raises(ValueError, match="noise seed"):
        aggregate_records([base, duplicate_seed], expected_repeats=2)


def test_aggregate_artifact_round_trip_preserves_values(tmp_path):
    first = evaluate_baseline(
        _events(), [0, 3, 1, 2],
        **_metadata(model_variant="baseline", day="tuesday", noise_ratio=0.1,
                    repeat=0, noise_seed=8, uncertainty_mode=None),
    )
    second = EvaluationRecord.from_dict(
        {**first.to_dict(), "repeat": 1, "noise_seed": 4, "roc_auc": 0.5}
    )
    aggregate = aggregate_records([second, first], expected_repeats=2)
    path = tmp_path / "baseline_aggregate.json"

    assert write_aggregate_record(aggregate, path) is None
    loaded = read_aggregate_record(path)

    assert loaded == aggregate
    assert loaded.noise_seeds == [8, 4]


def test_aggregate_reader_rejects_schema_and_scientific_inconsistency(tmp_path):
    valid = {
        "schema_version": 1,
        "group": {
            "protocol": "baseline", "model_variant": "baseline", "day": "tuesday",
            "noise_ratio": 0.1, "checkpoint_id": "checkpoint", "training_seed": 3,
            "update_injected_state": True, "uncertainty_mode": None,
        },
        "values": [1.0, 0.5], "mean": 0.75, "std": 0.25,
        "repeats": [0, 1], "noise_seeds": [8, 4],
    }
    invalid_payloads = [
        {**valid, "schema_version": 2},
        {**valid, "schema_version": True},
        {**valid, "unexpected": 1},
        {key: value for key, value in valid.items() if key != "mean"},
        {**valid, "group": []},
        {**valid, "group": {**valid["group"], "extra": "field"}},
        {**valid, "group": {**valid["group"], "training_seed": True}},
        {**valid, "values": [True, 0.5]},
        {**valid, "values": [1.1, 0.5]},
        {**valid, "values": [float("nan"), 0.5]},
        {**valid, "repeats": [0]},
        {**valid, "repeats": [1, 0]},
        {**valid, "noise_seeds": [8]},
        {**valid, "noise_seeds": [8, 8]},
        {**valid, "mean": 0.7},
        {**valid, "std": 0.2},
    ]
    path = tmp_path / "aggregate.json"
    for payload in invalid_payloads:
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            read_aggregate_record(path)


def test_aggregate_writer_preserves_existing_target_if_replace_fails(tmp_path, monkeypatch):
    record = AggregateRecord(
        schema_version=1,
        group={
            "protocol": "baseline", "model_variant": "baseline", "day": "tuesday",
            "noise_ratio": 0.1, "checkpoint_id": "checkpoint", "training_seed": 3,
            "update_injected_state": True, "uncertainty_mode": None,
        },
        values=[0.75], mean=0.75, std=0.0, repeats=[0], noise_seeds=[8],
    )
    path = tmp_path / "aggregate.json"
    path.write_text("existing\n", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(evaluation_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_aggregate_record(record, path)

    assert path.read_text(encoding="utf-8") == "existing\n"
    assert list(tmp_path.iterdir()) == [path]


def test_records_reject_nonfinite_or_invalid_fields(tmp_path):
    record = evaluate_baseline(_events(), [0, 3, 1, 2], **_metadata(model_variant="baseline"))
    invalid = EvaluationRecord.from_dict({**record.to_dict(), "roc_auc": float("nan")})
    with pytest.raises(ValueError, match="roc_auc"):
        write_result_record(invalid, tmp_path / "invalid.json")
    invalid_repeat = EvaluationRecord.from_dict({**record.to_dict(), "repeat": -1})
    with pytest.raises(ValueError, match="repeat"):
        aggregate_records([invalid_repeat], expected_repeats=1)


def test_score_and_result_artifacts_round_trip_and_detect_tampering(tmp_path):
    artifact = ScoreArtifact(
        t=np.arange(4), is_attack=np.array([0, 1, 0, 1], dtype=bool),
        is_injected=np.array([0, 0, 1, 0], dtype=bool), s_mu=np.arange(4.0),
        s_sigma=np.arange(4.0) + 5, metadata={"day": "wednesday"},
    )
    score_path = tmp_path / "scores.npz"
    digest = write_score_artifact(artifact, score_path)
    loaded = read_score_artifact(score_path, expected_sha256=digest)
    assert np.array_equal(loaded.s_sigma, artifact.s_sigma)

    record = evaluate_baseline(_events(), [0, 3, 1, 2], **_metadata(model_variant="baseline"))
    result_path = tmp_path / "record.json"
    write_result_record(record, result_path)
    assert read_result_record(result_path) == record
    payload = bytearray(score_path.read_bytes()); payload[-1] ^= 1; score_path.write_bytes(payload)
    with pytest.raises(ValueError, match="integrity"):
        read_score_artifact(score_path, expected_sha256=digest)
    assert json.loads(result_path.read_text())["schema_version"] == 1


def test_score_artifact_rejects_invalid_timestamp_dtype_and_chronology(tmp_path):
    common = dict(is_attack=np.array([False, True]), is_injected=np.array([False, False]),
                  s_mu=np.array([0.0, 1.0]), s_sigma=None, metadata={})
    with pytest.raises(ValueError, match="integer"):
        write_score_artifact(ScoreArtifact(t=np.array([0.0, 1.0]), **common), tmp_path / "float.npz")
    with pytest.raises(ValueError, match="nondecreasing"):
        write_score_artifact(ScoreArtifact(t=np.array([2, 1]), **common), tmp_path / "reverse.npz")

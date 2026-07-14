"""Paper-facing scoring, rejection, aggregation, and result artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import EventStream


SCHEMA_VERSION = 1
BASELINE_MU_PERCENTILES = tuple(range(60, 102, 2))
MU_PERCENTILES = (86, 88, 90, 92, 94)
SIGMA_PERCENTILES = (75, 77, 79, 81, 83)
_AGGREGATE_GROUP_FIELDS = {
    "protocol", "model_variant", "day", "noise_ratio", "checkpoint_id",
    "training_seed", "update_injected_state", "uncertainty_mode",
}


@dataclass(frozen=True)
class EvaluationRecord:
    schema_version: int
    protocol: str
    model_variant: str
    day: str
    noise_ratio: float
    repeat: int
    training_seed: int
    noise_seed: int
    checkpoint_id: str
    update_injected_state: bool
    uncertainty_mode: str | None
    roc_auc: float
    counts: dict[str, int]
    thresholds: dict[str, Any]
    supporting_metrics: dict[str, Any]
    semantics: dict[str, Any]
    manifest: dict[str, Any]
    score_artifact: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationRecord":
        data = dict(value)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported evaluation schema_version")
        return cls(**data)


@dataclass(frozen=True)
class BaselineThresholdChoice:
    schema_version: int
    checkpoint_id: str
    epoch: int
    mu_percentile: int
    mu_threshold: float
    f1: float
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ThresholdChoice:
    schema_version: int
    checkpoint_id: str
    epoch: int
    mu_percentile: int
    sigma_percentile: int
    mu_threshold: float
    sigma_threshold: float
    f1: float
    rejected_count: int
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TuningSelectionArtifact:
    schema_version: int
    model_variant: str
    selection_policy: str
    training_seed: int
    tuning_day: str
    tuning_noise_ratio: float
    tuning_noise_seed: int
    selected_epoch: int
    selected_logical_checkpoint_id: str
    selected_checkpoint_sha256: str
    f1: float
    thresholds: dict[str, Any]
    candidate_grid: tuple[dict[str, Any], ...]
    dataset_identity: dict[str, Any]
    normal_denominator: int
    realized_injected: int
    feature_distribution: str
    update_injected_state: bool
    uncertainty_mode: str | None
    source_revision: dict[str, Any]
    score_artifact: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AggregateRecord:
    schema_version: int
    group: dict[str, Any]
    values: list[float]
    mean: float
    std: float
    repeats: list[int]
    noise_seeds: list[int]


@dataclass(frozen=True)
class ScoreArtifact:
    t: np.ndarray
    is_attack: np.ndarray
    is_injected: np.ndarray
    s_mu: np.ndarray
    s_sigma: np.ndarray | None
    metadata: dict[str, Any]


def attack_targets(events: EventStream) -> np.ndarray:
    """Return attack-positive labels; benign and injected events are negative."""
    labels = np.asarray(events.is_attack.detach().cpu().tolist(), dtype=bool)
    if np.any(labels & np.asarray(events.is_injected.detach().cpu().tolist(), dtype=bool)):
        raise ValueError("injected events cannot be attack targets")
    return labels


def evaluate_baseline(events: EventStream, s_mu: Sequence[float], **metadata: Any) -> EvaluationRecord:
    scores, labels = _scores_and_labels(events, s_mu)
    return _record("baseline", labels, events, _roc_auc(labels, scores), {}, {}, metadata)


def evaluate_paper_recovered(
    events: EventStream,
    s_mu: Sequence[float],
    s_sigma: Sequence[float],
    **metadata: Any,
) -> EvaluationRecord:
    mean_scores, labels = _scores_and_labels(events, s_mu)
    if np.any(mean_scores < 0):
        raise ValueError("s_mu must be nonnegative for paper_recovered rejection")
    sigma = _array(s_sigma, len(events), "s_sigma")
    candidates = np.unique(sigma[(sigma > 5.0) & (sigma < 25.0)])
    if not candidates.size:
        raise ValueError("no observed s_sigma thresholds satisfy strict 5 < t < 25")
    aucs = _paper_recovered_aucs(labels, mean_scores, sigma)
    thresholds = {
        "candidate_rule": "unique_observed_s_sigma_strict_5_25",
        "candidates": candidates.tolist(),
        "auc_by_threshold": aucs,
        "rejection_rule": "s_sigma > threshold",
        "rejected_score": -100.0,
        "reduction": "unweighted_arithmetic_mean",
    }
    return _record("paper_recovered", labels, events, float(np.mean(aucs)), thresholds, {}, metadata)


class _FenwickCounts:
    """Prefix counts over compressed score ranks for one sweep."""

    def __init__(self, size: int) -> None:
        self._tree = [0] * (size + 1)

    def add(self, index: int) -> None:
        index += 1
        while index < len(self._tree):
            self._tree[index] += 1
            index += index & -index

    def prefix(self, stop: int) -> int:
        """Return the count at ranks strictly below ``stop``."""
        total = 0
        while stop:
            total += self._tree[stop]
            stop -= stop & -stop
        return total


def _paper_recovered_aucs(
    labels: np.ndarray, mean_scores: np.ndarray, sigma: np.ndarray
) -> list[float]:
    """Compute every recovered rejection AUC in one incremental rank sweep."""
    _, ranks = np.unique(mean_scores, return_inverse=True)
    rank_count = int(ranks.max()) + 1
    positive_tree = _FenwickCounts(rank_count)
    negative_tree = _FenwickCounts(rank_count)
    positive_at_rank = [0] * rank_count
    negative_at_rank = [0] * rank_count

    positives = int(labels.sum())
    negatives = labels.size - positives
    retained_positives = 0
    retained_negatives = 0
    retained_u_twice = 0
    aucs: list[float] = []

    order = np.argsort(sigma, kind="stable")
    start = 0
    while start < order.size:
        threshold = sigma[order[start]]
        if threshold >= 25.0:
            break
        stop = start + 1
        while stop < order.size and sigma[order[stop]] == threshold:
            stop += 1

        for event_index in order[start:stop]:
            rank = int(ranks[event_index])
            if labels[event_index]:
                retained_u_twice += 2 * negative_tree.prefix(rank)
                retained_u_twice += negative_at_rank[rank]
                positive_tree.add(rank)
                positive_at_rank[rank] += 1
                retained_positives += 1
            else:
                positives_at_or_below = positive_tree.prefix(rank + 1)
                retained_u_twice += 2 * (retained_positives - positives_at_or_below)
                retained_u_twice += positive_at_rank[rank]
                negative_tree.add(rank)
                negative_at_rank[rank] += 1
                retained_negatives += 1

        if threshold > 5.0:
            rejected_positives = positives - retained_positives
            rejected_negatives = negatives - retained_negatives
            total_u_twice = retained_u_twice
            total_u_twice += 2 * retained_positives * rejected_negatives
            total_u_twice += rejected_positives * rejected_negatives
            aucs.append(total_u_twice / (2 * positives * negatives))
        start = stop

    return aucs


def tune_baseline_threshold(
    checkpoint_candidates: Sequence[Mapping[str, Any]],
    wednesday_events: EventStream,
    *,
    day: str = "wednesday",
    noise_ratio: float = 0.05,
) -> BaselineThresholdChoice:
    """Select a baseline epoch and validation-derived threshold on Wednesday 5%."""
    _validate_tuning_context(day, noise_ratio)
    if not checkpoint_candidates:
        raise ValueError("at least one checkpoint candidate is required")
    labels = attack_targets(wednesday_events)
    _require_binary(labels)
    rows: list[dict[str, Any]] = []
    for candidate in checkpoint_candidates:
        checkpoint_id = str(candidate["checkpoint_id"])
        epoch = int(candidate["epoch"])
        mean_scores = _array(candidate["s_mu"], len(wednesday_events), "s_mu")
        validation_scores = _finite_1d(candidate["validation_s_mu"], "validation_s_mu")
        for percentile in BASELINE_MU_PERCENTILES:
            threshold = float(np.percentile(validation_scores, percentile))
            rows.append({
                "checkpoint_id": checkpoint_id,
                "epoch": epoch,
                "mu_percentile": percentile,
                "mu_threshold": threshold,
                "f1": _f1(labels, mean_scores >= threshold),
            })
    return _choose_baseline_threshold(rows)


def merge_baseline_threshold_choices(
    choices: Sequence[BaselineThresholdChoice],
) -> BaselineThresholdChoice:
    """Merge streamed baseline choices without discarding candidate rows."""
    if not choices:
        raise ValueError("at least one baseline threshold choice is required")
    if any(choice.schema_version != SCHEMA_VERSION for choice in choices):
        raise ValueError("unsupported baseline threshold choice schema_version")
    return _choose_baseline_threshold(
        [row for choice in choices for row in choice.candidates]
    )


def tune_fixed_threshold(
    checkpoint_candidates: Sequence[Mapping[str, Any]],
    wednesday_events: EventStream,
    *,
    day: str = "wednesday",
    noise_ratio: float = 0.05,
) -> ThresholdChoice:
    """Tune the recovered percentile-pair grid on Wednesday 5% only."""
    _validate_tuning_context(day, noise_ratio)
    if not checkpoint_candidates:
        raise ValueError("at least one checkpoint candidate is required")
    labels = attack_targets(wednesday_events)
    _require_binary(labels)
    rows: list[dict[str, Any]] = []
    for candidate in checkpoint_candidates:
        checkpoint_id = str(candidate["checkpoint_id"])
        epoch = int(candidate["epoch"])
        mean_scores = _array(candidate["s_mu"], len(wednesday_events), "s_mu")
        sigma = _array(candidate["s_sigma"], len(wednesday_events), "s_sigma")
        val_mu = _finite_1d(candidate["validation_s_mu"], "validation_s_mu")
        val_sigma = _finite_1d(candidate["validation_s_sigma"], "validation_s_sigma")
        for mu_percentile in MU_PERCENTILES:
            mu_threshold = float(np.percentile(val_mu, mu_percentile))
            for sigma_percentile in SIGMA_PERCENTILES:
                sigma_threshold = float(np.percentile(val_sigma, sigma_percentile))
                rejected = sigma > sigma_threshold
                predictions = (mean_scores >= mu_threshold) & ~rejected
                rows.append({
                    "checkpoint_id": checkpoint_id, "epoch": epoch,
                    "mu_percentile": mu_percentile, "sigma_percentile": sigma_percentile,
                    "mu_threshold": mu_threshold, "sigma_threshold": sigma_threshold,
                    "f1": _f1(labels, predictions),
                    "rejected_count": int(rejected.sum()),
                })
    return _choose_fixed_threshold(rows)


def merge_fixed_threshold_choices(
    choices: Sequence[ThresholdChoice],
) -> ThresholdChoice:
    """Merge streamed robust choices exactly as if candidates were tuned together."""
    if not choices:
        raise ValueError("at least one fixed threshold choice is required")
    if any(choice.schema_version != SCHEMA_VERSION for choice in choices):
        raise ValueError("unsupported fixed threshold choice schema_version")
    return _choose_fixed_threshold([row for choice in choices for row in choice.candidates])


def make_tuning_selection_artifact(
    choice: BaselineThresholdChoice | ThresholdChoice,
    *,
    training_seed: int,
    tuning_noise_seed: int,
    selected_checkpoint_sha256: str,
    dataset_identity: Mapping[str, Any],
    normal_denominator: int,
    realized_injected: int,
    feature_distribution: str,
    update_injected_state: bool,
    uncertainty_mode: str | None,
    source_revision: Mapping[str, Any],
    score_artifact: Mapping[str, str],
) -> TuningSelectionArtifact:
    """Serialize one already-selected Wednesday choice without re-ranking it."""
    if choice.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported threshold choice schema_version")
    if isinstance(choice, BaselineThresholdChoice):
        model_variant = "baseline"
        thresholds = {
            "mu_percentile": choice.mu_percentile,
            "mu_threshold": choice.mu_threshold,
        }
    elif isinstance(choice, ThresholdChoice):
        model_variant = "robust"
        thresholds = {
            "mu_percentile": choice.mu_percentile,
            "sigma_percentile": choice.sigma_percentile,
            "mu_threshold": choice.mu_threshold,
            "sigma_threshold": choice.sigma_threshold,
            "rejected_count": choice.rejected_count,
        }
    else:
        raise TypeError("choice must be a BaselineThresholdChoice or ThresholdChoice")
    artifact = TuningSelectionArtifact(
        schema_version=SCHEMA_VERSION,
        model_variant=model_variant,
        selection_policy="wednesday_f1",
        training_seed=training_seed,
        tuning_day="wednesday",
        tuning_noise_ratio=0.05,
        tuning_noise_seed=tuning_noise_seed,
        selected_epoch=choice.epoch,
        selected_logical_checkpoint_id=choice.checkpoint_id,
        selected_checkpoint_sha256=selected_checkpoint_sha256,
        f1=choice.f1,
        thresholds=thresholds,
        candidate_grid=tuple(dict(row) for row in choice.candidates),
        dataset_identity=dict(dataset_identity),
        normal_denominator=normal_denominator,
        realized_injected=realized_injected,
        feature_distribution=feature_distribution,
        update_injected_state=update_injected_state,
        uncertainty_mode=uncertainty_mode,
        source_revision=dict(source_revision),
        score_artifact=dict(score_artifact),
    )
    _validate_tuning_selection_artifact(artifact)
    return artifact


def _validate_tuning_context(day: str, noise_ratio: float) -> None:
    if day.lower() != "wednesday" or not np.isclose(noise_ratio, 0.05):
        raise ValueError("thresholds may be tuned only on Wednesday with 5% noise")


def _baseline_choice_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (-row["f1"], row["epoch"], row["mu_percentile"], row["checkpoint_id"])


def _fixed_choice_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (-row["f1"], row["rejected_count"], row["epoch"], row["mu_percentile"],
            -row["sigma_percentile"], row["checkpoint_id"])


def _choose_baseline_threshold(rows: Sequence[dict[str, Any]]) -> BaselineThresholdChoice:
    best = min(rows, key=_baseline_choice_key)
    return BaselineThresholdChoice(
        schema_version=SCHEMA_VERSION, candidates=tuple(rows), **best
    )


def _choose_fixed_threshold(rows: Sequence[dict[str, Any]]) -> ThresholdChoice:
    best = min(rows, key=_fixed_choice_key)
    return ThresholdChoice(schema_version=SCHEMA_VERSION, candidates=tuple(rows), **best)


def evaluate_fixed_threshold(
    events: EventStream,
    s_mu: Sequence[float],
    s_sigma: Sequence[float],
    choice: ThresholdChoice,
    **metadata: Any,
) -> EvaluationRecord:
    if metadata.get("checkpoint_id") != choice.checkpoint_id:
        raise ValueError("fixed-threshold evaluation must use the tuned checkpoint unchanged")
    mean_scores, labels = _scores_and_labels(events, s_mu)
    sigma = _array(s_sigma, len(events), "s_sigma")
    rejected = sigma > choice.sigma_threshold
    transformed = np.where(rejected, -100.0, mean_scores)
    predictions = (mean_scores >= choice.mu_threshold) & ~rejected
    matrix = _confusion(labels, predictions)
    supporting = {"f1": _f1(labels, predictions),
                  "confusion_matrix": matrix.tolist(), "rejected_count": int(rejected.sum())}
    thresholds = {"checkpoint_id": choice.checkpoint_id, "epoch": choice.epoch,
                  "mu_percentile": choice.mu_percentile, "sigma_percentile": choice.sigma_percentile,
                  "mu_threshold": choice.mu_threshold, "sigma_threshold": choice.sigma_threshold,
                  "rejection_rule": "s_sigma > sigma_threshold", "rejected_score": -100.0}
    return _record("fixed_threshold", labels, events, _roc_auc(labels, transformed),
                   thresholds, supporting, metadata)


def aggregate_records(records: Sequence[EvaluationRecord], *, expected_repeats: int = 5) -> AggregateRecord:
    if not records:
        raise ValueError("cannot aggregate no records")
    keys = ("protocol", "model_variant", "day", "noise_ratio", "checkpoint_id",
            "training_seed", "update_injected_state", "uncertainty_mode")
    group = {key: getattr(records[0], key) for key in keys}
    for record in records:
        _validate_record(record)
        if record.schema_version != SCHEMA_VERSION or any(getattr(record, key) != value for key, value in group.items()):
            raise ValueError("records are not a compatible aggregation group")
        if record.semantics != records[0].semantics or record.manifest != records[0].manifest:
            raise ValueError("records are not a compatible aggregation group")
    ordered = sorted(records, key=lambda item: item.repeat)
    repeats = [item.repeat for item in ordered]
    if len(set(repeats)) != len(repeats):
        raise ValueError("duplicate repeat index")
    if repeats != list(range(expected_repeats)):
        raise ValueError(f"expected repeat indices 0..{expected_repeats - 1}")
    noise_seeds = [item.noise_seed for item in ordered]
    if len(set(noise_seeds)) != len(noise_seeds):
        raise ValueError("duplicate noise seed")
    values = [item.roc_auc for item in ordered]
    return AggregateRecord(SCHEMA_VERSION, group, values, float(np.mean(values)),
                           float(np.std(values, ddof=0)), repeats,
                           noise_seeds)


def write_aggregate_record(record: AggregateRecord, path: str | Path) -> None:
    """Atomically write one validated aggregate without changing its schema."""
    _validate_aggregate_record(record)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(asdict(record)) + "\n"; temporary = _temporary(path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_aggregate_record(path: str | Path) -> AggregateRecord:
    """Read and scientifically validate one schema-v1 aggregate JSON file."""
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("aggregate record must be a JSON object")
    expected = set(AggregateRecord.__dataclass_fields__)
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing:
        raise ValueError(f"missing aggregate fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown aggregate fields: {', '.join(sorted(unknown))}")
    record = AggregateRecord(**value)
    _validate_aggregate_record(record)
    return record


def write_score_artifact(artifact: ScoreArtifact, path: str | Path) -> str:
    path = Path(path); _validate_artifact(artifact); path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _canonical_json({"schema_version": SCHEMA_VERSION, **artifact.metadata})
    arrays: dict[str, Any] = {"t": artifact.t, "is_attack": artifact.is_attack,
                              "is_injected": artifact.is_injected, "s_mu": artifact.s_mu,
                              "metadata_json": np.array(metadata)}
    if artifact.s_sigma is not None:
        arrays["s_sigma"] = artifact.s_sigma
    temporary = _temporary(path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def read_score_artifact(path: str | Path, *, expected_sha256: str | None = None) -> ScoreArtifact:
    path = Path(path)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("score artifact integrity check failed")
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("unsupported score artifact schema_version")
        artifact = ScoreArtifact(data["t"].copy(), data["is_attack"].copy(),
                                 data["is_injected"].copy(), data["s_mu"].copy(),
                                 data["s_sigma"].copy() if "s_sigma" in data else None, metadata)
    _validate_artifact(artifact)
    return artifact


def write_result_record(record: EvaluationRecord, path: str | Path) -> None:
    _validate_record(record)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(record.to_dict()) + "\n"; temporary = _temporary(path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_result_record(path: str | Path) -> EvaluationRecord:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("result record must be a JSON object")
    record = EvaluationRecord.from_dict(value)
    _validate_record(record)
    return record


def write_tuning_selection_artifact(
    artifact: TuningSelectionArtifact, path: str | Path
) -> None:
    """Atomically write one canonical tuning-selection JSON artifact."""
    _validate_tuning_selection_artifact(artifact)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(artifact.to_dict()) + "\n"; temporary = _temporary(path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_tuning_selection_artifact(path: str | Path) -> TuningSelectionArtifact:
    """Read and validate a schema-v1 tuning-selection artifact."""
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("tuning selection artifact must be a JSON object")
    expected = set(TuningSelectionArtifact.__dataclass_fields__)
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing:
        raise ValueError(f"missing tuning selection fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown tuning selection fields: {', '.join(sorted(unknown))}")
    if not isinstance(value["candidate_grid"], list):
        raise ValueError("candidate_grid must be a JSON array")
    artifact = TuningSelectionArtifact(
        **{**value, "candidate_grid": tuple(value["candidate_grid"])}
    )
    _validate_tuning_selection_artifact(artifact)
    return artifact


def _record(protocol: str, labels: np.ndarray, events: EventStream, auc: float,
            thresholds: dict[str, Any], supporting: dict[str, Any], metadata: Mapping[str, Any]) -> EvaluationRecord:
    required = {"model_variant", "day", "noise_ratio", "repeat", "training_seed", "noise_seed",
                "checkpoint_id", "update_injected_state", "uncertainty_mode"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"missing evaluation metadata: {', '.join(sorted(missing))}")
    injected = np.asarray(events.is_injected.detach().cpu().tolist(), dtype=bool)
    benign = int((~labels & ~injected).sum())
    injected_count = int(injected.sum())
    counts = {"attack": int(labels.sum()), "injected": injected_count,
              "benign": benign, "normal_denominator": benign,
              "realized_injected": injected_count}
    selected = {key: metadata[key] for key in required}
    manifest = {key: value for key, value in metadata.items()
                if key not in required and key != "score_artifact"}
    return EvaluationRecord(SCHEMA_VERSION, protocol, counts=counts, roc_auc=auc,
                            thresholds=thresholds, supporting_metrics=supporting,
                            semantics={"target": "attack_vs_benign_and_injected",
                                       "positive_label": "attack"},
                            manifest=manifest,
                            score_artifact=metadata.get("score_artifact"), **selected)


def _scores_and_labels(events: EventStream, scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    labels = attack_targets(events); _require_binary(labels)
    return _array(scores, len(events), "s_mu"), labels


def _require_binary(labels: np.ndarray) -> None:
    if np.unique(labels).size != 2:
        raise ValueError("ROC-AUC evaluation requires both attack and non-attack events")


def _array(value: Sequence[float], length: int, name: str) -> np.ndarray:
    result = _finite_1d(value, name)
    if result.size != length:
        raise ValueError(f"{name} must contain one value per event")
    return result


def _finite_1d(value: Sequence[float], name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        value = value.detach().cpu().tolist()  # type: ignore[union-attr]
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def _validate_artifact(artifact: ScoreArtifact) -> None:
    arrays = [np.asarray(artifact.t), np.asarray(artifact.is_attack),
              np.asarray(artifact.is_injected), np.asarray(artifact.s_mu)]
    if artifact.s_sigma is not None: arrays.append(np.asarray(artifact.s_sigma))
    if any(array.ndim != 1 for array in arrays) or len({array.size for array in arrays}) != 1:
        raise ValueError("score artifact arrays must be one-dimensional and equally sized")
    timestamps = np.asarray(artifact.t)
    if not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError("score artifact timestamps must use an integer dtype")
    if timestamps.size > 1 and np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("score artifact timestamps must be nondecreasing")
    if np.asarray(artifact.is_attack).dtype != np.bool_ or np.asarray(artifact.is_injected).dtype != np.bool_:
        raise ValueError("score artifact labels must be boolean")
    if np.any(np.asarray(artifact.is_attack) & np.asarray(artifact.is_injected)):
        raise ValueError("injected events cannot be attack targets")
    if not np.issubdtype(np.asarray(artifact.s_mu).dtype, np.number) or (
        artifact.s_sigma is not None and not np.issubdtype(np.asarray(artifact.s_sigma).dtype, np.number)
    ):
        raise ValueError("score artifact scores must be numeric")
    if not np.all(np.isfinite(artifact.s_mu)) or (artifact.s_sigma is not None and not np.all(np.isfinite(artifact.s_sigma))):
        raise ValueError("score artifact scores must be finite")
    _canonical_json(artifact.metadata)


def _canonical_json(value: Mapping[str, Any]) -> str:
    try: return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error: raise ValueError(f"metadata must be JSON-compatible: {error}") from error


def _validate_record(record: EvaluationRecord) -> None:
    if record.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported evaluation schema_version")
    if record.protocol not in {"baseline", "paper_recovered", "fixed_threshold"}:
        raise ValueError("unsupported evaluation protocol")
    if not record.model_variant or not record.day or not record.checkpoint_id:
        raise ValueError("model_variant, day, and checkpoint_id must be nonempty")
    if not np.isfinite(record.roc_auc) or not 0.0 <= record.roc_auc <= 1.0:
        raise ValueError("roc_auc must be finite and between zero and one")
    if record.repeat < 0:
        raise ValueError("repeat must be nonnegative")
    if not np.isfinite(record.noise_ratio) or record.noise_ratio < 0:
        raise ValueError("noise_ratio must be finite and nonnegative")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in record.counts.values()):
        raise ValueError("record counts must be nonnegative integers")
    for name, value in (("thresholds", record.thresholds),
                        ("supporting_metrics", record.supporting_metrics),
                        ("semantics", record.semantics), ("manifest", record.manifest)):
        try:
            _canonical_json(value)
        except ValueError as error:
            raise ValueError(f"{name} must be JSON-compatible: {error}") from error
    _canonical_json(record.to_dict())


def _validate_aggregate_record(record: AggregateRecord) -> None:
    if (not isinstance(record.schema_version, int)
            or isinstance(record.schema_version, bool)
            or record.schema_version != SCHEMA_VERSION):
        raise ValueError("unsupported aggregate schema_version")
    if not isinstance(record.group, dict) or set(record.group) != _AGGREGATE_GROUP_FIELDS:
        fields = ", ".join(sorted(_AGGREGATE_GROUP_FIELDS))
        raise ValueError(f"aggregate group fields must be exactly {fields}")
    group = record.group
    if group["protocol"] not in {"baseline", "paper_recovered", "fixed_threshold"}:
        raise ValueError("unsupported aggregate protocol")
    if group["model_variant"] not in {"baseline", "robust"}:
        raise ValueError("unsupported aggregate model_variant")
    for name in ("day", "checkpoint_id"):
        if not isinstance(group[name], str) or not group[name].strip():
            raise ValueError(f"aggregate group {name} must be nonempty")
    ratio = group["noise_ratio"]
    if (not isinstance(ratio, (int, float)) or isinstance(ratio, bool)
            or not np.isfinite(ratio) or ratio < 0):
        raise ValueError("aggregate group noise_ratio must be finite and nonnegative")
    if not isinstance(group["training_seed"], int) or isinstance(group["training_seed"], bool):
        raise ValueError("aggregate group training_seed must be an integer")
    if not isinstance(group["update_injected_state"], bool):
        raise ValueError("aggregate group update_injected_state must be boolean")
    uncertainty = group["uncertainty_mode"]
    if uncertainty is not None and not isinstance(uncertainty, str):
        raise ValueError("aggregate group uncertainty_mode must be a string or null")
    _canonical_json(group)

    if not isinstance(record.values, list) or not record.values:
        raise ValueError("aggregate values must be a nonempty JSON array")
    for value in record.values:
        _bounded_score(value, "aggregate value")
    _bounded_score(record.mean, "aggregate mean")
    _bounded_score(record.std, "aggregate std")

    length = len(record.values)
    for name, items in (("repeats", record.repeats), ("noise_seeds", record.noise_seeds)):
        if not isinstance(items, list) or len(items) != length:
            raise ValueError(f"aggregate {name} must align one-to-one with values")
        if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in items):
            raise ValueError(f"aggregate {name} must contain nonnegative integers")
    if record.repeats != list(range(length)):
        raise ValueError("aggregate repeats must be ordered contiguously from zero")
    if len(set(record.noise_seeds)) != length:
        raise ValueError("aggregate noise_seeds must be distinct")

    expected_mean = float(np.mean(record.values))
    expected_std = float(np.std(record.values, ddof=0))
    if not np.isclose(record.mean, expected_mean, rtol=1e-12, atol=1e-15):
        raise ValueError("aggregate mean disagrees with raw values")
    if not np.isclose(record.std, expected_std, rtol=1e-12, atol=1e-15):
        raise ValueError("aggregate std disagrees with raw values")


def _validate_tuning_selection_artifact(artifact: TuningSelectionArtifact) -> None:
    if artifact.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported tuning selection schema_version")
    if artifact.model_variant not in {"baseline", "robust"}:
        raise ValueError("unsupported tuning selection model_variant")
    if artifact.selection_policy != "wednesday_f1":
        raise ValueError("unsupported tuning selection policy")
    valid_ratio = (
        isinstance(artifact.tuning_noise_ratio, (int, float))
        and not isinstance(artifact.tuning_noise_ratio, bool)
        and np.isclose(artifact.tuning_noise_ratio, 0.05)
    )
    if artifact.tuning_day != "wednesday" or not valid_ratio:
        raise ValueError("tuning selection must use Wednesday with 5% noise")
    _nonnegative_integer(artifact.training_seed, "training_seed")
    _nonnegative_integer(artifact.tuning_noise_seed, "tuning_noise_seed")
    if not isinstance(artifact.selected_epoch, int) or isinstance(
        artifact.selected_epoch, bool
    ) or artifact.selected_epoch <= 0:
        raise ValueError("selected_epoch must be a positive integer")
    if not isinstance(artifact.selected_logical_checkpoint_id, str) or not (
        artifact.selected_logical_checkpoint_id.strip()
    ):
        raise ValueError("selected logical checkpoint ID must be nonempty")
    _validate_digest(artifact.selected_checkpoint_sha256, "selected checkpoint SHA-256")
    _bounded_score(artifact.f1, "f1")
    _nonnegative_integer(artifact.normal_denominator, "normal_denominator")
    _nonnegative_integer(artifact.realized_injected, "realized_injected")
    if artifact.feature_distribution not in {"paper_covariance_5", "historical_scale_5"}:
        raise ValueError("unsupported feature_distribution")
    if not isinstance(artifact.update_injected_state, bool):
        raise ValueError("update_injected_state must be boolean")
    if artifact.model_variant == "baseline":
        if artifact.uncertainty_mode is not None:
            raise ValueError("baseline tuning selection uncertainty_mode must be null")
    elif artifact.uncertainty_mode not in {
        "per_dimension_log_variance", "historical_shared_per_node"
    }:
        raise ValueError("unsupported robust tuning selection uncertainty_mode")
    _validate_source_revision(artifact.source_revision)
    _validate_score_reference(artifact.score_artifact)
    _validate_dataset_identity(artifact.dataset_identity, artifact.tuning_day)
    if not isinstance(artifact.candidate_grid, tuple) or not artifact.candidate_grid:
        raise ValueError("candidate_grid must be a nonempty tuple")
    _validate_selection_thresholds(artifact.model_variant, artifact.thresholds)
    for row in artifact.candidate_grid:
        _validate_candidate_row(artifact.model_variant, row)
    selected = {
        "checkpoint_id": artifact.selected_logical_checkpoint_id,
        "epoch": artifact.selected_epoch,
        "f1": artifact.f1,
        **artifact.thresholds,
    }
    if sum(row == selected for row in artifact.candidate_grid) != 1:
        raise ValueError("selected tuning choice must appear exactly once in candidate_grid")
    _canonical_json(artifact.to_dict())


def _nonnegative_integer(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")


def _validate_source_revision(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"git_commit", "dirty"}:
        raise ValueError("source_revision must contain exactly git_commit and dirty")
    commit, dirty = value["git_commit"], value["dirty"]
    if commit is not None and (
        not isinstance(commit, str) or len(commit) != 40 or
        any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("source_revision git_commit must be a hexadecimal Git SHA or null")
    if dirty is not None and not isinstance(dirty, bool):
        raise ValueError("source_revision dirty must be boolean or null")


def _validate_score_reference(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("score_artifact must contain exactly path and sha256")
    path = value["path"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("score_artifact path must be a nonempty relative path")
    parsed = Path(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError("score_artifact path must be relative and remain under its root")
    _validate_digest(value["sha256"], "score artifact SHA-256")


def _validate_dataset_identity(value: object, tuning_day: str) -> None:
    if not isinstance(value, dict) or set(value) != {"day", "configured_path", "sha256"}:
        raise ValueError(
            "dataset_identity must contain exactly day, configured_path, and sha256"
        )
    if value["day"] != tuning_day:
        raise ValueError("dataset_identity day must equal tuning_day")
    if not isinstance(value["configured_path"], str) or not value["configured_path"].strip():
        raise ValueError("dataset_identity configured_path must be nonempty")
    _validate_digest(value["sha256"], "dataset identity SHA-256")


def _validate_selection_thresholds(model_variant: str, value: object) -> None:
    baseline = {"mu_percentile", "mu_threshold"}
    robust = baseline | {"sigma_percentile", "sigma_threshold", "rejected_count"}
    expected = baseline if model_variant == "baseline" else robust
    if not isinstance(value, dict) or set(value) != expected:
        fields = ", ".join(sorted(expected))
        raise ValueError(f"{model_variant} thresholds must contain exactly {fields}")
    mu_grid = BASELINE_MU_PERCENTILES if model_variant == "baseline" else MU_PERCENTILES
    _validate_percentile(value["mu_percentile"], mu_grid, "mu_percentile")
    _finite_number(value["mu_threshold"], "mu_threshold")
    if model_variant == "robust":
        _validate_percentile(
            value["sigma_percentile"], SIGMA_PERCENTILES, "sigma_percentile"
        )
        _finite_number(value["sigma_threshold"], "sigma_threshold")
        _nonnegative_integer(value["rejected_count"], "rejected_count")


def _validate_candidate_row(model_variant: str, row: object) -> None:
    baseline = {"checkpoint_id", "epoch", "mu_percentile", "mu_threshold", "f1"}
    robust = baseline | {"sigma_percentile", "sigma_threshold", "rejected_count"}
    expected = baseline if model_variant == "baseline" else robust
    if not isinstance(row, dict) or set(row) != expected:
        fields = ", ".join(sorted(expected))
        raise ValueError(f"candidate_grid row fields must be exactly {fields}")
    if not isinstance(row["checkpoint_id"], str) or not row["checkpoint_id"].strip():
        raise ValueError("candidate_grid checkpoint_id must be nonempty")
    if (
        not isinstance(row["epoch"], int)
        or isinstance(row["epoch"], bool)
        or row["epoch"] <= 0
    ):
        raise ValueError("candidate_grid epoch must be a positive integer")
    mu_grid = BASELINE_MU_PERCENTILES if model_variant == "baseline" else MU_PERCENTILES
    _validate_percentile(row["mu_percentile"], mu_grid, "candidate_grid mu_percentile")
    _finite_number(row["mu_threshold"], "candidate_grid mu_threshold")
    _bounded_score(row["f1"], "candidate_grid f1")
    if model_variant == "robust":
        _validate_percentile(
            row["sigma_percentile"], SIGMA_PERCENTILES,
            "candidate_grid sigma_percentile",
        )
        _finite_number(row["sigma_threshold"], "candidate_grid sigma_threshold")
        _nonnegative_integer(row["rejected_count"], "candidate_grid rejected_count")


def _validate_percentile(value: object, allowed: Sequence[int], name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value not in allowed:
        raise ValueError(f"{name} is outside the frozen candidate grid")


def _finite_number(value: object, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")


def _bounded_score(value: object, name: str) -> None:
    _finite_number(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _temporary(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor); return Path(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary ROC-AUC via average ranks, including tied scores."""
    _require_binary(labels)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positives = int(labels.sum())
    negatives = labels.size - positives
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) /
                 (positives * negatives))


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.array([[np.sum(~labels & ~predictions), np.sum(~labels & predictions)],
                     [np.sum(labels & ~predictions), np.sum(labels & predictions)]], dtype=int)


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = int(np.sum(labels & predictions))
    denominator = 2 * true_positive + int(np.sum(~labels & predictions)) + int(np.sum(labels & ~predictions))
    return 0.0 if denominator == 0 else 2 * true_positive / denominator

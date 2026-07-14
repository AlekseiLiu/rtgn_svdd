"""Small command interface for companion training and evaluation workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Config, load_config, resolve_config, write_config


_SECTIONS = {"schema_version", "experiment", "data", "model", "training", "evaluation"}
_FIELDS = {
    "experiment": {"name", "output_root", "device"},
    "data": {"days", "feature_columns", "num_train", "num_validation"},
    "model": {"memory_dim", "time_dim", "endpoint_dim", "neighbor_size", "uncertainty_mode"},
    "training": {"day", "epochs", "batch_size", "learning_rate", "weight_decay",
                 "negative_ratio", "negative_covariance", "seed", "update_injected_state",
                 "checkpoint_selection"},
    "evaluation": {"protocol", "tuning", "reporting_days", "noise_ratios", "noise_seeds",
                   "feature_distribution", "allow_self_edges"},
}


@dataclass(frozen=True)
class EvaluationJob:
    day: str
    noise_ratio: float
    repeat: int
    noise_seed: int


def validate_experiment_config(value: Mapping[str, object]) -> Config:
    """Validate the deliberately small v1 experiment configuration."""
    config = resolve_config(value)
    _exact_keys(config, _SECTIONS, "configuration")
    if config["schema_version"] != 1:
        raise ValueError("only experiment schema_version 1 is supported")
    for section, fields in _FIELDS.items():
        item = config.get(section)
        if not isinstance(item, dict):
            raise ValueError(f"{section} must be a JSON object")
        _exact_keys(item, fields, section)

    experiment, data = config["experiment"], config["data"]
    model, training, evaluation = config["model"], config["training"], config["evaluation"]
    _nonempty_string(experiment["name"], "experiment.name")
    _nonempty_string(experiment["output_root"], "experiment.output_root")
    if experiment["device"] != "cpu" and not (
        isinstance(experiment["device"], str) and experiment["device"].startswith("cuda")
    ):
        raise ValueError("experiment.device must be 'cpu' or a CUDA device")
    days = data["days"]
    if not isinstance(days, dict) or not days:
        raise ValueError("data.days must be a nonempty object")
    for day, path in days.items():
        _nonempty_string(day, "data day")
        _nonempty_string(path, f"data.days.{day}")
    features = data["feature_columns"]
    if features is not None and (not isinstance(features, list) or not features or
                                 any(not isinstance(name, str) or not name for name in features)):
        raise ValueError("data.feature_columns must be null or a nonempty string list")
    for name in ("num_train", "num_validation"):
        _positive_int(data[name], f"data.{name}")
    for name in ("memory_dim", "time_dim", "endpoint_dim", "neighbor_size"):
        _positive_int(model[name], f"model.{name}")
    if model["endpoint_dim"] % 8:
        raise ValueError("model.endpoint_dim must be divisible by 8")
    if model["uncertainty_mode"] not in {
        "per_dimension_log_variance",
        "historical_shared_per_node",
    }:
        raise ValueError("unsupported model.uncertainty_mode")

    if training["day"] != "wednesday":
        raise ValueError("training.day must be 'wednesday' for the paper workflow")
    for name in ("epochs", "batch_size"):
        _positive_int(training[name], f"training.{name}")
    _nonnegative_int(training["seed"], "training.seed")
    for name in ("learning_rate", "negative_covariance"):
        _positive_number(training[name], f"training.{name}")
    _nonnegative_number(training["weight_decay"], "training.weight_decay")
    _bounded_number(training["negative_ratio"], "training.negative_ratio", 0.0, 1.0)
    _boolean(training["update_injected_state"], "training.update_injected_state")
    if training["checkpoint_selection"] not in {
        "final_configured_epoch",
        "wednesday_f1",
    }:
        raise ValueError("unsupported training.checkpoint_selection")

    if evaluation["protocol"] != "paper_recovered":
        raise ValueError("evaluation.protocol must be 'paper_recovered'; fixed_threshold needs explicit checkpoint candidates and is not exposed by CLI v1")
    tuning = evaluation["tuning"]
    if not isinstance(tuning, dict) or set(tuning) != {"day", "noise_ratio", "noise_seed"}:
        raise ValueError("evaluation.tuning must contain exactly day, noise_ratio, and noise_seed")
    if tuning["day"] != "wednesday" or not _same_number(tuning["noise_ratio"], 0.05):
        raise ValueError("evaluation tuning is reserved for Wednesday with noise ratio 0.05")
    _nonnegative_int(tuning["noise_seed"], "evaluation.tuning.noise_seed")
    if tuning["day"] not in days:
        raise ValueError("evaluation tuning day must have a configured data path")
    if training["day"] not in days:
        raise ValueError("training.day must have a configured data path")
    reporting = evaluation["reporting_days"]
    if not isinstance(reporting, list) or not reporting or len(set(reporting)) != len(reporting):
        raise ValueError("evaluation.reporting_days must be a nonempty unique list")
    if tuning["day"] in reporting:
        raise ValueError("evaluation tuning day must be excluded from reporting days")
    missing_days = [day for day in reporting if day not in days]
    if missing_days:
        raise ValueError(f"reporting days lack data paths: {', '.join(missing_days)}")
    ratios = evaluation["noise_ratios"]
    if not isinstance(ratios, list) or not ratios:
        raise ValueError("evaluation.noise_ratios must be a nonempty list")
    for ratio in ratios:
        _nonnegative_number(ratio, "evaluation noise ratio")
    if len(set(ratios)) != len(ratios):
        raise ValueError("evaluation.noise_ratios must be unique")
    seeds = evaluation["noise_seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("evaluation.noise_seeds must be a nonempty list")
    for seed in seeds:
        _nonnegative_int(seed, "evaluation noise seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation.noise_seeds must be unique")
    if evaluation["feature_distribution"] not in {"paper_covariance_5", "historical_scale_5"}:
        raise ValueError("unsupported evaluation.feature_distribution")
    _boolean(evaluation["allow_self_edges"], "evaluation.allow_self_edges")
    return config


def evaluation_jobs(config: Mapping[str, Any]) -> list[EvaluationJob]:
    """Expand reporting settings; repeats vary noise only, never training."""
    evaluation = config["evaluation"]
    return [
        EvaluationJob(day, float(ratio), repeat, int(seed))
        for day in evaluation["reporting_days"]
        for ratio in evaluation["noise_ratios"]
        for repeat, seed in enumerate(evaluation["noise_seeds"])
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = validate_experiment_config(load_config(args.config))
        if args.command == "validate":
            print(json.dumps(config, indent=2, sort_keys=True))
            if args.resolved_out is not None:
                args.resolved_out.parent.mkdir(parents=True, exist_ok=True)
                write_config(config, args.resolved_out)
        elif args.command == "train":
            _train(config, args.variant)
        elif args.command == "evaluate":
            _evaluate(config, args.variant)
        elif args.command == "diagnostic":
            from .plotting import plot_score_diagnostic
            scores = args.scores
            if scores is None:
                if config["training"]["checkpoint_selection"] != "wednesday_f1":
                    raise ValueError(
                        "final_configured_epoch diagnostic requires --scores"
                    )
                paths = _paths(config, "robust")
                _load_tuning_selection(config, "robust", paths)
                scores = paths["tuning_score"]
            output = args.output
            if output is None:
                output = (Path(config["experiment"]["output_root"]) / "plots" /
                          config["experiment"]["name"] / "wednesday" / "noise_0.05" /
                          "robust_diagnostic.png")
            print(plot_score_diagnostic(scores, output))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtgn-svdd", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate and resolve a configuration without loading data")
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--resolved-out", type=Path)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name, help=f"{name} one companion model variant")
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--variant", required=True, choices=("baseline", "robust"))
    diagnostic = commands.add_parser("diagnostic", help="plot a saved robust score artifact")
    diagnostic.add_argument("--config", required=True, type=Path)
    diagnostic.add_argument("--scores", type=Path)
    diagnostic.add_argument("--output", type=Path)
    return parser


def _train(config: Config, variant: str) -> None:
    import torch
    from .data import split_chronologically
    from .training import create_optimizer

    paths = _paths(config, variant)
    if config["training"]["checkpoint_selection"] == "wednesday_f1":
        paths["selection"].unlink(missing_ok=True)
    torch.manual_seed(config["training"]["seed"])
    streams = _load_streams(config)
    train_day = config["training"]["day"]
    split = split_chronologically(streams[train_day], num_train=config["data"]["num_train"],
                                  num_validation=config["data"]["num_validation"])
    attack_count = int(split.train.is_attack.sum())
    injected_count = int(split.train.is_injected.sum())
    if attack_count or injected_count:
        raise ValueError(
            "one-class training requires only normal events; "
            f"found attacks={attack_count}, injected={injected_count} "
            f"in {train_day} training split"
        )
    model = _make_model(config, variant, streams)
    training = _training_config(config, variant)
    optimizer = create_optimizer(model, training)
    generator = torch.Generator(device="cpu").manual_seed(config["training"]["seed"])
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    resolved = {**config, "revision": _revision_manifest()}
    write_config(resolved, paths["manifest"])
    if config["training"]["checkpoint_selection"] == "wednesday_f1":
        _train_wednesday_f1(
            config, variant, split, streams, model, optimizer, training, generator,
            resolved, paths,
        )
        print(paths["checkpoint"])
        return

    from .training import save_checkpoint, train_epoch
    for epoch in range(1, config["training"]["epochs"] + 1):
        loss = train_epoch(model, split.train, optimizer, training, generator=generator)
        print(f"epoch={epoch} loss={loss:.8g}")
    save_checkpoint(paths["checkpoint"], model=model, optimizer=optimizer,
                    config=resolved, epoch=config["training"]["epochs"])
    print(paths["checkpoint"])


def _train_wednesday_f1(
    config, variant, split, streams, model, optimizer, training, generator,
    resolved, paths,
) -> None:
    import torch
    from .evaluation import (
        ScoreArtifact,
        make_tuning_selection_artifact,
        merge_baseline_threshold_choices,
        merge_fixed_threshold_choices,
        read_score_artifact,
        tune_baseline_threshold,
        tune_fixed_threshold,
        write_score_artifact,
        write_tuning_selection_artifact,
    )
    from .noise import inject_test_noise
    from .training import load_checkpoint, replay_splits, save_checkpoint, train_epoch

    original = streams[config["evaluation"]["tuning"]["day"]]
    node_ids = torch.unique(torch.cat((original.src, original.dst))).cpu()
    tuning = inject_test_noise(
        split.test,
        node_ids=node_ids,
        ratio=config["evaluation"]["tuning"]["noise_ratio"],
        seed=config["evaluation"]["tuning"]["noise_seed"],
        feature_distribution=config["evaluation"]["feature_distribution"],
        allow_self_edges=config["evaluation"]["allow_self_edges"],
    )
    global_choice = None
    output_root = Path(config["experiment"]["output_root"])
    for epoch in range(1, config["training"]["epochs"] + 1):
        loss = train_epoch(model, split.train, optimizer, training, generator=generator)
        print(f"epoch={epoch} loss={loss:.8g}")
        _, validation_output, tuning_output = replay_splits(
            model, split.train, split.validation, tuning.events,
            batch_size=config["training"]["batch_size"],
            update_injected_state=config["training"]["update_injected_state"],
        )
        logical_id = f"epoch_{epoch:04d}"
        candidate = {
            "checkpoint_id": logical_id,
            "epoch": epoch,
            "s_mu": tuning_output.scores.s_mu,
            "validation_s_mu": validation_output.scores.s_mu,
        }
        if variant == "baseline":
            epoch_choice = tune_baseline_threshold([candidate], tuning.events)
            merged = (
                epoch_choice if global_choice is None
                else merge_baseline_threshold_choices([global_choice, epoch_choice])
            )
        else:
            candidate.update({
                "s_sigma": tuning_output.scores.s_sigma,
                "validation_s_sigma": validation_output.scores.s_sigma,
            })
            epoch_choice = tune_fixed_threshold([candidate], tuning.events)
            merged = (
                epoch_choice if global_choice is None
                else merge_fixed_threshold_choices([global_choice, epoch_choice])
            )
        previous_id = None if global_choice is None else global_choice.checkpoint_id
        if merged.checkpoint_id == logical_id and merged.checkpoint_id != previous_id:
            save_checkpoint(
                paths["checkpoint"], model=model, optimizer=optimizer,
                config=resolved, epoch=epoch,
            )
            checkpoint_sha = _sha256(paths["checkpoint"])
            score = ScoreArtifact(
                t=tuning.events.t.cpu().numpy(),
                is_attack=tuning.events.is_attack.cpu().numpy(),
                is_injected=tuning.events.is_injected.cpu().numpy(),
                s_mu=tuning_output.scores.s_mu.detach().cpu().numpy(),
                s_sigma=(None if tuning_output.scores.s_sigma is None else
                         tuning_output.scores.s_sigma.detach().cpu().numpy()),
                metadata={
                    "model_variant": variant,
                    "selection_policy": "wednesday_f1",
                    "day": "wednesday",
                    "noise_ratio": 0.05,
                    "noise_seed": config["evaluation"]["tuning"]["noise_seed"],
                    "logical_checkpoint_id": logical_id,
                    "epoch": epoch,
                    "selected_checkpoint_sha256": checkpoint_sha,
                },
            )
            write_score_artifact(score, paths["tuning_score"])
        global_choice = merged

    assert global_choice is not None
    checkpoint_sha = _sha256(paths["checkpoint"])
    checkpoint_metadata = load_checkpoint(
        paths["checkpoint"], model=model, map_location=config["experiment"]["device"]
    )
    score_sha = _sha256(paths["tuning_score"])
    score = read_score_artifact(paths["tuning_score"], expected_sha256=score_sha)
    expected = {
        "model_variant": variant,
        "selection_policy": "wednesday_f1",
        "day": "wednesday",
        "noise_ratio": 0.05,
        "noise_seed": config["evaluation"]["tuning"]["noise_seed"],
        "logical_checkpoint_id": global_choice.checkpoint_id,
        "epoch": global_choice.epoch,
        "selected_checkpoint_sha256": checkpoint_sha,
    }
    if checkpoint_metadata.epoch != global_choice.epoch or score.metadata != expected:
        raise ValueError("selected checkpoint and Wednesday tuning score do not agree")
    score_reference = {
        "path": str(paths["tuning_score"].relative_to(output_root)),
        "sha256": score_sha,
    }
    artifact = make_tuning_selection_artifact(
        global_choice,
        training_seed=config["training"]["seed"],
        tuning_noise_seed=config["evaluation"]["tuning"]["noise_seed"],
        selected_checkpoint_sha256=checkpoint_sha,
        dataset_identity={
            "day": "wednesday",
            "configured_path": config["data"]["days"]["wednesday"],
            "sha256": _sha256(Path(config["data"]["days"]["wednesday"])),
        },
        normal_denominator=tuning.normal_denominator,
        realized_injected=tuning.injected_count,
        feature_distribution=tuning.feature_distribution,
        update_injected_state=config["training"]["update_injected_state"],
        uncertainty_mode=(config["model"]["uncertainty_mode"] if variant == "robust" else None),
        source_revision=resolved["revision"],
        score_artifact=score_reference,
    )
    write_tuning_selection_artifact(artifact, paths["selection"])


def _evaluate(config: Config, variant: str) -> None:
    import torch
    from .data import split_chronologically
    from .evaluation import (ScoreArtifact, aggregate_records, evaluate_baseline,
                             evaluate_paper_recovered, write_result_record,
                             write_aggregate_record, write_score_artifact)
    from .noise import inject_test_noise
    from .training import load_checkpoint, replay_splits

    paths = _paths(config, variant)
    selection = None
    selection_reference = None
    if config["training"]["checkpoint_selection"] == "wednesday_f1":
        selection, selection_reference = _load_tuning_selection(config, variant, paths)
    streams = _load_streams(config)
    model = _make_model(config, variant, streams)
    checkpoint = paths["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist; run train first: {checkpoint}")
    metadata = load_checkpoint(checkpoint, model=model, map_location=config["experiment"]["device"])
    checkpoint_id = _sha256(checkpoint)
    if selection is not None:
        if metadata.epoch != selection.selected_epoch:
            raise ValueError("selected checkpoint epoch does not match tuning selection")
        if metadata.config.get("revision") != selection.source_revision:
            raise ValueError("selected checkpoint source revision does not match tuning selection")
        if _replay_semantics(metadata.config) != _replay_semantics(config):
            raise ValueError("current replay semantics do not match the selected checkpoint")
    dataset_identities = {
        day: {
            "day": day,
            "configured_path": path,
            "sha256": _sha256(Path(path)),
        }
        for day, path in config["data"]["days"].items()
    }
    records_by_group: dict[tuple[str, float], list[Any]] = {}
    for job in evaluation_jobs(config):
        original = streams[job.day]
        split = split_chronologically(original, num_train=config["data"]["num_train"],
                                      num_validation=config["data"]["num_validation"])
        node_ids = torch.unique(torch.cat((original.src, original.dst))).cpu()
        injected = inject_test_noise(
            split.test, node_ids=node_ids, ratio=job.noise_ratio, seed=job.noise_seed,
            feature_distribution=config["evaluation"]["feature_distribution"],
            allow_self_edges=config["evaluation"]["allow_self_edges"],
        )
        _, _, output = replay_splits(
            model, split.train, split.validation, injected.events,
            batch_size=config["training"]["batch_size"],
            update_injected_state=config["training"]["update_injected_state"],
        )
        base = _result_directory(config, job)
        score_path = Path(config["experiment"]["output_root"]) / "scores" / base / f"{variant}.npz"
        result_path = Path(config["experiment"]["output_root"]) / "results" / base / f"{variant}.json"
        scores = ScoreArtifact(
            t=injected.events.t.cpu().numpy(),
            is_attack=injected.events.is_attack.cpu().numpy(),
            is_injected=injected.events.is_injected.cpu().numpy(),
            s_mu=output.scores.s_mu.detach().cpu().numpy(),
            s_sigma=(None if output.scores.s_sigma is None else
                     output.scores.s_sigma.detach().cpu().numpy()),
            metadata={"day": job.day, "noise_ratio": job.noise_ratio, "repeat": job.repeat,
                      "noise_seed": job.noise_seed, "checkpoint_id": checkpoint_id},
        )
        digest = write_score_artifact(scores, score_path)
        output_root = Path(config["experiment"]["output_root"])
        common = dict(
            model_variant=variant, day=job.day, noise_ratio=job.noise_ratio,
            repeat=job.repeat, training_seed=config["training"]["seed"],
            noise_seed=job.noise_seed, checkpoint_id=checkpoint_id,
            update_injected_state=config["training"]["update_injected_state"],
            uncertainty_mode=(config["model"]["uncertainty_mode"] if variant == "robust" else None),
            score_artifact={"path": str(score_path.relative_to(output_root)), "sha256": digest},
            checkpoint_epoch=metadata.epoch, feature_distribution=injected.feature_distribution,
            normal_denominator=injected.normal_denominator,
            realized_injected=injected.injected_count,
            source_revision=metadata.config.get("revision"),
            dataset_identity=dataset_identities[job.day],
            checkpoint_selection=config["training"]["checkpoint_selection"],
        )
        if selection is not None:
            common.update(
                tuning_selection_artifact=selection_reference,
                selected_epoch=selection.selected_epoch,
                selected_logical_checkpoint_id=selection.selected_logical_checkpoint_id,
            )
        if variant == "baseline":
            record = evaluate_baseline(injected.events, scores.s_mu, **common)
        else:
            assert scores.s_sigma is not None
            record = evaluate_paper_recovered(injected.events, scores.s_mu, scores.s_sigma, **common)
        write_result_record(record, result_path)
        records_by_group.setdefault((job.day, job.noise_ratio), []).append(record)
    for (day, ratio), records in records_by_group.items():
        aggregate = aggregate_records(records, expected_repeats=len(config["evaluation"]["noise_seeds"]))
        path = Path(config["experiment"]["output_root"]) / "results" / config["experiment"]["name"] / day / _noise_name(ratio) / f"{variant}_aggregate.json"
        write_aggregate_record(aggregate, path)


def _load_tuning_selection(config, variant, paths):
    from .evaluation import read_score_artifact, read_tuning_selection_artifact

    selection_path = paths["selection"]
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"tuning selection does not exist; run train first: {selection_path}"
        )
    selection = read_tuning_selection_artifact(selection_path)
    if selection.model_variant != variant:
        raise ValueError("tuning selection model variant does not match the command")
    expected_uncertainty = (
        config["model"]["uncertainty_mode"] if variant == "robust" else None
    )
    expected = {
        "selection_policy": config["training"]["checkpoint_selection"],
        "training_seed": config["training"]["seed"],
        "tuning_day": config["evaluation"]["tuning"]["day"],
        "tuning_noise_ratio": config["evaluation"]["tuning"]["noise_ratio"],
        "tuning_noise_seed": config["evaluation"]["tuning"]["noise_seed"],
        "feature_distribution": config["evaluation"]["feature_distribution"],
        "update_injected_state": config["training"]["update_injected_state"],
        "uncertainty_mode": expected_uncertainty,
    }
    actual = {name: getattr(selection, name) for name in expected}
    if actual != expected:
        raise ValueError("current selection semantics do not match the tuning selection")
    wednesday_path = config["data"]["days"]["wednesday"]
    dataset_identity = {
        "day": "wednesday",
        "configured_path": wednesday_path,
        "sha256": _sha256(Path(wednesday_path)),
    }
    if selection.dataset_identity != dataset_identity:
        raise ValueError("current Wednesday dataset does not match the tuning selection")
    checkpoint = paths["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint does not exist: {checkpoint}")
    if _sha256(checkpoint) != selection.selected_checkpoint_sha256:
        raise ValueError("selected checkpoint integrity check failed")
    output_root = Path(config["experiment"]["output_root"])
    score_path = output_root / selection.score_artifact["path"]
    if score_path != paths["tuning_score"] or not score_path.is_file():
        raise FileNotFoundError("selected Wednesday score artifact does not exist at its fixed path")
    score = read_score_artifact(
        score_path, expected_sha256=selection.score_artifact["sha256"]
    )
    expected_score_metadata = {
        "model_variant": variant,
        "selection_policy": "wednesday_f1",
        "day": "wednesday",
        "noise_ratio": 0.05,
        "noise_seed": selection.tuning_noise_seed,
        "logical_checkpoint_id": selection.selected_logical_checkpoint_id,
        "epoch": selection.selected_epoch,
        "selected_checkpoint_sha256": selection.selected_checkpoint_sha256,
    }
    if score.metadata != expected_score_metadata:
        raise ValueError("selected Wednesday score metadata does not match tuning selection")
    return selection, {
        "path": str(selection_path.relative_to(output_root)),
        "sha256": _sha256(selection_path),
    }


def _replay_semantics(config):
    return {
        "model": config["model"],
        "data": {
            "feature_columns": config["data"]["feature_columns"],
            "num_train": config["data"]["num_train"],
            "num_validation": config["data"]["num_validation"],
            "wednesday_path": config["data"]["days"]["wednesday"],
        },
        "training": {
            name: config["training"][name]
            for name in ("batch_size", "seed", "update_injected_state", "checkpoint_selection")
        },
        "evaluation": {
            "tuning": config["evaluation"]["tuning"],
            "feature_distribution": config["evaluation"]["feature_distribution"],
            "allow_self_edges": config["evaluation"]["allow_self_edges"],
        },
    }


def _load_streams(config: Mapping[str, Any]):
    from .data import load_cic_csv
    return {day: load_cic_csv(path, feature_columns=config["data"]["feature_columns"])
            for day, path in config["data"]["days"].items()}


def _make_model(config: Mapping[str, Any], variant: str, streams: Mapping[str, Any]):
    import torch
    from .training import TGNModel
    torch.manual_seed(config["training"]["seed"])
    msg_dims = {stream.msg.shape[1] for stream in streams.values()}
    if len(msg_dims) != 1:
        raise ValueError("all configured streams must use the same message feature dimension")
    max_node = max(int(torch.max(torch.cat((stream.src, stream.dst)))) for stream in streams.values())
    generator = torch.Generator(device="cpu").manual_seed(config["training"]["seed"])
    center = torch.randn(2 * config["model"]["endpoint_dim"], generator=generator)
    return TGNModel(num_nodes=max_node + 1, msg_dim=msg_dims.pop(), center=center,
                    variant=variant, device=config["experiment"]["device"], **config["model"])


def _training_config(config: Mapping[str, Any], variant: str):
    from .training import TrainingConfig
    values = {key: value for key, value in config["training"].items()
              if key in {"batch_size", "learning_rate", "weight_decay", "negative_ratio",
                         "negative_covariance", "update_injected_state"}}
    return TrainingConfig(variant=variant, **values)


def _paths(config: Mapping[str, Any], variant: str) -> dict[str, Path]:
    root = Path(config["experiment"]["output_root"])
    name = config["experiment"]["name"]
    return {
        "checkpoint": root / "checkpoints" / name / variant / "checkpoint.pt",
        "manifest": root / "manifests" / name / f"{variant}.json",
        "selection": root / "tuning" / name / variant / "selection.json",
        "tuning_score": (root / "scores" / name / "wednesday" / "noise_0.05" /
                         "tuning" / f"{variant}.npz"),
    }


def _result_directory(config: Mapping[str, Any], job: EvaluationJob) -> Path:
    return Path(config["experiment"]["name"]) / job.day / _noise_name(job.noise_ratio) / f"repeat_{job.repeat}"


def _noise_name(ratio: float) -> str:
    return f"noise_{ratio:.2f}"


def _revision_manifest() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        return {"git_commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"git_commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "dirty": None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    missing, unknown = expected - value.keys(), value.keys() - expected
    if missing:
        raise ValueError(f"missing {name} fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(sorted(unknown))}")


def _nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_number(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _nonnegative_number(value: object, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def _bounded_number(value: object, name: str, low: float, high: float) -> None:
    _nonnegative_number(value, name)
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _boolean(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")


def _same_number(value: object, expected: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isclose(value, expected)


if __name__ == "__main__":
    raise SystemExit(main())

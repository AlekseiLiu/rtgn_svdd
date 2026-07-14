import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rtgn_svdd.cli import evaluation_jobs, main, validate_experiment_config
from rtgn_svdd.config import load_config
from rtgn_svdd.evaluation import (
    ScoreArtifact,
    read_aggregate_record,
    read_score_artifact,
    read_tuning_selection_artifact,
    write_score_artifact,
)


ROOT = Path(__file__).parents[1]


def test_cli_help_describes_companion_workflow(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    assert "companion training and evaluation workflows" in capsys.readouterr().out


def test_validate_prints_canonical_config_without_loading_missing_data(capsys):
    status = main(["validate", "--config", str(ROOT / "configs/smoke.json")])
    assert status == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["training"]["day"] == "wednesday"
    assert printed == load_config(ROOT / "configs/smoke.json")


def test_validate_can_save_resolved_config(tmp_path):
    output = tmp_path / "nested" / "resolved.json"
    assert main(["validate", "--config", str(ROOT / "configs/smoke.json"),
                 "--resolved-out", str(output)]) == 0
    assert load_config(output)["experiment"]["name"] == "smoke"


def test_demo_preset_freezes_bounded_practical_workflow():
    config = validate_experiment_config(load_config(ROOT / "configs/demo.json"))
    jobs = evaluation_jobs(config)

    assert len(jobs) == 1 * 3 * 3
    assert {job.day for job in jobs} == {"tuesday"}
    assert {job.noise_ratio for job in jobs} == {0.1, 0.3, 0.5}
    assert {job.noise_seed for job in jobs} == {0, 1, 2}
    assert config["training"]["epochs"] == 15
    assert config["training"]["checkpoint_selection"] == "wednesday_f1"
    assert config["training"]["update_injected_state"] is False
    assert config["experiment"] == {
        "device": "cuda:0",
        "name": "demo",
        "output_root": "outputs/demo",
    }


def test_demo_runner_orders_public_commands_and_refuses_completed_run(tmp_path):
    config = load_config(ROOT / "configs/demo.json")
    config["experiment"]["output_root"] = str(tmp_path / "outputs")
    config_path = tmp_path / "demo.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    command_log = tmp_path / "commands.log"
    fake_cli = tmp_path / "rtgn-svdd"
    fake_cli.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    environment = {
        **os.environ,
        "COMMAND_LOG": str(command_log),
        "PYTHON_BIN": sys.executable,
        "RTGN_SVDD_BIN": str(fake_cli),
    }

    subprocess.run(
        ["bash", str(ROOT / "scripts/run_demo.sh"), str(config_path)],
        check=True,
        cwd=ROOT,
        env=environment,
    )

    assert command_log.read_text(encoding="utf-8").splitlines() == [
        f"validate --config {config_path}",
        f"train --config {config_path} --variant baseline",
        f"train --config {config_path} --variant robust",
        f"evaluate --config {config_path} --variant baseline",
        f"evaluate --config {config_path} --variant robust",
        f"diagnostic --config {config_path}",
    ]
    marker = tmp_path / "outputs/demo.completed"
    assert marker.is_file()

    repeated = subprocess.run(
        ["bash", str(ROOT / "scripts/run_demo.sh"), str(config_path)],
        check=False,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "refusing to overwrite completed demo" in repeated.stderr
    assert len(command_log.read_text(encoding="utf-8").splitlines()) == 6


def test_smoke_preset_explicitly_uses_final_epoch_selection():
    config = validate_experiment_config(load_config(ROOT / "configs/smoke.json"))
    assert config["training"]["checkpoint_selection"] == "final_configured_epoch"
    assert config["evaluation"]["tuning"]["noise_seed"] == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("evaluation", "protocol", "fixed_threshold"), "paper_recovered"),
        (("training", "day", "tuesday"), "wednesday"),
        (("model", "endpoint_dim", 7), "divisible by 8"),
        (("evaluation", "noise_seeds", [0, 0]), "unique"),
        (("model", "uncertainty_mode", "shared_log_variance"), "uncertainty_mode"),
        (("training", "checkpoint_selection", "best_auc"), "checkpoint_selection"),
    ],
)
def test_validation_rejects_unsupported_or_inconsistent_choices(change, message):
    config = load_config(ROOT / "configs/smoke.json")
    section, key, value = change
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_experiment_config(config)


def test_validation_rejects_unknown_fields():
    config = load_config(ROOT / "configs/smoke.json")
    config["training"]["typo"] = 1
    with pytest.raises(ValueError, match="unknown training fields: typo"):
        validate_experiment_config(config)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("training", "checkpoint_selection"),
        ("tuning", "noise_seed"),
    ],
)
def test_validation_rejects_selection_fields_missing_from_old_configs(section, key):
    config = load_config(ROOT / "configs/smoke.json")
    target = config["evaluation"]["tuning"] if section == "tuning" else config[section]
    del target[key]
    with pytest.raises(ValueError, match=key):
        validate_experiment_config(config)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "0"])
def test_validation_requires_nonnegative_integer_tuning_seed(seed):
    config = load_config(ROOT / "configs/smoke.json")
    config["evaluation"]["tuning"]["noise_seed"] = seed
    with pytest.raises(ValueError, match="evaluation.tuning.noise_seed.*nonnegative integer"):
        validate_experiment_config(config)


def test_validation_requires_a_configured_tuning_day():
    config = load_config(ROOT / "configs/smoke.json")
    del config["data"]["days"]["wednesday"]
    with pytest.raises(ValueError, match="tuning day must have a configured data path"):
        validate_experiment_config(config)


def test_validation_excludes_tuning_day_from_reporting_days():
    config = load_config(ROOT / "configs/smoke.json")
    config["evaluation"]["reporting_days"] = ["wednesday"]
    with pytest.raises(ValueError, match="tuning day must be excluded"):
        validate_experiment_config(config)


def test_public_cli_baseline_smoke_trains_and_evaluates(tmp_path):
    header = "src_ip,dst_ip,bidirectional_first_seen_ms,label,attack,feature\n"
    rows = []
    for index in range(16):
        label = int(index >= 12 and index % 2 == 1)
        rows.append(f"{index % 4},{(index + 1) % 4},{index},{label},{'attack' if label else ''},{index / 10}\n")
    days = {}
    for day in ("wednesday", "tuesday", "thursday", "friday"):
        path = tmp_path / f"{day}.csv"
        path.write_text(header + "".join(rows), encoding="utf-8")
        days[day] = str(path)
    config = load_config(ROOT / "configs/smoke.json")
    config["data"]["days"] = days
    config["experiment"]["output_root"] = str(tmp_path / "outputs")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["train", "--config", str(config_path), "--variant", "baseline"]) == 0
    assert main(["evaluate", "--config", str(config_path), "--variant", "baseline"]) == 0
    output = tmp_path / "outputs"
    assert (output / "checkpoints/smoke/baseline/checkpoint.pt").is_file()
    assert (output / "results/smoke/tuesday/noise_0.10/repeat_0/baseline.json").is_file()
    assert (output / "results/smoke/tuesday/noise_0.10/baseline_aggregate.json").is_file()
    assert not (output / "tuning").exists()
    result = json.loads(
        (output / "results/smoke/tuesday/noise_0.10/repeat_0/baseline.json").read_text()
    )
    assert result["manifest"]["dataset_identity"] == {
        "day": "tuesday",
        "configured_path": days["tuesday"],
        "sha256": hashlib.sha256(Path(days["tuesday"]).read_bytes()).hexdigest(),
    }
    assert result["manifest"]["checkpoint_selection"] == "final_configured_epoch"
    source_revision = result["manifest"]["source_revision"]
    if (ROOT / ".git").exists():
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if revision.returncode == 0:
            assert source_revision["git_commit"] == revision.stdout.strip()
            assert isinstance(source_revision["dirty"], bool)
        else:
            assert source_revision == {"git_commit": None, "dirty": None}
    else:
        assert source_revision == {"git_commit": None, "dirty": None}


def test_public_cli_wednesday_f1_selects_nonfinal_epoch_and_persists_one_best(
    tmp_path, capsys
):
    header = "src_ip,dst_ip,bidirectional_first_seen_ms,label,attack,feature\n"
    rows = []
    for index in range(36):
        label = int(index >= 12 and index % 4 == 1)
        rows.append(
            f"{index % 4},{(index + 1) % 4},{index},{label},"
            f"{'attack' if label else ''},{index / 10}\n"
        )
    days = {}
    for day in ("wednesday", "tuesday", "thursday", "friday"):
        path = tmp_path / f"{day}.csv"
        day_rows = rows
        if day == "tuesday":
            day_rows = [
                f"{index % 4},{(index + 1) % 4},{index},"
                f"{int(index < 12 or (index >= 12 and index % 4 == 1))},"
                f"{'attack' if index < 12 or (index >= 12 and index % 4 == 1) else ''},"
                f"{index / 10}\n"
                for index in range(36)
            ]
        path.write_text(header + "".join(day_rows), encoding="utf-8")
        days[day] = str(path)

    config = load_config(ROOT / "configs/smoke.json")
    config["data"]["days"] = days
    config["experiment"]["name"] = "selection-smoke"
    config["experiment"]["output_root"] = str(tmp_path / "outputs")
    config["training"]["checkpoint_selection"] = "wednesday_f1"
    config["training"]["epochs"] = 2
    config["training"]["learning_rate"] = 1e-30
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["train", "--config", str(config_path), "--variant", "baseline"]) == 0

    root = tmp_path / "outputs"
    checkpoint = root / "checkpoints/selection-smoke/baseline/checkpoint.pt"
    selection_path = root / "tuning/selection-smoke/baseline/selection.json"
    score_path = root / "scores/selection-smoke/wednesday/noise_0.05/tuning/baseline.npz"
    selection = read_tuning_selection_artifact(selection_path)
    selection_bytes = selection_path.read_bytes()
    score = read_score_artifact(
        score_path, expected_sha256=selection.score_artifact["sha256"]
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    assert torch.load(checkpoint, weights_only=False)["epoch"] == 1
    assert selection.selected_epoch == 1
    assert selection.selected_logical_checkpoint_id == "epoch_0001"
    assert selection.selected_checkpoint_sha256 == checkpoint_sha
    assert len(selection.candidate_grid) == 2 * 21
    for first, second in zip(selection.candidate_grid[:21], selection.candidate_grid[21:]):
        assert first["mu_percentile"] == second["mu_percentile"]
        assert first["mu_threshold"] == second["mu_threshold"]
        assert first["f1"] == second["f1"]
    assert selection.normal_denominator == 18
    assert selection.realized_injected == 1
    assert score.metadata["logical_checkpoint_id"] == "epoch_0001"
    assert score.metadata["epoch"] == 1
    assert score.metadata["selected_checkpoint_sha256"] == checkpoint_sha
    assert [path for path in (root / "checkpoints").rglob("*.pt")] == [checkpoint]
    assert [path for path in (root / "scores").rglob("*.npz")] == [score_path]
    assert [path for path in (root / "tuning").rglob("*.json")] == [selection_path]
    assert list(root.rglob("*.tmp")) == []

    assert main(["evaluate", "--config", str(config_path), "--variant", "baseline"]) == 0
    result_path = root / "results/selection-smoke/tuesday/noise_0.10/repeat_0/baseline.json"
    aggregate_path = root / "results/selection-smoke/tuesday/noise_0.10/baseline_aggregate.json"
    result = json.loads(result_path.read_text())
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    assert aggregate_path.is_file()
    aggregate = read_aggregate_record(aggregate_path)
    assert aggregate.values == [result["roc_auc"]]
    assert aggregate.repeats == [0]
    assert aggregate.noise_seeds == [config["evaluation"]["noise_seeds"][0]]
    assert result["checkpoint_id"] == checkpoint_sha
    assert result["protocol"] == "baseline"
    assert result["thresholds"] == {}
    assert result["manifest"]["checkpoint_selection"] == "wednesday_f1"
    assert result["manifest"]["selected_epoch"] == 1
    assert result["manifest"]["selected_logical_checkpoint_id"] == "epoch_0001"
    assert result["manifest"]["tuning_selection_artifact"] == {
        "path": "tuning/selection-smoke/baseline/selection.json",
        "sha256": selection_sha,
    }
    assert selection_path.read_bytes() == selection_bytes

    results_root = root / "results/selection-smoke"
    shutil.rmtree(results_root)
    checkpoint_bytes = checkpoint.read_bytes()
    score_bytes = score_path.read_bytes()
    wednesday = Path(days["wednesday"])
    wednesday_bytes = wednesday.read_bytes()

    def rejected(message):
        with pytest.raises(SystemExit):
            main(["evaluate", "--config", str(config_path), "--variant", "baseline"])
        assert message in capsys.readouterr().err
        assert not results_root.exists()

    config["training"]["update_injected_state"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    rejected("selection semantics")
    config["training"]["update_injected_state"] = True

    config["training"]["batch_size"] = 3
    config_path.write_text(json.dumps(config), encoding="utf-8")
    rejected("replay semantics")
    config["training"]["batch_size"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")

    checkpoint.write_bytes(checkpoint_bytes + b"tampered")
    rejected("checkpoint integrity")
    checkpoint.write_bytes(checkpoint_bytes)

    score_path.write_bytes(score_bytes + b"tampered")
    rejected("score artifact integrity")
    score_path.write_bytes(score_bytes)

    wednesday.write_bytes(wednesday_bytes + b"\n")
    rejected("Wednesday dataset")
    wednesday.write_bytes(wednesday_bytes)

    robust_selection = root / "tuning/selection-smoke/robust/selection.json"
    robust_selection.parent.mkdir(parents=True)
    robust_selection.write_bytes(selection_bytes)
    with pytest.raises(SystemExit):
        main(["evaluate", "--config", str(config_path), "--variant", "robust"])
    assert "model variant" in capsys.readouterr().err
    assert not results_root.exists()
    with pytest.raises(SystemExit):
        main(["diagnostic", "--config", str(config_path)])
    assert "model variant" in capsys.readouterr().err
    assert not (
        root / "plots/selection-smoke/wednesday/noise_0.05/robust_diagnostic.png"
    ).exists()


def test_public_cli_wednesday_f1_writes_robust_threshold_choice(tmp_path, capsys):
    header = "src_ip,dst_ip,bidirectional_first_seen_ms,label,attack,feature\n"
    rows = [
        f"{index % 4},{(index + 1) % 4},{index},{int(index >= 12 and index % 4 == 1)},"
        f"{'attack' if index >= 12 and index % 4 == 1 else ''},{index / 10}\n"
        for index in range(36)
    ]
    days = {}
    for day in ("wednesday", "tuesday", "thursday", "friday"):
        path = tmp_path / f"{day}.csv"
        path.write_text(header + "".join(rows), encoding="utf-8")
        days[day] = str(path)
    config = load_config(ROOT / "configs/smoke.json")
    config["data"]["days"] = days
    config["experiment"]["name"] = "robust-selection"
    config["experiment"]["output_root"] = str(tmp_path / "outputs")
    config["training"]["checkpoint_selection"] = "wednesday_f1"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["train", "--config", str(config_path), "--variant", "robust"]) == 0
    output = tmp_path / "outputs"
    checkpoint = output / "checkpoints/robust-selection/robust/checkpoint.pt"
    score_path = output / "scores/robust-selection/wednesday/noise_0.05/tuning/robust.npz"
    selection = read_tuning_selection_artifact(
        output / "tuning/robust-selection/robust/selection.json"
    )
    score = read_score_artifact(
        score_path, expected_sha256=selection.score_artifact["sha256"]
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert selection.model_variant == "robust"
    assert selection.uncertainty_mode == "per_dimension_log_variance"
    assert selection.selected_checkpoint_sha256 == checkpoint_sha
    assert score.metadata["selected_checkpoint_sha256"] == checkpoint_sha
    assert score.metadata["logical_checkpoint_id"] == selection.selected_logical_checkpoint_id
    assert {"mu_threshold", "sigma_threshold", "rejected_count"} <= selection.thresholds.keys()
    assert len(selection.candidate_grid) == 25

    pytest.importorskip("matplotlib").use("Agg")
    selection_path = output / "tuning/robust-selection/robust/selection.json"
    selection_bytes = selection_path.read_bytes()
    assert main(["diagnostic", "--config", str(config_path)]) == 0
    default_plot = output / "plots/robust-selection/wednesday/noise_0.05/robust_diagnostic.png"
    assert default_plot.stat().st_size > 0
    custom_plot = tmp_path / "custom-diagnostic.svg"
    assert main(["diagnostic", "--config", str(config_path),
                 "--output", str(custom_plot)]) == 0
    assert custom_plot.stat().st_size > 0
    assert selection_path.read_bytes() == selection_bytes

    with pytest.raises(SystemExit):
        main(["evaluate", "--config", str(config_path), "--variant", "robust"])
    assert "no observed s_sigma thresholds satisfy strict 5 < t < 25" in capsys.readouterr().err
    assert not (
        output / "results/robust-selection/tuesday/noise_0.10/repeat_0/robust.json"
    ).exists()
    assert (
        output / "scores/robust-selection/tuesday/noise_0.10/repeat_0/robust.npz"
    ).is_file()


def test_public_cli_rejects_non_normal_events_in_training_prefix(tmp_path, capsys):
    header = "src_ip,dst_ip,bidirectional_first_seen_ms,label,attack,feature\n"
    days = {}
    for day in ("wednesday", "tuesday", "thursday", "friday"):
        path = tmp_path / f"{day}.csv"
        rows = []
        for index in range(16):
            is_training_attack = day == "wednesday" and index == 2
            rows.append(
                f"{index % 4},{(index + 1) % 4},{index},"
                f"{int(is_training_attack)},"
                f"{'attack' if is_training_attack else ''},{index / 10}\n"
            )
        path.write_text(header + "".join(rows), encoding="utf-8")
        days[day] = str(path)

    config = load_config(ROOT / "configs/smoke.json")
    config["data"]["days"] = days
    config["experiment"]["output_root"] = str(tmp_path / "outputs")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["train", "--config", str(config_path), "--variant", "baseline"])

    error = capsys.readouterr().err
    assert "one-class training requires only normal events" in error
    assert "attacks=1, injected=0" in error
    assert not (tmp_path / "outputs/checkpoints/smoke/baseline/checkpoint.pt").exists()


def test_wednesday_f1_failure_removes_stale_selection_commit_marker(tmp_path, capsys):
    header = "src_ip,dst_ip,bidirectional_first_seen_ms,label,attack,feature\n"
    days = {}
    for day in ("wednesday", "tuesday", "thursday", "friday"):
        path = tmp_path / f"{day}.csv"
        rows = [
            f"{index % 4},{(index + 1) % 4},{index},{int(day == 'wednesday' and index == 2)},"
            f"{'attack' if day == 'wednesday' and index == 2 else ''},{index / 10}\n"
            for index in range(16)
        ]
        path.write_text(header + "".join(rows), encoding="utf-8")
        days[day] = str(path)
    config = load_config(ROOT / "configs/smoke.json")
    config["data"]["days"] = days
    config["experiment"]["output_root"] = str(tmp_path / "outputs")
    config["training"]["checkpoint_selection"] = "wednesday_f1"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    selection = tmp_path / "outputs/tuning/smoke/baseline/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("stale", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["train", "--config", str(config_path), "--variant", "baseline"])
    assert "one-class training" in capsys.readouterr().err
    assert not selection.exists()


def test_wednesday_f1_reporting_requires_selection_before_loading_data(capsys):
    with pytest.raises(SystemExit):
        main(["evaluate", "--config", str(ROOT / "configs/demo.json"),
              "--variant", "baseline"])
    assert "tuning selection does not exist" in capsys.readouterr().err


def test_final_epoch_diagnostic_requires_explicit_scores(capsys):
    with pytest.raises(SystemExit):
        main(["diagnostic", "--config", str(ROOT / "configs/smoke.json")])
    assert "final_configured_epoch diagnostic requires --scores" in capsys.readouterr().err


def test_diagnostic_command_reads_saved_scores_without_loading_data(tmp_path):
    pytest.importorskip("matplotlib").use("Agg")
    import numpy as np
    scores = tmp_path / "scores.npz"
    write_score_artifact(
        ScoreArtifact(t=np.arange(3), is_attack=np.array([False, True, False]),
                      is_injected=np.array([False, False, True]), s_mu=np.arange(3.0),
                      s_sigma=np.arange(3.0) + 1, metadata={"day": "wednesday", "noise_ratio": 0.05}),
        scores,
    )
    output = tmp_path / "diagnostic.svg"
    assert main(["diagnostic", "--config", str(ROOT / "configs/smoke.json"),
                 "--scores", str(scores), "--output", str(output)]) == 0
    assert output.stat().st_size > 0
    demo_output = tmp_path / "demo-explicit.svg"
    assert main(["diagnostic", "--config", str(ROOT / "configs/demo.json"),
                 "--scores", str(scores), "--output", str(demo_output)]) == 0
    assert demo_output.stat().st_size > 0

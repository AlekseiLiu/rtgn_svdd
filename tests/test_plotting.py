import numpy as np
import pytest

from rtgn_svdd.evaluation import ScoreArtifact, write_score_artifact


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _artifact(*, s_sigma=True):
    return ScoreArtifact(
        t=np.array([10, 20, 30]),
        is_attack=np.array([False, True, False]),
        is_injected=np.array([False, False, True]),
        s_mu=np.array([1.0, 3.0, 2.0]),
        s_sigma=np.array([2.0, 1.0, 4.0]) if s_sigma else None,
        metadata={"day": "wednesday", "noise_ratio": 0.05},
    )


def test_plot_score_diagnostic_regenerates_from_saved_artifact(tmp_path):
    from rtgn_svdd.plotting import plot_score_diagnostic

    score_path = tmp_path / "scores.npz"
    write_score_artifact(_artifact(), score_path)
    output = plot_score_diagnostic(score_path, tmp_path / "diagnostic.svg")
    assert output == tmp_path / "diagnostic.svg"
    assert output.stat().st_size > 0
    svg = output.read_text()
    assert all(label in svg for label in ("Benign", "Attack", "Injected noise"))
    assert "s_\\mu" in svg
    assert "s_\\sigma" in svg


def test_plot_score_diagnostic_requires_uncertainty_scores(tmp_path):
    from rtgn_svdd.plotting import plot_score_diagnostic

    with pytest.raises(ValueError, match="s_sigma"):
        plot_score_diagnostic(_artifact(s_sigma=False), tmp_path / "diagnostic.png")

"""Diagnostic plots regenerated from saved score artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .evaluation import ScoreArtifact, read_score_artifact


def plot_score_diagnostic(
    artifact: ScoreArtifact | str | Path,
    output_path: str | Path,
) -> Path:
    """Plot robust mean-distance and uncertainty scores from a saved artifact."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise RuntimeError("diagnostic plotting requires the 'plotting' extra") from error

    if not isinstance(artifact, ScoreArtifact):
        artifact = read_score_artifact(artifact)
    if artifact.s_sigma is None:
        raise ValueError("diagnostic plotting requires s_sigma scores")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    injected = np.asarray(artifact.is_injected, dtype=bool)
    attack = np.asarray(artifact.is_attack, dtype=bool)
    categories = (
        ("Benign", ~attack & ~injected, "#31688e", 5.0, 0.25),
        ("Attack", attack, "#fde725", 13.0, 0.9),
        ("Injected noise", injected, "#35b779", 9.0, 0.7),
    )
    panels = (
        (np.asarray(artifact.s_mu), r"$s_\mu$ (squared mean distance)"),
        (np.asarray(artifact.s_sigma), r"$s_\sigma$ (mean predicted variance)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    try:
        for axis, (scores, label) in zip(axes, panels):
            for category, mask, color, size, alpha in categories:
                axis.scatter(artifact.t[mask], scores[mask], label=category,
                             color=color, s=size, alpha=alpha, linewidths=0)
            for boundary in artifact.metadata.get("split_boundaries", []):
                axis.axvline(boundary, color="0.5", linestyle="--", linewidth=1)
            axis.set_xlabel("Event time")
            axis.set_ylabel(label)
            axis.set_title(label)
        axes[0].legend(loc="best")
        context = _context_title(artifact.metadata)
        if context:
            figure.suptitle(context)
        figure.tight_layout()
        figure.savefig(output_path, dpi=180)
    finally:
        plt.close(figure)
    return output_path


def _context_title(metadata: dict[str, object]) -> str:
    parts = []
    if metadata.get("day"):
        parts.append(str(metadata["day"]).capitalize())
    if metadata.get("noise_ratio") is not None:
        parts.append(f"{100 * float(metadata['noise_ratio']):g}% injected noise")
    return " — ".join(parts)

"""Probabilistic objectives and sampling primitives for rTGN-SVDD."""

from __future__ import annotations

import torch
from torch import Tensor


# Bounds keep exponentiation finite in float32 while retaining a broad variance range.
LOG_VARIANCE_MIN = -30.0
LOG_VARIANCE_MAX = 20.0


def stable_log_variance(log_variance: Tensor) -> Tensor:
    """Clamp log-variance to the documented numerically stable bounds."""
    return torch.clamp(log_variance, LOG_VARIANCE_MIN, LOG_VARIANCE_MAX)


def variance_from_log_variance(log_variance: Tensor) -> Tensor:
    """Convert log-variance to variance after clamping to ``[-30, 20]``."""
    return torch.exp(stable_log_variance(log_variance))


def deterministic_svdd_loss(s_mu: Tensor) -> Tensor:
    """Correct Deep-SVDD objective: mean squared center distance."""
    return s_mu.mean()


def positive_gaussian_nll(
    event_mean: Tensor, center: Tensor, event_log_variance: Tensor
) -> Tensor:
    """Mean diagonal-Gaussian NLL for observed normal events, up to a constant."""
    stable_log_var = stable_log_variance(event_log_variance)
    variance = torch.exp(stable_log_var)
    return 0.5 * (stable_log_var + (event_mean - center) ** 2 / variance).mean()


def negative_uncertainty_nll(target: Tensor, event_log_variance: Tensor) -> Tensor:
    """Mean NLL that trains only uncertainty on sampled negative event targets."""
    if target.shape != event_log_variance.shape:
        raise ValueError("target and event_log_variance must have the same shape")
    stable_log_var = stable_log_variance(event_log_variance)
    variance = torch.exp(stable_log_var)
    return 0.5 * (stable_log_var + target**2 / variance).mean()


def robust_training_loss(
    positive_mean: Tensor,
    center: Tensor,
    positive_log_variance: Tensor,
    negative_target: Tensor,
    negative_log_variance: Tensor,
) -> Tensor:
    """Paper-recovered equal sum of separately averaged positive and negative NLLs."""
    return positive_gaussian_nll(positive_mean, center, positive_log_variance) + (
        negative_uncertainty_nll(negative_target, negative_log_variance)
    )


def initialize_center(
    event_dim: int,
    *,
    generator: torch.Generator,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Draw the trainable SVDD center from the recovered N(0, I) initialization."""
    if event_dim <= 0:
        raise ValueError("event_dim must be positive")
    return torch.randn(event_dim, generator=generator, device=device, dtype=dtype)


def sample_negative_targets(
    count: int,
    event_dim: int,
    *,
    generator: torch.Generator,
    covariance: float = 5.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Draw targets from N(0, covariance I) without touching global RNG state."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if event_dim <= 0:
        raise ValueError("event_dim must be positive")
    if covariance <= 0:
        raise ValueError("covariance must be positive")
    samples = torch.randn(
        (count, event_dim), generator=generator, device=generator.device, dtype=dtype
    )
    targets = samples * covariance**0.5
    return targets.to(device=device) if device is not None else targets

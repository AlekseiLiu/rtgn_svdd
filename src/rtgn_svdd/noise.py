"""Seeded synthetic test-noise injection for the paper workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .data import EventStream


_PAPER_DISTRIBUTION = "paper_covariance_5"
_HISTORICAL_DISTRIBUTION = "historical_scale_5"


@dataclass(frozen=True)
class NoiseInjection:
    """Injected stream and the experiment-defining choices used to create it."""

    events: EventStream
    normal_denominator: int
    injected_count: int
    noise_ratio: float
    noise_seed: int
    node_population_policy: str
    feature_distribution: str
    allow_self_edges: bool
    timestamp_tie_policy: str


def inject_test_noise(
    test: EventStream,
    *,
    node_ids: torch.Tensor,
    ratio: float,
    seed: int,
    feature_distribution: str = _PAPER_DISTRIBUTION,
    allow_self_edges: bool = False,
) -> NoiseInjection:
    """Return an independent test stream augmented with seeded random events.

    ``node_ids`` is the sorted, unique population observed in the original
    complete dataset. It is supplied explicitly so this transformation never
    needs labels or another split to discover eligible endpoints.
    """
    _validate_inputs(test, node_ids, ratio, feature_distribution, allow_self_edges)

    normal_denominator = int((~test.is_attack & ~test.is_injected).sum().item())
    injected_count = math.floor(ratio * normal_denominator + 0.5)
    if normal_denominator == 0 and ratio > 0:
        raise ValueError("cannot inject a positive noise ratio without original normal events")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_nodes = node_ids.numel()

    source_index = torch.randint(num_nodes, (injected_count,), generator=generator)
    if allow_self_edges:
        destination_index = torch.randint(num_nodes, (injected_count,), generator=generator)
    else:
        destination_index = torch.randint(num_nodes - 1, (injected_count,), generator=generator)
        destination_index += (destination_index >= source_index).to(torch.long)

    time_span = int(test.t[-1].item() - test.t[0].item()) + 1 if len(test) else 0
    injected_t = (
        torch.randint(time_span, (injected_count,), generator=generator) + int(test.t[0].item())
        if injected_count
        else torch.empty(0, dtype=torch.long)
    )
    feature_scale = math.sqrt(5.0) if feature_distribution == _PAPER_DISTRIBUTION else 5.0
    injected_msg = torch.randn(
        (injected_count, test.msg.shape[1]), generator=generator, dtype=torch.float32
    ) * feature_scale

    injected_src = node_ids[source_index].to(test.src.device)
    injected_dst = node_ids[destination_index].to(test.dst.device)
    all_t = torch.cat((test.t.clone(), injected_t.to(test.t.device)))
    all_injected = torch.cat(
        (
            test.is_injected.clone(),
            torch.ones(injected_count, dtype=torch.bool, device=test.is_injected.device),
        )
    )
    # Encode the concatenation position as a unique secondary key. This is
    # equivalent to a stable timestamp sort on Torch versions predating the
    # ``stable`` argsort argument, without synchronizing each event to Python.
    num_events = all_t.numel()
    relative_t = all_t - all_t[0]
    max_relative_t = int(test.t[-1].item() - test.t[0].item())
    if max_relative_t > (torch.iinfo(torch.long).max - num_events + 1) // num_events:
        raise ValueError("timestamp range is too large to construct a stable sort key")
    sort_key = relative_t * num_events + torch.arange(num_events, device=all_t.device)
    order = torch.argsort(sort_key)

    attack_name = None
    if test.attack_name is not None:
        combined_names = test.attack_name + (None,) * injected_count
        attack_name = tuple(combined_names[index] for index in order.cpu().tolist())

    events = EventStream(
        src=torch.cat((test.src.clone(), injected_src))[order],
        dst=torch.cat((test.dst.clone(), injected_dst))[order],
        t=all_t[order],
        msg=torch.cat((test.msg.clone(), injected_msg.to(test.msg.device)))[order],
        is_attack=torch.cat(
            (
                test.is_attack.clone(),
                torch.zeros(injected_count, dtype=torch.bool, device=test.is_attack.device),
            )
        )[order],
        is_injected=all_injected[order],
        attack_name=attack_name,
    )
    return NoiseInjection(
        events=events,
        normal_denominator=normal_denominator,
        injected_count=injected_count,
        noise_ratio=ratio,
        noise_seed=seed,
        node_population_policy="dataset_observed_nodes",
        feature_distribution=feature_distribution,
        allow_self_edges=allow_self_edges,
        timestamp_tie_policy="original_before_injected_stable",
    )


def _validate_inputs(
    test: EventStream,
    node_ids: torch.Tensor,
    ratio: float,
    feature_distribution: str,
    allow_self_edges: bool,
) -> None:
    if len(test) == 0:
        raise ValueError("test stream must contain at least one event")
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0:
        raise ValueError("noise ratio must be a finite nonnegative number")
    if node_ids.ndim != 1 or node_ids.dtype != torch.long or node_ids.device.type != "cpu":
        raise TypeError("node_ids must be a one-dimensional CPU torch.long tensor")
    if node_ids.numel() == 0 or bool(torch.any(node_ids < 0)):
        raise ValueError("node_ids must contain nonnegative node identifiers")
    if node_ids.numel() > 1 and not bool(torch.all(node_ids[1:] > node_ids[:-1])):
        raise ValueError("node_ids must be sorted and unique")
    if not allow_self_edges and node_ids.numel() < 2:
        raise ValueError("at least two node_ids are required when self-edges are disabled")
    if feature_distribution not in {_PAPER_DISTRIBUTION, _HISTORICAL_DISTRIBUTION}:
        raise ValueError(
            "feature_distribution must be 'paper_covariance_5' or 'historical_scale_5'"
        )

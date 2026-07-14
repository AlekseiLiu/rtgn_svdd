"""Small model heads shared by deterministic TGN-SVDD and rTGN-SVDD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .objectives import variance_from_log_variance


class TemporalGraphEncoder(nn.Module):
    """Shared time-aware graph-convolution step for both model variants.

    The concrete graph convolution is supplied by composition, keeping this module
    independent of temporal-state ownership and PyG version details.
    """

    def __init__(self, time_encoder: nn.Module, convolution: nn.Module) -> None:
        super().__init__()
        self.time_encoder = time_encoder
        self.convolution = convolution

    def forward(
        self,
        node_memory: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        event_time: Tensor,
        message: Tensor,
    ) -> Tensor:
        relative_time = last_update[edge_index[0]] - event_time
        encoded_time = self.time_encoder(relative_time.to(node_memory.dtype))
        edge_features = torch.cat((encoded_time, message), dim=-1)
        return self.convolution(node_memory, edge_index, edge_features)


@dataclass(frozen=True)
class ModelScores:
    """Named per-event scores used by downstream evaluation."""

    s_mu: Tensor
    s_sigma: Optional[Tensor]


@dataclass(frozen=True)
class EventModelOutput:
    """Event parameters and their anomaly scores."""

    event_mean: Tensor
    event_log_variance: Optional[Tensor]
    scores: ModelScores


class SVDDHead(nn.Module):
    """Concatenate endpoint embeddings and score distance from a center."""

    def __init__(self, endpoint_dim: int, center: Tensor) -> None:
        super().__init__()
        if center.shape != (2 * endpoint_dim,):
            raise ValueError("center must have shape [2 * endpoint_dim]")
        self.endpoint_dim = endpoint_dim
        self.center = nn.Parameter(center.detach().clone())

    def forward(self, node_embeddings: Tensor, src: Tensor, dst: Tensor) -> EventModelOutput:
        event_mean = torch.cat((node_embeddings[src], node_embeddings[dst]), dim=-1)
        s_mu = torch.sum((event_mean - self.center) ** 2, dim=-1)
        return EventModelOutput(
            event_mean=event_mean,
            event_log_variance=None,
            scores=ModelScores(s_mu=s_mu, s_sigma=None),
        )


class RobustSVDDHead(SVDDHead):
    """Build event-wise Gaussian parameters from endpoint node outputs."""

    def __init__(
        self,
        endpoint_dim: int,
        center: Tensor,
        uncertainty_mode: str = "per_dimension_log_variance",
    ) -> None:
        super().__init__(endpoint_dim=endpoint_dim, center=center)
        if uncertainty_mode not in {
            "per_dimension_log_variance",
            "historical_shared_per_node",
        }:
            raise ValueError(f"unknown uncertainty mode: {uncertainty_mode}")
        self.uncertainty_mode = uncertainty_mode

    def forward(self, node_embeddings: Tensor, src: Tensor, dst: Tensor) -> EventModelOutput:
        if node_embeddings.shape[-1] != 2 * self.endpoint_dim:
            raise ValueError("robust node embeddings must have shape [num_nodes, 2 * endpoint_dim]")
        node_mean, node_log_variance = torch.chunk(node_embeddings, 2, dim=-1)
        if self.uncertainty_mode == "historical_shared_per_node":
            node_log_variance = node_log_variance.mean(dim=-1, keepdim=True).expand_as(node_mean)

        event_mean = torch.cat((node_mean[src], node_mean[dst]), dim=-1)
        event_log_variance = torch.cat(
            (node_log_variance[src], node_log_variance[dst]), dim=-1
        )
        scores = ModelScores(
            s_mu=torch.sum((event_mean - self.center) ** 2, dim=-1),
            s_sigma=variance_from_log_variance(event_log_variance).mean(dim=-1),
        )
        return EventModelOutput(event_mean, event_log_variance, scores)

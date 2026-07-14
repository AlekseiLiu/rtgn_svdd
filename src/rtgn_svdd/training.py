"""Training and chronological temporal-state traversal for both model variants."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastNeighborLoader,
    MeanAggregator,
    TGNMemory,
)

from .data import EventStream
from .models import (
    EventModelOutput,
    ModelScores,
    RobustSVDDHead,
    SVDDHead,
    TemporalGraphEncoder,
)
from .objectives import (
    deterministic_svdd_loss,
    positive_gaussian_nll,
    robust_training_loss,
    sample_negative_targets,
)


Variant = Literal["baseline", "robust"]


@dataclass(frozen=True)
class TrainingConfig:
    """Small, versionable optimization and traversal configuration."""

    variant: Variant
    batch_size: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    negative_ratio: float = 0.3
    negative_covariance: float = 5.0
    update_injected_state: bool = True

    def __post_init__(self) -> None:
        if self.variant not in {"baseline", "robust"}:
            raise ValueError("variant must be 'baseline' or 'robust'")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if not 0 <= self.negative_ratio <= 1:
            raise ValueError("negative_ratio must be between zero and one")
        if self.negative_covariance <= 0:
            raise ValueError("negative_covariance must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    epoch: int
    config: dict[str, object]


class TGNModel(nn.Module):
    """Concrete TGN memory, graph attention, and SVDD head composition."""

    def __init__(
        self,
        *,
        num_nodes: int,
        msg_dim: int,
        memory_dim: int,
        time_dim: int,
        endpoint_dim: int,
        center: Tensor,
        variant: Variant,
        neighbor_size: int = 10,
        uncertainty_mode: str = "per_dimension_log_variance",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if variant not in {"baseline", "robust"}:
            raise ValueError("variant must be 'baseline' or 'robust'")
        if min(num_nodes, msg_dim, memory_dim, time_dim, endpoint_dim, neighbor_size) <= 0:
            raise ValueError("model dimensions and neighbor_size must be positive")
        output_dim = endpoint_dim if variant == "baseline" else 2 * endpoint_dim
        if output_dim % 8:
            raise ValueError("the variant encoder output dimension must be divisible by 8")

        self.variant = variant
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.memory = TGNMemory(
            num_nodes,
            msg_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=MeanAggregator(),
        )
        convolution = TransformerConv(
            memory_dim,
            output_dim // 8,
            heads=8,
            dropout=0.1,
            edge_dim=msg_dim + time_dim,
        )
        self.encoder = TemporalGraphEncoder(self.memory.time_enc, convolution)
        if variant == "baseline":
            self.head: SVDDHead = SVDDHead(endpoint_dim, center)
        else:
            self.head = RobustSVDDHead(endpoint_dim, center, uncertainty_mode)

        target_device = torch.device(device) if device is not None else center.device
        self.to(target_device)
        self.neighbor_loader = LastNeighborLoader(
            num_nodes, size=neighbor_size, device=target_device
        )
        self.register_buffer(
            "assoc",
            torch.empty(num_nodes, dtype=torch.long, device=target_device),
            persistent=False,
        )
        self.reset_state()

    @property
    def device(self) -> torch.device:
        return self.head.center.device

    @property
    def state_event_count(self) -> int:
        """Number of real or injected events currently represented in graph state."""
        return self._history_t.numel()

    def reset_state(self) -> None:
        """Clear all transient memory, graph, and inserted-edge history."""
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        self._history_t = torch.empty(0, dtype=torch.long, device=self.device)
        self._history_msg = torch.empty(
            (0, self.msg_dim), dtype=torch.float32, device=self.device
        )

    def detach_state(self) -> None:
        self.memory.detach()

    def _outputs_for_pairs(self, embeddings: Tensor, src: Tensor, dst: Tensor) -> EventModelOutput:
        return self.head(embeddings, self.assoc[src], self.assoc[dst])

    def _encode(self, src: Tensor, dst: Tensor) -> Tensor:
        query_nodes = torch.cat((src, dst)).unique()
        n_id, edge_index, e_id = self.neighbor_loader(query_nodes)
        self.assoc[n_id] = torch.arange(n_id.numel(), device=self.device)
        memory, last_update = self.memory(n_id)
        return self.encoder(
            memory,
            last_update,
            edge_index,
            self._history_t[e_id],
            self._history_msg[e_id],
        )

    def forward_events(
        self,
        events: EventStream,
        *,
        update_state: bool = True,
        update_injected_state: bool = True,
    ) -> EventModelOutput:
        """Score one contiguous batch, then update its selected events as a group.

        Events within the batch never observe one another. The next batch observes
        either all events or the order-preserving non-injected subset.
        """
        src = events.src.to(self.device)
        dst = events.dst.to(self.device)
        output = self._outputs_for_pairs(self._encode(src, dst), src, dst)
        if update_state:
            self.update_events(events, update_injected_state=update_injected_state)
        return output

    def score_pairs(self, src: Tensor, dst: Tensor) -> EventModelOutput:
        """Score auxiliary pairs against current state without updating it."""
        src, dst = src.to(self.device), dst.to(self.device)
        return self._outputs_for_pairs(self._encode(src, dst), src, dst)

    def update_events(self, events: EventStream, *, update_injected_state: bool) -> None:
        """Append and insert exactly the selected batch events in original order."""
        keep = torch.ones(len(events), dtype=torch.bool, device=self.device)
        if not update_injected_state:
            keep = ~events.is_injected.to(self.device)
        if not bool(keep.any()):
            return
        src, dst = events.src.to(self.device)[keep], events.dst.to(self.device)[keep]
        t, msg = events.t.to(self.device)[keep], events.msg.to(self.device)[keep]
        self._history_t = torch.cat((self._history_t, t))
        self._history_msg = torch.cat((self._history_msg, msg))
        self.memory.update_state(src, dst, t, msg)
        self.neighbor_loader.insert(src, dst)


def _batches(events: EventStream, batch_size: int):
    for start in range(0, len(events), batch_size):
        yield events.slice(start, min(start + batch_size, len(events)))


def _concatenate(outputs: list[EventModelOutput]) -> EventModelOutput:
    if not outputs:
        raise ValueError("cannot concatenate an empty stream")
    log_variance = None
    s_sigma = None
    if outputs[0].event_log_variance is not None:
        log_variance = torch.cat(
            [
                item.event_log_variance
                for item in outputs
                if item.event_log_variance is not None
            ]
        )
        s_sigma = torch.cat(
            [item.scores.s_sigma for item in outputs if item.scores.s_sigma is not None]
        )
    return EventModelOutput(
        event_mean=torch.cat([item.event_mean for item in outputs]),
        event_log_variance=log_variance,
        scores=ModelScores(
            s_mu=torch.cat([item.scores.s_mu for item in outputs]),
            s_sigma=s_sigma,
        ),
    )


def _offload_output(output: EventModelOutput) -> EventModelOutput:
    return EventModelOutput(
        event_mean=output.event_mean.cpu(),
        event_log_variance=(
            None
            if output.event_log_variance is None
            else output.event_log_variance.cpu()
        ),
        scores=ModelScores(
            s_mu=output.scores.s_mu.cpu(),
            s_sigma=(
                None if output.scores.s_sigma is None else output.scores.s_sigma.cpu()
            ),
        ),
    )


@torch.no_grad()
def score_stream(
    model: TGNModel,
    events: EventStream,
    *,
    batch_size: int,
    update_injected_state: bool = True,
) -> EventModelOutput:
    """Score chronologically and accumulate completed batch outputs on CPU."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    try:
        outputs = [
            _offload_output(
                model.forward_events(
                    batch, update_injected_state=update_injected_state
                )
            )
            for batch in _batches(events, batch_size)
        ]
    finally:
        model.train(was_training)
    return _concatenate(outputs)


def sample_negative_pairs(
    training_nodes: Tensor,
    count: int,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Uniformly sample non-self pairs from actual training-observed nodes."""
    nodes = torch.unique(training_nodes)
    if nodes.ndim != 1 or nodes.numel() < 2:
        raise ValueError("training_nodes must contain at least two distinct node IDs")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if count == 0:
        empty = nodes.new_empty(0)
        return empty, empty
    src_idx = torch.randint(nodes.numel(), (count,), generator=generator)
    dst_idx = torch.randint(nodes.numel() - 1, (count,), generator=generator)
    dst_idx += (dst_idx >= src_idx).to(dst_idx.dtype)
    return nodes[src_idx].to(training_nodes.device), nodes[dst_idx].to(training_nodes.device)


def create_optimizer(model: TGNModel, config: TrainingConfig) -> torch.optim.Adam:
    if model.variant != config.variant:
        raise ValueError("model and training variants must match")
    return torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def train_epoch(
    model: TGNModel,
    events: EventStream,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    *,
    generator: torch.Generator,
    training_nodes: Tensor | None = None,
) -> float:
    """Train one epoch from fresh state using contiguous chronological batches."""
    if len(events) == 0:
        raise ValueError("cannot train on an empty event stream")
    if model.variant != config.variant:
        raise ValueError("model and training variants must match")
    if config.variant == "robust" and training_nodes is None:
        training_nodes = torch.unique(torch.cat((events.src, events.dst)))

    model.train()
    model.reset_state()
    total_loss = 0.0
    total_events = 0
    for batch in _batches(events, config.batch_size):
        optimizer.zero_grad()
        positive = model.forward_events(batch, update_state=False)
        if config.variant == "baseline":
            loss = deterministic_svdd_loss(positive.scores.s_mu)
        else:
            assert positive.event_log_variance is not None
            count = int(len(batch) * config.negative_ratio)
            if count:
                assert training_nodes is not None
                neg_src, neg_dst = sample_negative_pairs(
                    training_nodes, count, generator=generator
                )
                negative = model.score_pairs(neg_src, neg_dst)
                assert negative.event_log_variance is not None
                targets = sample_negative_targets(
                    count,
                    negative.event_mean.shape[-1],
                    generator=generator,
                    covariance=config.negative_covariance,
                    device=model.device,
                    dtype=negative.event_mean.dtype,
                )
                loss = robust_training_loss(
                    positive.event_mean,
                    model.head.center,
                    positive.event_log_variance,
                    targets,
                    negative.event_log_variance,
                )
            else:
                loss = positive_gaussian_nll(
                    positive.event_mean, model.head.center, positive.event_log_variance
                )
        model.update_events(batch, update_injected_state=config.update_injected_state)
        loss.backward()
        optimizer.step()
        model.detach_state()
        total_loss += float(loss.detach()) * len(batch)
        total_events += len(batch)
    return total_loss / total_events


def replay_splits(
    model: TGNModel,
    train: EventStream,
    validation: EventStream,
    test: EventStream,
    *,
    batch_size: int,
    update_injected_state: bool = True,
) -> tuple[EventModelOutput, EventModelOutput, EventModelOutput]:
    """Rebuild fresh state and continuously traverse train, validation, then test."""
    model.reset_state()
    return (
        score_stream(
            model,
            train,
            batch_size=batch_size,
            update_injected_state=update_injected_state,
        ),
        score_stream(
            model,
            validation,
            batch_size=batch_size,
            update_injected_state=update_injected_state,
        ),
        score_stream(
            model,
            test,
            batch_size=batch_size,
            update_injected_state=update_injected_state,
        ),
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: TGNModel,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, object],
    epoch: int,
) -> None:
    """Atomically save learned parameters while excluding transient TGN state."""
    learned = {name: value.detach().cpu() for name, value in model.named_parameters()}
    payload = {
        "schema_version": 1,
        "epoch": epoch,
        "config": dict(config),
        "learned_parameters": learned,
        "optimizer": optimizer.state_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    model: TGNModel,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
) -> CheckpointMetadata:
    """Restore learned parameters and optional optimizer, then clear transient state."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema_version")
    saved = checkpoint["learned_parameters"]
    current = dict(model.named_parameters())
    if set(saved) != set(current):
        raise ValueError("checkpoint learned parameter keys do not match the model")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(saved[name].to(parameter.device))
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    model.reset_state()
    return CheckpointMetadata(
        schema_version=1,
        epoch=int(checkpoint["epoch"]),
        config=dict(checkpoint["config"]),
    )

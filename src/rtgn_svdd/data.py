"""Event loading and chronological splitting for the CIC-IDS2017 workflow."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch


_REQUIRED_COLUMNS = {
    "src_ip",
    "dst_ip",
    "bidirectional_first_seen_ms",
    "label",
    "attack",
}
_NON_FEATURE_COLUMNS = _REQUIRED_COLUMNS | {
    "ip_version",
    "tunnel_id",
    "application_category_name",
    "protocol",
    "application_name",
    "client_fingerprint",
    "server_fingerprint",
    "application_is_guessed",
}


@dataclass(frozen=True)
class EventStream:
    """Chronologically ordered temporal graph events shared by both models."""

    src: torch.Tensor
    dst: torch.Tensor
    t: torch.Tensor
    msg: torch.Tensor
    is_attack: torch.Tensor
    is_injected: torch.Tensor
    attack_name: tuple[str | None, ...] | None = None

    def __post_init__(self) -> None:
        num_events = self.src.shape[0]
        one_dimensional = {
            "src": self.src,
            "dst": self.dst,
            "t": self.t,
            "is_attack": self.is_attack,
            "is_injected": self.is_injected,
        }
        for name, tensor in one_dimensional.items():
            if tensor.ndim != 1 or tensor.shape[0] != num_events:
                raise ValueError(f"{name} must have shape [num_events]")
        if self.msg.ndim != 2 or self.msg.shape[0] != num_events:
            raise ValueError("msg must have shape [num_events, num_features]")
        if self.src.dtype != torch.long or self.dst.dtype != torch.long or self.t.dtype != torch.long:
            raise TypeError("src, dst, and t must use torch.long")
        if self.msg.dtype != torch.float32:
            raise TypeError("msg must use torch.float32")
        if self.is_attack.dtype != torch.bool or self.is_injected.dtype != torch.bool:
            raise TypeError("is_attack and is_injected must use torch.bool")
        if self.attack_name is not None and len(self.attack_name) != num_events:
            raise ValueError("attack_name must contain one value per event")
        if num_events > 1 and bool(torch.any(self.t[1:] < self.t[:-1])):
            raise ValueError("events must be ordered by nondecreasing timestamp")
        if bool(torch.any(self.src < 0)) or bool(torch.any(self.dst < 0)):
            raise ValueError("node identifiers must be nonnegative")
        if not bool(torch.all(torch.isfinite(self.msg))):
            raise ValueError("msg must contain only finite values")
        if bool(torch.any(self.is_attack & self.is_injected)):
            raise ValueError("an injected event cannot be an attack target")

    def __len__(self) -> int:
        return self.src.shape[0]

    def slice(self, start: int, stop: int) -> EventStream:
        """Return an independent contiguous part of this stream."""
        attack_name = None
        if self.attack_name is not None:
            attack_name = self.attack_name[start:stop]
        return EventStream(
            src=self.src[start:stop].clone(),
            dst=self.dst[start:stop].clone(),
            t=self.t[start:stop].clone(),
            msg=self.msg[start:stop].clone(),
            is_attack=self.is_attack[start:stop].clone(),
            is_injected=self.is_injected[start:stop].clone(),
            attack_name=attack_name,
        )


@dataclass(frozen=True)
class EventSplits:
    """Contiguous chronological training, validation, and test streams."""

    train: EventStream
    validation: EventStream
    test: EventStream


def split_chronologically(
    events: EventStream,
    *,
    num_train: int = 200_000,
    num_validation: int = 70_000,
) -> EventSplits:
    """Split by fixed event counts, leaving at least one test event."""
    if num_train <= 0 or num_validation <= 0:
        raise ValueError("num_train and num_validation must be positive")
    test_start = num_train + num_validation
    if len(events) <= test_start:
        raise ValueError(
            "dataset must contain at least one test event after "
            f"{num_train} training and {num_validation} validation events"
        )
    return EventSplits(
        train=events.slice(0, num_train),
        validation=events.slice(num_train, test_start),
        test=events.slice(test_start, len(events)),
    )


def load_cic_csv(
    path: str | Path,
    *,
    feature_columns: Sequence[str] | None = None,
) -> EventStream:
    """Load a paper-preprocessed, named-column CIC-IDS2017 CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CIC event CSV does not exist: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if columns is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = sorted(_REQUIRED_COLUMNS - set(columns))
        if missing:
            raise ValueError(f"CSV {csv_path} is missing required columns: {', '.join(missing)}")

        if feature_columns is None:
            selected_features = [name for name in columns if name not in _NON_FEATURE_COLUMNS]
        else:
            selected_features = list(feature_columns)
            missing_features = [name for name in selected_features if name not in columns]
            if missing_features:
                raise ValueError(
                    f"CSV {csv_path} is missing requested feature columns: "
                    f"{', '.join(missing_features)}"
                )
        if not selected_features:
            raise ValueError(f"CSV {csv_path} provides no message feature columns")

        src: list[int] = []
        dst: list[int] = []
        timestamps: list[int] = []
        labels: list[bool] = []
        messages: list[list[float]] = []
        attack_names: list[str | None] = []
        for line_number, row in enumerate(reader, start=2):
            src.append(_parse_integer(row["src_ip"], "src_ip", csv_path, line_number))
            dst.append(_parse_integer(row["dst_ip"], "dst_ip", csv_path, line_number))
            timestamps.append(
                _parse_integer(
                    row["bidirectional_first_seen_ms"],
                    "bidirectional_first_seen_ms",
                    csv_path,
                    line_number,
                )
            )
            label = row["label"].strip()
            if label not in {"0", "1"}:
                raise ValueError(
                    f"CSV {csv_path} line {line_number}: label must be 0 or 1, got {label!r}"
                )
            labels.append(label == "1")
            attack_name = row["attack"].strip()
            attack_names.append(attack_name or None)
            messages.append(
                [
                    _parse_feature(row[name], name, csv_path, line_number)
                    for name in selected_features
                ]
            )

    if not src:
        raise ValueError(f"CSV {csv_path} contains no event rows")

    return EventStream(
        src=torch.tensor(src, dtype=torch.long),
        dst=torch.tensor(dst, dtype=torch.long),
        t=torch.tensor(timestamps, dtype=torch.long),
        msg=torch.tensor(messages, dtype=torch.float32),
        is_attack=torch.tensor(labels, dtype=torch.bool),
        is_injected=torch.zeros(len(src), dtype=torch.bool),
        attack_name=tuple(attack_names),
    )


def _parse_integer(value: str, column: str, path: Path, line_number: int) -> int:
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"CSV {path} line {line_number}: {column} must be numeric, got {value!r}"
        ) from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(
            f"CSV {path} line {line_number}: {column} must be a finite integer, got {value!r}"
        )
    return int(number)


def _parse_feature(value: str, column: str, path: Path, line_number: int) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"CSV {path} line {line_number}: feature {column} must be numeric, got {value!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(
            f"CSV {path} line {line_number}: feature {column} must be finite, got {value!r}"
        )
    return number

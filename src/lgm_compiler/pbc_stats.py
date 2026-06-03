"""Pauli-weight statistics for PBC JSONL files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PAULI_SYMBOLS = frozenset({"I", "X", "Y", "Z"})
PREFERRED_KEYS = ("basis", "pauli", "P", "pauli_string", "ops", "rotation", "Rotation", "measurement", "Measurement")


@dataclass
class PBCStatsReport:
    num_rotations: int = 0
    pauli_weight_histogram: dict[int, int] = field(default_factory=dict)
    num_weight_1: int = 0
    num_weight_2: int = 0
    num_weight_gt_2: int = 0
    mean_weight: float = 0.0
    max_weight: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "num_rotations": self.num_rotations,
            "pauli_weight_histogram": dict(sorted(self.pauli_weight_histogram.items())),
            "num_weight_1": self.num_weight_1,
            "num_weight_2": self.num_weight_2,
            "num_weight_gt_2": self.num_weight_gt_2,
            "mean_weight": self.mean_weight,
            "max_weight": self.max_weight,
        }


class PBCStatsError(ValueError):
    pass


def load_pbc_stats(path: str | Path) -> PBCStatsReport:
    pbc_path = Path(path)
    weights: list[int] = []

    for line_number, raw_line in enumerate(pbc_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PBCStatsError(f"invalid JSON at {pbc_path}:{line_number}") from exc

        pauli = infer_pauli_string(record)
        if pauli is None:
            keys = sorted(record.keys()) if isinstance(record, dict) else [type(record).__name__]
            raise PBCStatsError(f"could not infer Pauli string at {pbc_path}:{line_number}; keys={keys}")
        weights.append(sum(1 for symbol in pauli if symbol != "I"))

    return pbc_stats_from_weights(weights)


def pbc_stats_from_weights(weights: list[int]) -> PBCStatsReport:
    histogram: dict[int, int] = {}
    for weight in weights:
        histogram[weight] = histogram.get(weight, 0) + 1

    total = len(weights)
    return PBCStatsReport(
        num_rotations=total,
        pauli_weight_histogram=histogram,
        num_weight_1=histogram.get(1, 0),
        num_weight_2=histogram.get(2, 0),
        num_weight_gt_2=sum(count for weight, count in histogram.items() if weight > 2),
        mean_weight=(sum(weights) / total if total else 0.0),
        max_weight=(max(weights) if weights else 0),
    )


def infer_pauli_string(value: Any) -> list[str] | None:
    parsed = _pauli_from_value(value)
    if parsed is not None:
        return parsed

    if isinstance(value, dict):
        for key in PREFERRED_KEYS:
            if key in value:
                parsed = infer_pauli_string(value[key])
                if parsed is not None:
                    return parsed
        for child in value.values():
            parsed = infer_pauli_string(child)
            if parsed is not None:
                return parsed

    if isinstance(value, list):
        for child in value:
            parsed = infer_pauli_string(child)
            if parsed is not None:
                return parsed

    return None


def _pauli_from_value(value: Any) -> list[str] | None:
    if isinstance(value, list):
        if value and all(isinstance(item, str) and item.upper() in PAULI_SYMBOLS for item in value):
            return [item.upper() for item in value]
        return _pauli_from_ops_list(value)

    if isinstance(value, str):
        compact = re.sub(r"[\s,_|]+", "", value).upper()
        if compact and all(char in PAULI_SYMBOLS for char in compact):
            return list(compact)
        return None

    if isinstance(value, dict):
        indexed = _pauli_from_indexed_dict(value)
        if indexed is not None:
            return indexed

    return None


def _pauli_from_indexed_dict(value: dict[Any, Any]) -> list[str] | None:
    entries: dict[int, str] = {}
    for key, item in value.items():
        if not isinstance(item, str) or item.upper() not in PAULI_SYMBOLS:
            return None
        try:
            index = int(key)
        except (TypeError, ValueError):
            return None
        entries[index] = item.upper()

    if not entries:
        return None
    max_index = max(entries)
    return [entries.get(index, "I") for index in range(max_index + 1)]


def _pauli_from_ops_list(value: list[Any]) -> list[str] | None:
    entries: dict[int, str] = {}
    for item in value:
        if not isinstance(item, dict):
            return None
        axis = item.get("pauli") or item.get("axis") or item.get("P")
        qubit = item.get("qubit") or item.get("q") or item.get("index")
        if not isinstance(axis, str) or axis.upper() not in PAULI_SYMBOLS:
            return None
        try:
            index = int(qubit)
        except (TypeError, ValueError):
            return None
        entries[index] = axis.upper()

    if not entries:
        return None
    max_index = max(entries)
    return [entries.get(index, "I") for index in range(max_index + 1)]

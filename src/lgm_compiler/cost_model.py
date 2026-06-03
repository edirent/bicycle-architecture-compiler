"""Small gross-code lowering cost model for LGM-IR reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from lgm_compiler.lgm_ir import (
    CNOT,
    FallbackPBC,
    FrameUpdate,
    FramedCNOT,
    FramedToffoli,
    LGMOp,
    Measure1,
    Measure2,
    Rot,
    Toffoli,
)

PIVOT_INDICES = frozenset({1, 7})


@dataclass(frozen=True)
class Placement:
    module_id: Mapping[int, int]
    logical_index: Mapping[int, int]

    def module(self, q: int) -> int:
        return self.module_id[q]

    def index(self, q: int) -> int:
        return self.logical_index[q]


@dataclass
class CostReport:
    direct_t_injection: int = 0
    nonpivot_rotation: int = 0
    fallback_or_route: int = 0
    in_module_measurement: int = 0
    inter_module_measurement: int = 0
    cnot_macro: int = 0
    toffoli_macro: int = 0
    fallback_pbc: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "direct_t_injection": self.direct_t_injection,
            "nonpivot_rotation": self.nonpivot_rotation,
            "fallback_or_route": self.fallback_or_route,
            "in_module_measurement": self.in_module_measurement,
            "inter_module_measurement": self.inter_module_measurement,
            "cnot_macro": self.cnot_macro,
            "toffoli_macro": self.toffoli_macro,
            "fallback_pbc": self.fallback_pbc,
        }


def estimate_gross_cost(ir_ops: list[LGMOp], placement: Placement) -> CostReport:
    report = CostReport()

    for op in ir_ops:
        if isinstance(op, Rot):
            _count_rotation(op, placement, report)
        elif isinstance(op, Measure2):
            if placement.module(op.q1) == placement.module(op.q2):
                report.in_module_measurement += 1
            else:
                report.inter_module_measurement += 1
        elif isinstance(op, (CNOT, FramedCNOT)):
            report.cnot_macro += 1
        elif isinstance(op, (Toffoli, FramedToffoli)):
            report.toffoli_macro += 1
        elif isinstance(op, FallbackPBC):
            report.fallback_pbc += 1
        elif isinstance(op, (FrameUpdate, Measure1)):
            continue
        else:
            raise TypeError(f"unknown LGM operation {op!r}")

    return report


def _count_rotation(op: Rot, placement: Placement, report: CostReport) -> None:
    if op.a in (-1, 1):
        if placement.index(op.q) in PIVOT_INDICES:
            report.direct_t_injection += 1
        else:
            report.nonpivot_rotation += 1
            report.fallback_or_route += 1
        return

    report.fallback_or_route += 1

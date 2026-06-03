"""Clifford accounting for compiled LGM circuits."""

from __future__ import annotations

from dataclasses import dataclass

from lgm_compiler.compiler import CompileResult
from lgm_compiler.input_ir import Circuit
from lgm_compiler.lgm_ir import CNOT, FallbackPBC, FrameUpdate, FramedCNOT, FramedToffoli


@dataclass
class CliffordAccountingReport:
    input_h: int = 0
    input_s: int = 0
    input_sdg: int = 0
    input_x: int = 0
    input_y: int = 0
    input_z: int = 0
    input_cnot: int = 0
    input_1q_clifford: int = 0
    input_entangling_clifford_macros: int = 0
    explicit_1q_clifford_after_lgm: int = 0
    final_nonidentity_frames: int = 0
    physical_1q_clifford_remaining: int = 0
    framed_cnot_macros: int = 0
    framed_toffoli_macros: int = 0
    frame_updates: int = 0
    fallback_pbc: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_h": self.input_h,
            "input_s": self.input_s,
            "input_sdg": self.input_sdg,
            "input_x": self.input_x,
            "input_y": self.input_y,
            "input_z": self.input_z,
            "input_cnot": self.input_cnot,
            "input_1q_clifford": self.input_1q_clifford,
            "input_entangling_clifford_macros": self.input_entangling_clifford_macros,
            "explicit_1q_clifford_after_lgm": self.explicit_1q_clifford_after_lgm,
            "final_nonidentity_frames": self.final_nonidentity_frames,
            "physical_1q_clifford_remaining": self.physical_1q_clifford_remaining,
            "framed_cnot_macros": self.framed_cnot_macros,
            "framed_toffoli_macros": self.framed_toffoli_macros,
            "frame_updates": self.frame_updates,
            "fallback_pbc": self.fallback_pbc,
        }


def account_cliffords(circuit: Circuit, result: CompileResult) -> CliffordAccountingReport:
    """Count input and residual Clifford work after LGM compilation.

    Final frames are implicit bookkeeping and are not counted as physical
    one-qubit Clifford gates. Structured CNOT macros are reported separately as
    entangling Clifford work. Toffoli macros are reported as macros, not as
    Clifford gates.
    """

    report = CliffordAccountingReport()
    _count_input_cliffords(circuit, report)
    _count_lgm_residuals(result, report)
    report.physical_1q_clifford_remaining = report.explicit_1q_clifford_after_lgm
    return report


def _count_input_cliffords(circuit: Circuit, report: CliffordAccountingReport) -> None:
    for gate in circuit.gates:
        if gate.name == "H":
            report.input_h += 1
        elif gate.name == "S":
            report.input_s += 1
        elif gate.name == "Sdg":
            report.input_sdg += 1
        elif gate.name == "X":
            report.input_x += 1
        elif gate.name == "Y":
            report.input_y += 1
        elif gate.name == "Z":
            report.input_z += 1
        elif gate.name == "CNOT":
            report.input_cnot += 1

    report.input_1q_clifford = (
        report.input_h
        + report.input_s
        + report.input_sdg
        + report.input_x
        + report.input_y
        + report.input_z
    )
    report.input_entangling_clifford_macros = report.input_cnot


def _count_lgm_residuals(result: CompileResult, report: CliffordAccountingReport) -> None:
    report.final_nonidentity_frames = sum(1 for frame in result.final_frames.values() if not frame.is_identity())

    for op in result.ir_ops:
        if isinstance(op, (CNOT, FramedCNOT)):
            report.framed_cnot_macros += 1
        elif isinstance(op, FramedToffoli):
            report.framed_toffoli_macros += 1
        elif isinstance(op, FrameUpdate):
            report.frame_updates += 1
        elif isinstance(op, FallbackPBC):
            report.fallback_pbc += 1

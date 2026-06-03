"""Local gate-measurement MVP compiler for gross-code experiments."""

from lgm_compiler.accounting import CliffordAccountingReport, account_cliffords
from lgm_compiler.compiler import CompileResult, compile_circuit, lower_framed_cnot_to_measurements
from lgm_compiler.cost_model import CostReport, Placement, estimate_gross_cost
from lgm_compiler.dataset_eval import BenchmarkEvalReport, evaluate_dataset
from lgm_compiler.frame import CliffordFrame
from lgm_compiler.input_ir import Circuit, Gate
from lgm_compiler.lgm_ir import (
    CNOT,
    FallbackPBC,
    FrameUpdate,
    FramedCNOT,
    FramedToffoli,
    Measure1,
    Measure2,
    Rot,
    Toffoli,
)
from lgm_compiler.pauli import SignedPauli
from lgm_compiler.pbc_stats import PBCStatsReport, load_pbc_stats
from lgm_compiler.qasm_loader import QASMLoadResult, UnsupportedGate, load_qasm

__all__ = [
    "BenchmarkEvalReport",
    "CNOT",
    "Circuit",
    "CliffordFrame",
    "CliffordAccountingReport",
    "CompileResult",
    "CostReport",
    "FallbackPBC",
    "FrameUpdate",
    "FramedCNOT",
    "FramedToffoli",
    "Gate",
    "Measure1",
    "Measure2",
    "PBCStatsReport",
    "Placement",
    "QASMLoadResult",
    "Rot",
    "SignedPauli",
    "Toffoli",
    "UnsupportedGate",
    "account_cliffords",
    "compile_circuit",
    "estimate_gross_cost",
    "evaluate_dataset",
    "load_pbc_stats",
    "load_qasm",
    "lower_framed_cnot_to_measurements",
]

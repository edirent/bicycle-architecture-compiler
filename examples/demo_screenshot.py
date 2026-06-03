#!/usr/bin/env python3
"""Run the HTHTHTSHTHTSHT LGM compiler demo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lgm_compiler.compiler import compile_circuit
from lgm_compiler.cost_model import Placement, estimate_gross_cost
from lgm_compiler.input_ir import Circuit, parse_single_qubit_sequence
from lgm_compiler.verifier import verification_error


def main() -> None:
    text = "H T H T H T S H T H T S H T"
    gates = parse_single_qubit_sequence(text, q=0, operator_order=True)
    circuit = Circuit(n_qubits=1, gates=gates)
    result = compile_circuit(circuit)
    cost = estimate_gross_cost(
        result.ir_ops,
        Placement(module_id={0: 0}, logical_index={0: 1}),
    )
    error = verification_error(circuit, result)

    print("Input gates:")
    print("  " + text)
    print("Application order:")
    print("  " + " ".join(g.name for g in gates))
    print()
    print("Emitted rotations:")
    for rot in result.rotations:
        print(f"  {rot.describe()}")
    print()
    print("Final frame:")
    print(f"  q=0: {result.final_frames[0]}")
    print()
    print("Gross cost, q=0 at module 0 logical index 1:")
    for key, value in cost.as_dict().items():
        print(f"  {key}: {value}")
    print()
    print("Verification error up to global phase:")
    print(f"  {error:.3e}")


if __name__ == "__main__":
    main()

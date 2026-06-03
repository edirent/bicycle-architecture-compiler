#!/usr/bin/env python3
"""Count Clifford gates and residual Clifford structures for the screenshot circuit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lgm_compiler.accounting import account_cliffords
from lgm_compiler.compiler import compile_circuit
from lgm_compiler.input_ir import Circuit, parse_single_qubit_sequence


def main() -> None:
    text = "H T H T H T S H T H T S H T"
    gates = parse_single_qubit_sequence(text, q=0, operator_order=True)
    circuit = Circuit(n_qubits=1, gates=gates)
    result = compile_circuit(circuit)
    report = account_cliffords(circuit, result)

    print("Input gates:")
    print("  " + text)
    print("Application order:")
    print("  " + " ".join(g.name for g in gates))
    print()
    print("Clifford accounting:")
    for key, value in report.as_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

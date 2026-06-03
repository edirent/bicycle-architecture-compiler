#!/usr/bin/env python3
# Copyright contributors to the Bicycle Architecture Compiler project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert a QASMBench OpenQASM circuit to bicycle-compiler PBC JSONL."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, transpile
import qiskit
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.transpiler.passes import LitinskiTransformation

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qiskit_parser import iter_qiskit_pbc_circuit  # noqa: E402

BASIS_GATES = ["rz", "t", "tdg"] + list(get_clifford_gate_names())
OPTIMIZATION_LEVEL = 0


def _count_ops(circuit: QuantumCircuit) -> dict[str, int]:
    return {str(name): int(count) for name, count in circuit.count_ops().items()}


def _load_qasm(path: Path) -> QuantumCircuit:
    try:
        return QuantumCircuit.from_qasm_file(str(path))
    except Exception:
        try:
            from qiskit import qasm2

            return qasm2.load(str(path))
        except Exception:
            raise


def _remove_final_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    try:
        return circuit.remove_final_measurements(inplace=False)
    except TypeError:
        clone = circuit.copy()
        clone.remove_final_measurements()
        return clone


def _build_metadata(
    qasm_path: Path,
    keep_final_measurements: bool,
    original: QuantumCircuit,
    converted: QuantumCircuit,
    transpiled: QuantumCircuit,
    pbc: QuantumCircuit,
    pbc_counts: dict[str, int],
) -> dict[str, Any]:
    original_ops = _count_ops(original)
    converted_ops = _count_ops(converted)
    transpiled_ops = _count_ops(transpiled)

    return {
        "qasm_path": str(qasm_path),
        "logical_qubits": int(original.num_qubits),
        "original_ops": original_ops,
        "converted_ops": converted_ops,
        "transpiled_ops": transpiled_ops,
        "original_cx_count": int(original_ops.get("cx", 0)),
        "original_rz_count": int(original_ops.get("rz", 0)),
        "transpiled_rz_count": int(transpiled_ops.get("rz", 0)),
        "pbc_rotation_count": int(pbc_counts["Rotation"]),
        "pbc_measurement_count": int(pbc_counts["Measurement"]),
        "pbc_instruction_count": int(sum(pbc_counts.values())),
        "include_final_measurements": bool(keep_final_measurements),
        "removed_final_measurements": not bool(keep_final_measurements),
        "qiskit_version": getattr(qiskit, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "basis_gates": BASIS_GATES,
        "optimization_level": OPTIMIZATION_LEVEL,
        "litinski_fix_clifford": False,
        "original_depth": int(original.depth()),
        "converted_depth": int(converted.depth()),
        "transpiled_depth": int(transpiled.depth()),
        "pbc_depth": int(pbc.depth()),
        "original_num_clbits": int(original.num_clbits),
        "converted_num_clbits": int(converted.num_clbits),
    }


def convert_qasm(path: Path, keep_final_measurements: bool) -> tuple[list[str], dict[str, Any]]:
    original = _load_qasm(path)
    converted = original if keep_final_measurements else _remove_final_measurements(original)

    transpiled = transpile(
        converted,
        basis_gates=BASIS_GATES,
        optimization_level=OPTIMIZATION_LEVEL,
    )

    litinski = LitinskiTransformation(fix_clifford=False)
    pbc = litinski(transpiled)

    lines: list[str] = []
    pbc_counts = {"Rotation": 0, "Measurement": 0}
    for instruction in iter_qiskit_pbc_circuit(pbc):
        if "Rotation" in instruction:
            pbc_counts["Rotation"] += 1
        elif "Measurement" in instruction:
            pbc_counts["Measurement"] += 1
        lines.append(json.dumps(instruction, separators=(",", ":")))

    metadata = _build_metadata(
        path,
        keep_final_measurements,
        original,
        converted,
        transpiled,
        pbc,
        pbc_counts,
    )
    return lines, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one OpenQASM file and print bicycle PBC JSONL to stdout.",
    )
    parser.add_argument("qasm_file", type=Path)
    parser.add_argument(
        "--keep-final-measurements",
        action="store_true",
        help="Keep trailing final measurements instead of removing them before PBC conversion.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="Optional path for conversion metadata used by batch runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qasm_file = args.qasm_file.resolve()
    try:
        lines, metadata = convert_qasm(qasm_file, args.keep_final_measurements)
    except Exception as exc:
        print(f"{qasm_file}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for line in lines:
        print(line)

    if args.metadata_json:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

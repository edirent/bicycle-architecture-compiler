"""Matrix verifier for single-qubit examples."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from lgm_compiler.compiler import CompileResult
from lgm_compiler.frame import CliffordFrame
from lgm_compiler.input_ir import Circuit, Gate
from lgm_compiler.lgm_ir import FrameUpdate, LGMOp, Measure1, Rot
from lgm_compiler.pauli import Axis

Matrix = NDArray[np.complex128]


def input_unitary(circuit: Circuit) -> Matrix:
    _require_single_qubit_circuit(circuit)

    unitary = np.eye(2, dtype=np.complex128)
    for gate in circuit.gates:
        unitary = gate_matrix(gate) @ unitary
    return unitary


def lgm_unitary(result: CompileResult, q: int = 0) -> Matrix:
    tail = np.eye(2, dtype=np.complex128)
    for op in result.ir_ops:
        if isinstance(op, Rot):
            if op.q != q:
                raise ValueError("single-qubit verifier can only handle one qubit")
            tail = rot_matrix(op) @ tail
        elif isinstance(op, (FrameUpdate, Measure1)):
            continue
        else:
            raise ValueError(f"single-qubit verifier cannot handle {op!r}")

    return frame_matrix(result.final_frames[q]) @ tail


def verification_error(circuit: Circuit, result: CompileResult, q: int = 0) -> float:
    return global_phase_distance(input_unitary(circuit), lgm_unitary(result, q=q))


def verify_single_qubit(circuit: Circuit, result: CompileResult, *, atol: float = 1e-10) -> bool:
    return verification_error(circuit, result) < atol


def global_phase_distance(left: Matrix, right: Matrix) -> float:
    overlap = np.vdot(right, left)
    if abs(overlap) == 0:
        phase = 1.0 + 0.0j
    else:
        phase = overlap / abs(overlap)
    return float(np.linalg.norm(left - phase * right))


def gate_matrix(gate: Gate) -> Matrix:
    name = gate.name
    if name == "H":
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    if name == "S":
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    if name == "Sdg":
        return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    if name == "T":
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    if name == "Tdg":
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)
    if name == "X":
        return pauli_matrix("X")
    if name == "Y":
        return pauli_matrix("Y")
    if name == "Z":
        return pauli_matrix("Z")
    if name == "RX":
        theta = _theta(gate)
        return _axis_rotation("X", theta)
    if name == "RY":
        theta = _theta(gate)
        return _axis_rotation("Y", theta)
    if name == "RZ":
        theta = _theta(gate)
        return _axis_rotation("Z", theta)
    raise ValueError(f"single-qubit verifier cannot handle {name}")


def rot_matrix(op: Rot) -> Matrix:
    pauli = pauli_matrix(op.axis)
    ident = np.eye(2, dtype=np.complex128)
    if op.a is not None:
        angle = op.a * np.pi / 8
        return np.cos(angle) * ident + 1j * np.sin(angle) * pauli
    if op.theta is not None:
        return _axis_rotation(op.axis, op.theta)
    raise ValueError("rotation has neither a nor theta")


def pauli_matrix(axis: Axis) -> Matrix:
    if axis == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if axis == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if axis == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(f"unknown Pauli axis {axis!r}")


def frame_matrix(frame: CliffordFrame) -> Matrix:
    table = _frame_unitary_table()
    try:
        return table[frame]
    except KeyError as exc:
        raise ValueError(f"frame is not a single-qubit Clifford: {frame}") from exc


def _axis_rotation(axis: Axis, theta: float) -> Matrix:
    ident = np.eye(2, dtype=np.complex128)
    return np.cos(theta / 2) * ident - 1j * np.sin(theta / 2) * pauli_matrix(axis)


def _theta(gate: Gate) -> float:
    if gate.theta is None:
        raise ValueError(f"{gate.name} requires theta")
    return gate.theta


def _require_single_qubit_circuit(circuit: Circuit) -> None:
    if circuit.n_qubits != 1:
        raise ValueError("single-qubit verifier requires a one-qubit circuit")
    for gate in circuit.gates:
        if len(gate.qubits) != 1 or gate.qubits[0] != 0:
            raise ValueError("single-qubit verifier only supports gates on q=0")
        if gate.name == "MEASURE_Z":
            raise ValueError("single-qubit verifier does not simulate measurements")


@lru_cache(maxsize=1)
def _frame_unitary_table() -> dict[CliffordFrame, Matrix]:
    generators = ("H", "S", "Sdg", "X", "Y", "Z")
    start = CliffordFrame.identity()
    table: dict[CliffordFrame, Matrix] = {start: np.eye(2, dtype=np.complex128)}
    frontier: list[CliffordFrame] = [start]

    while frontier:
        frame = frontier.pop()
        unitary = table[frame]
        for name in generators:
            next_frame = frame.updated(name)
            if next_frame in table:
                continue
            next_unitary = gate_matrix(Gate(name=name, qubits=(0,))) @ unitary
            table[next_frame] = next_unitary
            frontier.append(next_frame)

    return table

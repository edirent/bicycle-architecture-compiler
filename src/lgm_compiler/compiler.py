"""Compiler pass from input circuits to local gate-measurement IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from lgm_compiler.frame import CliffordFrame
from lgm_compiler.input_ir import Circuit, Gate
from lgm_compiler.lgm_ir import (
    FallbackPBC,
    FrameUpdate,
    FramedCNOT,
    FramedToffoli,
    LGMOp,
    Measure1,
    Measure2,
    Rot,
)
from lgm_compiler.pauli import MeasureAxis, SignedPauli

CLIFFORD_GATES = frozenset({"H", "S", "Sdg", "X", "Y", "Z"})


@dataclass(frozen=True)
class CompileResult:
    ir_ops: list[LGMOp]
    final_frames: dict[int, CliffordFrame]

    @property
    def rotations(self) -> list[Rot]:
        return [op for op in self.ir_ops if isinstance(op, Rot)]


def compile_circuit(circuit: Circuit, *, emit_final_frames: bool = False) -> CompileResult:
    """Compile a circuit into LGM-IR while keeping only local Clifford frames."""

    frames = {q: CliffordFrame.identity() for q in range(circuit.n_qubits)}
    ir_ops: list[LGMOp] = []

    for gate in circuit.gates:
        name = gate.name
        if name in CLIFFORD_GATES:
            q = gate.q
            frames[q] = frames[q].updated(name)
        elif name == "T":
            ir_ops.append(_rotation_from_frame(gate.q, frames[gate.q].image("Z"), a=-1))
        elif name == "Tdg":
            ir_ops.append(_rotation_from_frame(gate.q, frames[gate.q].image("Z"), a=1))
        elif name == "RX":
            ir_ops.append(_rotation_from_frame(gate.q, frames[gate.q].image("X"), theta=_theta(gate)))
        elif name == "RY":
            ir_ops.append(_rotation_from_frame(gate.q, frames[gate.q].image("Y"), theta=_theta(gate)))
        elif name == "RZ":
            ir_ops.append(_rotation_from_frame(gate.q, frames[gate.q].image("Z"), theta=_theta(gate)))
        elif name == "CNOT":
            c, t = gate.qubits
            ir_ops.append(FramedCNOT(c=c, t=t, frame_c=frames[c], frame_t=frames[t]))
            frames[c] = CliffordFrame.identity()
            frames[t] = CliffordFrame.identity()
        elif name == "TOFFOLI":
            c1, c2, t = gate.qubits
            ir_ops.append(
                FramedToffoli(
                    c1=c1,
                    c2=c2,
                    t=t,
                    frame_c1=frames[c1],
                    frame_c2=frames[c2],
                    frame_t=frames[t],
                )
            )
            frames[c1] = CliffordFrame.identity()
            frames[c2] = CliffordFrame.identity()
            frames[t] = CliffordFrame.identity()
        elif name == "MEASURE_Z":
            signed_axis = frames[gate.q].image("Z")
            ir_ops.append(Measure1(q=gate.q, axis=signed_axis.axis, sign=signed_axis.sign))
            frames[gate.q] = CliffordFrame.identity()
        else:
            raise ValueError(f"unsupported input gate {name!r}")

    final_frames = dict(frames)
    if emit_final_frames:
        for q, frame in final_frames.items():
            if not frame.is_identity():
                ir_ops.append(FrameUpdate(q=q, frame=frame))

    return CompileResult(ir_ops=ir_ops, final_frames=final_frames)


def _theta(gate: Gate) -> float:
    if gate.theta is None:
        raise ValueError(f"{gate.name} requires theta")
    return gate.theta


def _rotation_from_frame(
    q: int,
    signed_axis: SignedPauli,
    *,
    a: int | None = None,
    theta: float | None = None,
) -> Rot:
    if a is not None:
        return Rot(q=q, axis=signed_axis.axis, a=a * signed_axis.sign)
    if theta is not None:
        return Rot(q=q, axis=signed_axis.axis, theta=theta * signed_axis.sign)
    raise ValueError("expected a T-like integer parameter or an arbitrary angle")


def lower_framed_cnot_to_measurements(
    cnot_op: FramedCNOT,
    *,
    ancilla: int = -1,
) -> list[LGMOp]:
    """Lower a framed CNOT to symbolic X/Z parity measurements when possible."""

    control_z = cnot_op.frame_c.image("Z")
    target_x = cnot_op.frame_t.image("X")
    if control_z.axis == "Y" or target_x.axis == "Y":
        return [
            FallbackPBC(
                reason="Y-axis two-qubit measurement not supported in X/Z MVP",
                op=cnot_op,
            )
        ]

    measure_axis_c: MeasureAxis = control_z.axis  # type: ignore[assignment]
    measure_axis_t: MeasureAxis = target_x.axis  # type: ignore[assignment]
    return [
        Measure2(cnot_op.c, measure_axis_c, ancilla, "Z", sign1=control_z.sign),
        Measure2(ancilla, "X", cnot_op.t, measure_axis_t, sign2=target_x.sign),
        FrameUpdate(cnot_op.c, "symbolic CNOT feedforward frame update"),
        FrameUpdate(cnot_op.t, "symbolic CNOT feedforward frame update"),
    ]


def compile_gates(n_qubits: int, gates: Iterable[Gate], *, emit_final_frames: bool = False) -> CompileResult:
    return compile_circuit(Circuit(n_qubits=n_qubits, gates=list(gates)), emit_final_frames=emit_final_frames)

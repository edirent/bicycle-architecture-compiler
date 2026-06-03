from lgm_compiler.compiler import compile_circuit, lower_framed_cnot_to_measurements
from lgm_compiler.frame import CliffordFrame
from lgm_compiler.input_ir import Circuit, gate, parse_single_qubit_sequence
from lgm_compiler.lgm_ir import FallbackPBC, FrameUpdate, FramedCNOT, Measure2, Rot
from lgm_compiler.verifier import verify_single_qubit


def test_h_t_h_emits_one_x_axis_t_like_rotation_and_verifies() -> None:
    circuit = Circuit(n_qubits=1, gates=[gate("H", 0), gate("T", 0), gate("H", 0)])
    result = compile_circuit(circuit)

    assert result.rotations == [Rot(q=0, axis="X", a=-1)]
    assert verify_single_qubit(circuit, result)


def test_t_s_verifies_against_direct_matrix() -> None:
    circuit = Circuit(n_qubits=1, gates=[gate("T", 0), gate("S", 0)])
    result = compile_circuit(circuit)

    assert verify_single_qubit(circuit, result)


def test_screenshot_example_verifies_up_to_global_phase() -> None:
    gates = parse_single_qubit_sequence("H T H T H T S H T H T S H T", q=0, operator_order=True)
    circuit = Circuit(n_qubits=1, gates=gates)
    result = compile_circuit(circuit)

    assert len(result.rotations) == 6
    assert [rot.axis for rot in result.rotations] == ["Z", "X", "Y", "X", "Z", "X"]
    assert verify_single_qubit(circuit, result)


def test_cnot_consumes_local_frames_without_spreading_t_to_high_weight_pauli() -> None:
    circuit = Circuit(
        n_qubits=2,
        gates=[gate("H", 0), gate("CNOT", 0, 1), gate("T", 1)],
    )
    result = compile_circuit(circuit)

    assert isinstance(result.ir_ops[0], FramedCNOT)
    assert result.ir_ops[0].frame_c.image("Z").axis == "X"
    assert result.ir_ops[1] == Rot(q=1, axis="Z", a=-1)
    assert all(isinstance(op, (FramedCNOT, Rot)) for op in result.ir_ops)


def test_framed_cnot_lowering_uses_xz_measurements_or_fallback() -> None:
    identity = CliffordFrame.identity()
    lowered = lower_framed_cnot_to_measurements(FramedCNOT(0, 1, identity, identity), ancilla=2)

    assert lowered == [
        Measure2(0, "Z", 2, "Z"),
        Measure2(2, "X", 1, "X"),
        FrameUpdate(0, "symbolic CNOT feedforward frame update"),
        FrameUpdate(1, "symbolic CNOT feedforward frame update"),
    ]

    s_frame = CliffordFrame.identity().updated("S")
    fallback = lower_framed_cnot_to_measurements(FramedCNOT(0, 1, identity, s_frame))

    assert len(fallback) == 1
    assert isinstance(fallback[0], FallbackPBC)
    assert "Y-axis" in fallback[0].reason

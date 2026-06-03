from lgm_compiler.accounting import account_cliffords
from lgm_compiler.compiler import CompileResult, compile_circuit
from lgm_compiler.frame import CliffordFrame
from lgm_compiler.input_ir import Circuit, gate, parse_single_qubit_sequence
from lgm_compiler.lgm_ir import FallbackPBC, FramedToffoli


def test_screenshot_clifford_accounting() -> None:
    gates = parse_single_qubit_sequence("H T H T H T S H T H T S H T", q=0, operator_order=True)
    circuit = Circuit(n_qubits=1, gates=gates)
    result = compile_circuit(circuit)
    report = account_cliffords(circuit, result)

    assert report.input_h == 6
    assert report.input_s == 2
    assert report.input_1q_clifford == 8
    assert report.explicit_1q_clifford_after_lgm == 0
    assert report.final_nonidentity_frames == 1
    assert report.physical_1q_clifford_remaining == 0
    assert report.framed_cnot_macros == 0


def test_cnot_is_reported_as_entangling_clifford_macro() -> None:
    circuit = Circuit(n_qubits=2, gates=[gate("H", 0), gate("CNOT", 0, 1), gate("T", 1)])
    result = compile_circuit(circuit)
    report = account_cliffords(circuit, result)

    assert report.input_h == 1
    assert report.input_cnot == 1
    assert report.input_1q_clifford == 1
    assert report.input_entangling_clifford_macros == 1
    assert report.framed_cnot_macros == 1
    assert report.physical_1q_clifford_remaining == 0


def test_final_frames_frame_updates_fallback_and_toffoli_are_separate() -> None:
    circuit = Circuit(n_qubits=1, gates=[gate("H", 0)])
    result = compile_circuit(circuit, emit_final_frames=True)
    report = account_cliffords(circuit, result)

    assert report.final_nonidentity_frames == 1
    assert report.frame_updates == 1
    assert report.physical_1q_clifford_remaining == 0

    macro_result = CompileResult(
        ir_ops=[
            FramedToffoli(
                0,
                1,
                2,
                CliffordFrame.identity(),
                CliffordFrame.identity(),
                CliffordFrame.identity(),
            ),
            FallbackPBC("fallback", object()),
        ],
        final_frames={0: CliffordFrame.identity(), 1: CliffordFrame.identity(), 2: CliffordFrame.identity()},
    )
    macro_report = account_cliffords(Circuit(n_qubits=3), macro_result)

    assert macro_report.framed_toffoli_macros == 1
    assert macro_report.fallback_pbc == 1
    assert macro_report.input_1q_clifford == 0

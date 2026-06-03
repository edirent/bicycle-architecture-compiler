import numpy as np

from lgm_compiler.compiler import compile_circuit
from lgm_compiler.input_ir import Circuit, gate
from lgm_compiler.verifier import verification_error, verify_single_qubit


def test_verifier_accepts_arbitrary_single_qubit_rotations_under_frame() -> None:
    circuit = Circuit(
        n_qubits=1,
        gates=[gate("H", 0), gate("RZ", 0, theta=0.25), gate("S", 0), gate("RX", 0, theta=-0.5)],
    )
    result = compile_circuit(circuit)

    assert verify_single_qubit(circuit, result)
    assert verification_error(circuit, result) < 1e-10


def test_global_phase_distance_handles_t_gate_convention() -> None:
    circuit = Circuit(n_qubits=1, gates=[gate("T", 0)])
    result = compile_circuit(circuit)

    assert np.isclose(verification_error(circuit, result), 0.0, atol=1e-10)

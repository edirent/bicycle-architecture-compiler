from __future__ import annotations

import unittest

from prototype.two_table_compiler.circuit_correctness import (
    CIRCUIT_A_NAME,
    CIRCUIT_B_NAME,
    CIRCUIT_C_NAME,
    CIRCUIT_D_NAME,
    CIRCUIT_E_NAME,
    CIRCUIT_F_NAME,
    run_correctness_check,
)


class CircuitCompileCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_correctness_check(write_failure_artifact=True)

    def _assert_circuit_passed(self, circuit_name: str) -> None:
        self.assertIn(circuit_name, self.report.circuit_results)
        circuit = self.report.circuit_results[circuit_name]
        if circuit.passed:
            return

        lines = [f"circuit {circuit_name} failed with {len(circuit.failures)} failures"]
        for failure in circuit.failures[:10]:
            lines.append(
                f"- idx={failure.target_index} target={failure.target_tail} "
                f"status={failure.search_status} note={failure.note}"
            )
        if self.report.failure_artifact_path is not None:
            lines.append(f"- failure_artifact={self.report.failure_artifact_path}")
        self.fail("\n".join(lines))

    def test_circuit_a_all_direct_local_control(self) -> None:
        self._assert_circuit_passed(CIRCUIT_A_NAME)

    def test_circuit_b_cross_logical_native_circuit(self) -> None:
        self._assert_circuit_passed(CIRCUIT_B_NAME)

    def test_circuit_c_joint_required_circuit(self) -> None:
        self._assert_circuit_passed(CIRCUIT_C_NAME)

    def test_circuit_d_mixed_p0_p2_p3_circuit(self) -> None:
        self._assert_circuit_passed(CIRCUIT_D_NAME)

    def test_circuit_e_exhaustive_weight1_suite(self) -> None:
        self._assert_circuit_passed(CIRCUIT_E_NAME)
        self.assertTrue(self.report.weight1_all_passed)

    def test_circuit_f_exhaustive_weight2_suite(self) -> None:
        self._assert_circuit_passed(CIRCUIT_F_NAME)
        self.assertTrue(self.report.weight2_all_passed)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import unittest

from prototype.two_table_compiler.io import (
    DEFAULT_JOINT_TABLE_PATH,
    DEFAULT_LOCAL_TABLE_PATH,
    load_joint_table,
    load_local_table,
)
from prototype.two_table_compiler.search import SearchStatus, audit_target_execution
from prototype.two_table_compiler.state import N_QUBITS


def _weight1_targets() -> list[str]:
    out: list[str] = []
    for idx in range(N_QUBITS):
        for pauli in ("X", "Y", "Z"):
            symbols = ["I"] * N_QUBITS
            symbols[idx] = pauli
            out.append("".join(symbols))
    return out


class Weight1CanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.local_table = load_local_table(DEFAULT_LOCAL_TABLE_PATH)
        cls.joint_table = load_joint_table(DEFAULT_JOINT_TABLE_PATH)
        cls.targets = _weight1_targets()

    def test_weight1_canary_budgeted_never_mislabels_rule_unreachable(self) -> None:
        # Fast always-on canary:
        # with a finite budget we may truncate, but we should not call those RULE_UNREACHABLE.
        for target in self.targets:
            report = audit_target_execution(
                target=target,
                local_table=self.local_table,
                joint_table=self.joint_table,
                max_popped_states=200,
                enable_p2=True,
                p2_use_cache=True,
                target_aware_p2_fallback=False,
            )
            self.assertNotEqual(
                report.search_status,
                SearchStatus.RULE_UNREACHABLE,
                msg=(
                    f"unexpected RULE_UNREACHABLE for {target}: "
                    f"in_p0={report.in_p0_source}, in_p2={report.in_p2_source}, "
                    f"same_tail_in_dist={report.any_state_with_same_tail_in_dist}, "
                    f"projected_best={report.projected_best_cost_exists}, popped={report.popped_states}, "
                    f"y_debug={report.y_debug}"
                ),
            )

    @unittest.skipUnless(
        os.getenv("RUN_FULL_WEIGHT1_CANARY") == "1",
        "set RUN_FULL_WEIGHT1_CANARY=1 to run the no-budget full-search canary",
    )
    def test_weight1_canary_full_search_all_reachable(self) -> None:
        # Strict diagnostic canary requested by audit:
        # no budget + no target-aware P2 + no memoization.
        failures: list[str] = []
        for target in self.targets:
            report = audit_target_execution(
                target=target,
                local_table=self.local_table,
                joint_table=self.joint_table,
                max_popped_states=None,
                enable_p2=True,
                p2_use_cache=False,
                target_aware_p2_fallback=False,
            )
            if report.search_status != SearchStatus.FOUND_OPTIMAL:
                failures.append(
                    f"{target}: status={report.search_status.value} "
                    f"in_p0={report.in_p0_source} in_p2={report.in_p2_source} "
                    f"same_tail_in_dist={report.any_state_with_same_tail_in_dist} "
                    f"projected_best={report.projected_best_cost_exists} "
                    f"popped={report.popped_states} y_debug={report.y_debug}"
                )
        self.assertEqual([], failures, msg="\n".join(failures))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import json
import unittest
from itertools import combinations
from pathlib import Path

from prototype.two_table_compiler.io import (
    DEFAULT_JOINT_TABLE_PATH,
    DEFAULT_LOCAL_TABLE_PATH,
    load_joint_table,
    load_local_table,
)
from prototype.two_table_compiler.search import SearchStatus, TargetAuditReport, audit_target_execution
from prototype.two_table_compiler.state import N_QUBITS

DEBUG_DIR = Path(__file__).resolve().parent / "debug"
FAILURE_CSV = DEBUG_DIR / "weight2_canary_failures.csv"
FAILURE_FIELDS = [
    "target_tail",
    "search_status",
    "p0_source_hit_count",
    "p2_source_hit_count",
    "projected_best_cost",
    "any_state_with_same_tail_seen",
    "any_accepting_state_with_same_tail_seen",
    "first_family_if_any",
    "note",
]


def _weight2_targets() -> list[str]:
    out: list[str] = []
    paulis = ("X", "Y", "Z")
    for idx in range(N_QUBITS):
        for pauli in paulis:
            symbols = ["I"] * N_QUBITS
            symbols[idx] = pauli
            out.append("".join(symbols))
    for lhs, rhs in combinations(range(N_QUBITS), 2):
        for pauli_lhs in paulis:
            for pauli_rhs in paulis:
                symbols = ["I"] * N_QUBITS
                symbols[lhs] = pauli_lhs
                symbols[rhs] = pauli_rhs
                out.append("".join(symbols))
    return out


def _weight1_y_targets() -> list[str]:
    out: list[str] = []
    for idx in range(N_QUBITS):
        symbols = ["I"] * N_QUBITS
        symbols[idx] = "Y"
        out.append("".join(symbols))
    return out


def _write_failure_csv(rows: list[dict[str, object]]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with FAILURE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report_to_failure_row(report: TargetAuditReport) -> dict[str, object]:
    return {
        "target_tail": report.target_tail.compact(),
        "search_status": report.search_status.value,
        "p0_source_hit_count": report.p0_source_hit_count,
        "p2_source_hit_count": report.p2_source_hit_count,
        "projected_best_cost": report.projected_best_cost,
        "any_state_with_same_tail_seen": report.any_state_with_same_tail_in_dist,
        "any_accepting_state_with_same_tail_seen": report.any_accepting_state_with_same_tail_in_dist,
        "first_family_if_any": report.first_family_if_any or "",
        "note": report.reason or "",
    }


class Weight2CanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.local_table = load_local_table(DEFAULT_LOCAL_TABLE_PATH)
        cls.joint_table = load_joint_table(DEFAULT_JOINT_TABLE_PATH)
        cls.targets = _weight2_targets()
        cls.y_targets = _weight1_y_targets()
        cls._report_cache: dict[str, TargetAuditReport] = {}

    @classmethod
    def _audit(cls, target: str) -> TargetAuditReport:
        report = cls._report_cache.get(target)
        if report is None:
            report = audit_target_execution(
                target=target,
                local_table=cls.local_table,
                joint_table=cls.joint_table,
                max_popped_states=None,
                enable_p2=True,
                p2_use_cache=True,
                target_aware_p2_fallback=False,
            )
            cls._report_cache[target] = report
        return report

    def test_weight2_canary_all_targets_found_optimal_under_no_budget(self) -> None:
        failures: list[dict[str, object]] = []
        for target in self.targets:
            report = self._audit(target)
            if report.search_status != SearchStatus.FOUND_OPTIMAL:
                failures.append(_report_to_failure_row(report))

        _write_failure_csv(failures)
        self.assertEqual(
            [],
            failures,
            msg=(
                f"weight<=2 no-budget canary found {len(failures)} non-optimal targets; "
                f"see {FAILURE_CSV}. "
                f"first_failures={json.dumps(failures[:5], sort_keys=True)}"
            ),
        )

    def test_all_single_qubit_y_targets_are_found_optimal_under_no_budget(self) -> None:
        failures: list[str] = []
        for target in self.y_targets:
            report = self._audit(target)
            if report.search_status == SearchStatus.FOUND_OPTIMAL:
                continue
            failures.append(
                f"{target}: status={report.search_status.value} "
                f"p0_hits={report.p0_source_hit_count} p2_hits={report.p2_source_hit_count} "
                f"projected_best_cost={report.projected_best_cost} "
                f"same_tail_seen={report.any_state_with_same_tail_in_dist} "
                f"accepting_same_tail_seen={report.any_accepting_state_with_same_tail_in_dist} "
                f"first_family={report.first_family_if_any} "
                f"y_debug={json.dumps(report.y_debug, sort_keys=True)}"
            )

        self.assertEqual([], failures, msg="\n".join(failures))


if __name__ == "__main__":
    unittest.main()

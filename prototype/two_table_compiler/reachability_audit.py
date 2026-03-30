from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .io import DEFAULT_JOINT_TABLE_PATH, DEFAULT_LOCAL_TABLE_PATH, load_joint_table, load_local_table
from .search import SearchStatus, TargetAuditReport, audit_target_execution, search_status_flags
from .state import N_QUBITS

RESULTS_ROOT = Path("benchmark") / "results"


def _latest_results_dir(root: Path = RESULTS_ROOT) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"results root does not exist: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no benchmark result directory found under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes")


def _load_unreachable_targets(per_target_path: Path) -> List[str]:
    with per_target_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    targets: List[str] = []
    for row in rows:
        status = str(row.get("search_status_ours_full", row.get("ours_full_search_status", ""))).strip()
        if status:
            if not search_status_flags(status)["reachable"]:
                targets.append(row["target"])
            continue
        if not _parse_bool(row["ours_full_reachable"]):
            targets.append(row["target"])
    return list(dict.fromkeys(targets))


def _is_reachable_status(status: SearchStatus) -> bool:
    return status in (SearchStatus.FOUND_OPTIMAL, SearchStatus.FOUND_REACHABLE_UPPER_BOUND)


def _report_to_row(report: TargetAuditReport) -> Dict[str, object]:
    return {
        "target_tail": report.target_tail.compact(),
        "search_status": report.search_status.value,
        "in_p0_source": report.in_p0_source,
        "in_p2_source": report.in_p2_source,
        "any_state_with_same_tail_in_dist": report.any_state_with_same_tail_in_dist,
        "projected_best_cost_exists": report.projected_best_cost_exists,
        "popped_states": report.popped_states,
        "plan_cost": report.plan_cost,
        "reason": report.reason or "",
        "y_debug": json.dumps(report.y_debug, sort_keys=True) if report.y_debug is not None else "",
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_tail",
        "search_status",
        "in_p0_source",
        "in_p2_source",
        "any_state_with_same_tail_in_dist",
        "projected_best_cost_exists",
        "popped_states",
        "plan_cost",
        "reason",
        "y_debug",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _weight1_targets() -> Iterable[str]:
    for idx in range(N_QUBITS):
        for pauli in ("X", "Y", "Z"):
            symbols = ["I"] * N_QUBITS
            symbols[idx] = pauli
            yield "".join(symbols)


def run_audit(
    *,
    benchmark_dir: Path,
    local_table_path: Path,
    joint_table_path: Path,
) -> Tuple[Path, Path]:
    local_table = load_local_table(local_table_path)
    joint_table = load_joint_table(joint_table_path)

    per_target_path = benchmark_dir / "per_target.csv"
    if not per_target_path.exists():
        raise FileNotFoundError(f"missing per_target.csv in {benchmark_dir}")

    unreachable_targets = _load_unreachable_targets(per_target_path)
    unreachable_reports: List[TargetAuditReport] = []
    for target in unreachable_targets:
        unreachable_reports.append(
            audit_target_execution(
                target=target,
                local_table=local_table,
                joint_table=joint_table,
                max_popped_states=None,
                enable_p2=True,
                p2_use_cache=False,
                target_aware_p2_fallback=False,
            )
        )

    canary_reports: List[TargetAuditReport] = []
    for target in _weight1_targets():
        canary_reports.append(
            audit_target_execution(
                target=target,
                local_table=local_table,
                joint_table=joint_table,
                max_popped_states=None,
                enable_p2=True,
                p2_use_cache=False,
                target_aware_p2_fallback=False,
            )
        )

    unreachable_csv = benchmark_dir / "audit_unreachable.csv"
    canary_csv = benchmark_dir / "audit_weight1_canary.csv"
    _write_csv(unreachable_csv, [_report_to_row(report) for report in unreachable_reports])
    _write_csv(canary_csv, [_report_to_row(report) for report in canary_reports])

    canary_failures = [report for report in canary_reports if not _is_reachable_status(report.search_status)]
    status_counts: Dict[str, int] = {}
    for report in canary_reports:
        key = report.search_status.value
        status_counts[key] = status_counts.get(key, 0) + 1

    print(f"audited unreachable targets: {len(unreachable_reports)}")
    print(f"wrote: {unreachable_csv}")
    print(f"wrote: {canary_csv}")
    print(f"weight1 canary status counts: {status_counts}")
    if canary_failures:
        print("weight1 canary failures:")
        for report in canary_failures:
            print(
                f"  target={report.target_tail.compact()} status={report.search_status.value} "
                f"in_p0={report.in_p0_source} in_p2={report.in_p2_source} "
                f"same_tail_in_dist={report.any_state_with_same_tail_in_dist} "
                f"projected_best={report.projected_best_cost_exists} "
                f"popped={report.popped_states}"
            )
            if report.y_debug is not None:
                print(f"    y_debug={json.dumps(report.y_debug, sort_keys=True)}")
    return unreachable_csv, canary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execution audit for reachability bugs")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="benchmark result directory that contains per_target.csv (default: latest under benchmark/results)",
    )
    parser.add_argument("--local-table-path", type=Path, default=DEFAULT_LOCAL_TABLE_PATH)
    parser.add_argument("--joint-table-path", type=Path, default=DEFAULT_JOINT_TABLE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir or _latest_results_dir()
    run_audit(
        benchmark_dir=benchmark_dir,
        local_table_path=args.local_table_path,
        joint_table_path=args.joint_table_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

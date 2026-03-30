from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import DEFAULT_JOINT_TABLE_PATH, DEFAULT_LOCAL_TABLE_PATH, load_joint_table, load_local_table
from .search import (
    SearchStatus,
    assert_no_budget_status_invariant,
    search_status_note,
    synthesize_search,
)
from .state import Tail


def _parse_bool(raw: Any) -> bool | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text == "":
        return None
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no"):
        return False
    return None


def _row_target(row: Dict[str, Any]) -> str:
    target = str(row.get("target_tail", row.get("target", ""))).strip()
    if not target:
        raise ValueError(f"input row is missing target/target_tail: {row}")
    return target


def _row_old_label(row: Dict[str, Any], index: int) -> str:
    for key in ("story_label", "label", "strata", "stratum", "old_label"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return f"row_{index}"


def _row_old_status(row: Dict[str, Any]) -> str:
    for key in (
        "ours_full_search_status",
        "search_status_ours_full",
        "search_status",
        "old_status",
    ):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    for key in ("ours_full_reachable", "reachable_ours_full", "reachable", "old_reachable"):
        value = _parse_bool(row.get(key))
        if value is not None:
            return f"reachable={value}"
    return ""


def _old_row_is_unreachable_like(row: Dict[str, Any]) -> bool:
    old_status = _row_old_status(row)
    if old_status in (
        SearchStatus.RULE_UNREACHABLE.value,
        SearchStatus.SEARCH_TRUNCATED.value,
    ):
        return True
    if old_status == "reachable=False":
        return True
    strata = str(row.get("strata", row.get("stratum", ""))).strip()
    label = str(row.get("story_label", row.get("label", ""))).strip()
    if strata == "frame_unreachable" or label == "frame_unreachable_case":
        return True
    reachable = _parse_bool(row.get("ours_full_reachable", row.get("reachable_ours_full", row.get("reachable"))))
    return reachable is False


def _load_input_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []
        if "story_label" in rows[0]:
            return rows
        return [row for row in rows if _old_row_is_unreachable_like(row)]

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle, start=1):
            target = raw.strip()
            if not target:
                continue
            rows.append({"old_label": f"line_{index}", "target_tail": target})
    return rows


def _canonical_old_semantics(old_status_or_old_reachable: str) -> str:
    if old_status_or_old_reachable in (
        SearchStatus.FOUND_OPTIMAL.value,
        SearchStatus.FOUND_REACHABLE_UPPER_BOUND.value,
        SearchStatus.SEARCH_TRUNCATED.value,
        SearchStatus.RULE_UNREACHABLE.value,
    ):
        return old_status_or_old_reachable
    if old_status_or_old_reachable == "reachable=True":
        return SearchStatus.FOUND_OPTIMAL.value
    if old_status_or_old_reachable == "reachable=False":
        return SearchStatus.RULE_UNREACHABLE.value
    return "UNKNOWN"


def _changed_semantics(old_status_or_old_reachable: str, new_search_status: str) -> bool:
    old_semantics = _canonical_old_semantics(old_status_or_old_reachable)
    if old_semantics == "UNKNOWN":
        return False
    return old_semantics != new_search_status


def _audit_note(old_status_or_old_reachable: str, new_status: SearchStatus) -> str:
    old_semantics = _canonical_old_semantics(old_status_or_old_reachable)
    if old_semantics == new_status.value:
        return search_status_note(new_status)
    if old_semantics == SearchStatus.RULE_UNREACHABLE.value:
        if new_status == SearchStatus.SEARCH_TRUNCATED:
            return "old unreachable label was a budget truncation, not a true rule-unreachable target"
        if new_status == SearchStatus.FOUND_REACHABLE_UPPER_BOUND:
            return "old unreachable label hid an upper-bound witness under finite-budget search"
        if new_status == SearchStatus.FOUND_OPTIMAL:
            return (
                "old unreachable label became reachable optimal under no-budget replay; "
                "likely legacy plan-null flattening or old target-aware P2 false negative"
            )
    return search_status_note(new_status)


def run_audit(
    *,
    input_path: Path,
    output_path: Path | None,
    local_table_path: Path,
    joint_table_path: Path,
) -> Path:
    local_table = load_local_table(local_table_path)
    joint_table = load_joint_table(joint_table_path)
    rows = _load_input_rows(input_path)

    output_rows: List[Dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        target = _row_target(row)
        result = synthesize_search(
            Tail.from_str(target),
            local_table,
            joint_table,
            max_popped_states=None,
            enable_p2=True,
            p2_use_cache=False,
        )
        new_status = result.status
        assert_no_budget_status_invariant(
            status=new_status,
            max_popped_states=None,
            context=f"audit_old_unreachable target={target}",
        )
        old_label = _row_old_label(row, index)
        old_status = _row_old_status(row)
        output_rows.append(
            {
                "old_label": old_label,
                "target_tail": target,
                "old_status_or_old_reachable": old_status,
                "new_search_status": new_status.value,
                "changed_semantics": _changed_semantics(old_status, new_status.value),
                "note": _audit_note(old_status, new_status),
            }
        )

    out = output_path or input_path.with_name(f"{input_path.stem}_audit_old_unreachable.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "old_label",
        "target_tail",
        "old_status_or_old_reachable",
        "new_search_status",
        "changed_semantics",
        "note",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay old unreachable labels under no-budget four-state search")
    parser.add_argument("--input", type=Path, required=True, help="old per_target.csv or old story target list")
    parser.add_argument("--output", type=Path, default=None, help="optional audit CSV output path")
    parser.add_argument("--local-table-path", type=Path, default=DEFAULT_LOCAL_TABLE_PATH)
    parser.add_argument("--joint-table-path", type=Path, default=DEFAULT_JOINT_TABLE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = run_audit(
        input_path=args.input,
        output_path=args.output,
        local_table_path=args.local_table_path,
        joint_table_path=args.joint_table_path,
    )
    print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

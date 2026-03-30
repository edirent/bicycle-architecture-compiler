from __future__ import annotations

import argparse
from pathlib import Path

from .build_assets import refresh_joint_table_asset, refresh_local_table_asset
from .examples import DEMO_JOINT_TABLE, DEMO_LOCAL_TABLE
from .io import (
    DEFAULT_JOINT_TABLE_PATH,
    DEFAULT_LOCAL_TABLE_PATH,
    load_joint_table,
    load_local_table,
)
from .rules import generate_p0_sources, generate_p2_sources
from .search import (
    SearchStatus,
    SynthesisSearchResult,
    assert_no_budget_status_invariant,
    search_status_note,
    synthesize_search,
)
from .state import DEFAULT_ACCEPTING_HIDDEN_CLASSES, State
from .tables import JointTable, LocalTable
from .targeting import classify_target


def _format_state(state: State) -> str:
    return f"State(hidden={state.hidden.name}, tail={state.tail.compact()})"


def _print_local_summary(local_table: LocalTable) -> None:
    grouped = local_table.by_head()
    print("local table summary:")
    print(f"  frame_id: {local_table.frame_id}")
    print(f"  native_entry_count: {len(local_table.entries)}")
    for head in ("I", "X", "Y", "Z"):
        print(f"  head[{head}] count: {len(grouped.get(head, []))}")


def _print_joint_summary(joint_table: JointTable) -> None:
    print("joint/protocol table summary:")
    print(f"  p2_rule_count: {len(joint_table.p2_rules)}")
    print(f"  p3_rule_count: {len(joint_table.p3_rules)}")
    accepting = ", ".join(hidden.name for hidden in DEFAULT_ACCEPTING_HIDDEN_CLASSES)
    print(f"  accepting_hidden_classes: {accepting}")


def _format_metadata(metadata: tuple[tuple[str, str], ...]) -> str:
    if not metadata:
        return "{}"
    return "{" + ", ".join(f"{k}={v}" for k, v in metadata) + "}"


def _step_kind_label(family: str) -> str:
    if family == "P0":
        return "direct_native"
    if family == "P2":
        return "composed_joint_source"
    if family == "P3":
        return "state_transition"
    return "unknown"


def _resolve_tables(
    *,
    use_demo_table: bool,
    auto_build_assets: bool,
    local_table_path: Path,
    joint_table_path: Path,
) -> tuple[LocalTable, JointTable]:
    if use_demo_table:
        return DEMO_LOCAL_TABLE, DEMO_JOINT_TABLE

    if auto_build_assets:
        if not local_table_path.exists():
            refresh_local_table_asset(local_table_path)
        if not joint_table_path.exists():
            refresh_joint_table_asset(joint_table_path)

    missing = []
    if not local_table_path.exists():
        missing.append(str(local_table_path))
    if not joint_table_path.exists():
        missing.append(str(joint_table_path))

    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            "missing cached assets: "
            f"{missing_str}. "
            "Run `python -m prototype.two_table_compiler.build_assets --refresh-all` "
            "or rerun demo with `--auto-build-assets` or `--use-demo-table`."
        )

    return load_local_table(local_table_path), load_joint_table(joint_table_path)


def run(
    target: str,
    local_table: LocalTable,
    joint_table: JointTable,
    max_popped_states: int | None = 2_000,
) -> int:
    p0_count = len(generate_p0_sources(local_table))
    p2_count = len(generate_p2_sources(local_table, joint_table))
    classification = classify_target(target, local_table)
    search_result: SynthesisSearchResult | None = None
    if classification.canonical_tail is not None and not classification.frame_unreachable:
        search_result = synthesize_search(
            classification.canonical_tail,
            local_table,
            joint_table,
            max_popped_states=max_popped_states,
        )

    print(f"target input: {target}")
    if classification.canonical_logical is not None:
        print(f"target canonical logical: {classification.canonical_logical}")
    if classification.canonical_tail is not None:
        print(f"target tail: {classification.canonical_tail.compact()}")
    else:
        print("target tail: <none>")
    print(f"classification native family hits: {','.join(classification.matched_native_ids()) or '<none>'}")
    _print_local_summary(local_table)
    _print_joint_summary(joint_table)
    print(f"P0 source count: {p0_count}")
    print(f"P2 source count: {p2_count}")

    search_status = (
        SearchStatus.RULE_UNREACHABLE
        if classification.frame_unreachable or classification.canonical_tail is None
        else search_result.status
    )
    assert_no_budget_status_invariant(
        status=search_status,
        max_popped_states=max_popped_states,
        context=f"demo target={target}",
    )
    print(f"search status: {search_status.value}")
    if search_result is not None and search_result.search_truncated:
        print("search truncated: yes")
    else:
        print("search truncated: no")

    if search_status == SearchStatus.RULE_UNREACHABLE:
        print("reachable verdict: RULE_UNREACHABLE")
        if classification.reason:
            print(f"reason: {classification.reason}")
        print("total cost beta(t): None")
        print("best final hidden class: None")
        print("remote inter-module count: 0")
        print("path details:")
        print("  <none>")
        print("final projected result:")
        print("  tail in M: no")
        return 1

    if search_result is None:
        raise RuntimeError("search_result is missing despite non-rule-unreachable classification")
    plan = search_result.plan
    if search_status == SearchStatus.SEARCH_TRUNCATED:
        print("reachable verdict: SEARCH_TRUNCATED")
        print("reason: search truncated before any witness was proven")
        print("total cost beta(t): None")
        print("best final hidden class: None")
        print("remote inter-module count: 0")
        print("path details:")
        print("  <none>")
        print("final projected result:")
        print("  tail in M: unknown")
        if classification.canonical_tail is not None:
            print(f"  best_cost[{classification.canonical_tail.compact()}] = None")
        return 1
    if plan is None:
        raise RuntimeError(f"unexpected {search_status.value} outcome without plan")

    print(f"reachable verdict: {search_status.value}")
    if search_status == SearchStatus.FOUND_REACHABLE_UPPER_BOUND:
        print(f"reason: {search_status_note(search_status)}")
    print(f"total cost beta(t): {plan.total_cost}")
    print(f"best final hidden class: {plan.best_hidden.name}")
    print(f"remote inter-module count: {plan.remote_inter_module_count}")
    print("path details:")

    cumulative = 0
    for idx, step in enumerate(plan.steps):
        cumulative += step.delta_cost
        src = "SOURCE" if step.src is None else _format_state(step.src)
        dst = _format_state(step.dst)
        native_ids = ",".join(step.native_ids)
        print(
            f"  step index={idx} rule={step.family} kind={_step_kind_label(step.family)} "
            f"src state={src} dst state={dst} incremental cost={step.delta_cost} "
            f"cumulative cost={cumulative} native ids used={native_ids} "
            f"input_native_ids={list(step.input_native_ids)} input_scopes={list(step.input_scopes)} "
            f"applied_rule_name={step.applied_rule_name or step.rule_name} "
            f"scope={step.scope} support={list(step.support)} metadata={_format_metadata(step.metadata)}"
        )

    print("final projected result:")
    print("  tail in M: yes")
    print(f"  best_cost[{plan.target_tail.compact()}] = {plan.projected_best_cost}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-table prototype compiler with cached assets")
    parser.add_argument(
        "--target",
        required=True,
        help="target tail or logical expression, e.g. XIIIIIIXIII or \"X1P⊗X1P'\"",
    )
    parser.add_argument("--use-demo-table", action="store_true", help="use small in-memory demo table fallback")
    parser.add_argument("--auto-build-assets", action="store_true", help="build missing cached assets automatically")
    parser.add_argument("--local-table-path", default=str(DEFAULT_LOCAL_TABLE_PATH))
    parser.add_argument("--joint-table-path", default=str(DEFAULT_JOINT_TABLE_PATH))
    parser.add_argument(
        "--max-popped-states",
        type=int,
        default=2_000,
        help="optional search budget for large cached tables (set <=0 for no limit)",
    )
    args = parser.parse_args()
    local_table_path = Path(args.local_table_path)
    joint_table_path = Path(args.joint_table_path)

    try:
        local_table, joint_table = _resolve_tables(
            use_demo_table=args.use_demo_table,
            auto_build_assets=args.auto_build_assets,
            local_table_path=local_table_path,
            joint_table_path=joint_table_path,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2

    max_popped_states = args.max_popped_states if args.max_popped_states > 0 else None
    return run(args.target, local_table, joint_table, max_popped_states=max_popped_states)


if __name__ == "__main__":
    raise SystemExit(main())

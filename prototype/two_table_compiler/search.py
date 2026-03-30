from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum
from math import inf
from typing import Dict, Iterable, List, Optional, Tuple

from .rules import Step, apply_p3_transitions, expand_p3_edges, generate_p0_sources, generate_p2_sources
from .state import DEFAULT_ACCEPTING_HIDDEN_CLASSES, HiddenClass, State, Tail, hidden_from_head
from .tables import JointTable, LocalTable, SCOPE_INTER_MODULE_BELL
from .targeting import TargetClassification, classify_target


@dataclass
class SynthesisPlan:
    target_tail: Tail
    best_state: State
    best_hidden: HiddenClass
    total_cost: int
    projected_best_cost: int
    p0_source_count: int
    p2_source_count: int
    remote_inter_module_count: int
    steps: List[Step]


class SearchStatus(str, Enum):
    FOUND_OPTIMAL = "FOUND_OPTIMAL"
    FOUND_REACHABLE_UPPER_BOUND = "FOUND_REACHABLE_UPPER_BOUND"
    SEARCH_TRUNCATED = "SEARCH_TRUNCATED"
    RULE_UNREACHABLE = "RULE_UNREACHABLE"


@dataclass
class SynthesisSearchResult:
    status: SearchStatus
    plan: Optional[SynthesisPlan]
    search_truncated: bool
    in_p0_source: bool
    in_p2_source: bool
    p0_source_hit_count: int
    p2_source_hit_count: int
    any_state_with_same_tail_in_dist: bool
    any_accepting_state_with_same_tail_in_dist: bool
    projected_best_cost_exists: bool
    projected_best_cost: int | None
    popped_states: int


@dataclass
class SynthesisOutcome:
    classification: TargetClassification
    plan: Optional[SynthesisPlan]
    frame_unreachable: bool
    reason: str | None
    search_status: SearchStatus
    search_truncated: bool = False
    projected_best_cost_exists: bool = False


@dataclass(frozen=True)
class TargetAuditReport:
    target_tail: Tail
    search_status: SearchStatus
    in_p0_source: bool
    in_p2_source: bool
    p0_source_hit_count: int
    p2_source_hit_count: int
    any_state_with_same_tail_in_dist: bool
    any_accepting_state_with_same_tail_in_dist: bool
    projected_best_cost_exists: bool
    projected_best_cost: int | None
    popped_states: int
    plan_cost: int | None
    first_family_if_any: str | None
    reason: str | None
    y_debug: Dict[str, object] | None = None


def search_status_name(status: SearchStatus | str) -> str:
    return status.value if isinstance(status, SearchStatus) else str(status)


def search_status_flags(status: SearchStatus | str) -> Dict[str, bool]:
    name = search_status_name(status)
    if name == SearchStatus.FOUND_OPTIMAL.value:
        return {
            "reachable": True,
            "optimal": True,
            "truncated": False,
            "rule_unreachable": False,
        }
    if name == SearchStatus.FOUND_REACHABLE_UPPER_BOUND.value:
        return {
            "reachable": True,
            "optimal": False,
            "truncated": False,
            "rule_unreachable": False,
        }
    if name == SearchStatus.SEARCH_TRUNCATED.value:
        return {
            "reachable": False,
            "optimal": False,
            "truncated": True,
            "rule_unreachable": False,
        }
    if name == SearchStatus.RULE_UNREACHABLE.value:
        return {
            "reachable": False,
            "optimal": False,
            "truncated": False,
            "rule_unreachable": True,
        }
    raise ValueError(f"unknown search status: {status!r}")


def search_status_note(status: SearchStatus | str) -> str:
    name = search_status_name(status)
    if name == SearchStatus.FOUND_OPTIMAL.value:
        return "optimal witness proven"
    if name == SearchStatus.FOUND_REACHABLE_UPPER_BOUND.value:
        return "upper bound found, not yet proven optimal"
    if name == SearchStatus.SEARCH_TRUNCATED.value:
        return "search truncated before any witness was proven"
    if name == SearchStatus.RULE_UNREACHABLE.value:
        return "rule-unreachable after full search"
    raise ValueError(f"unknown search status: {status!r}")


def is_reachable_status(status: SearchStatus | str) -> bool:
    return bool(search_status_flags(status)["reachable"])


def assert_no_budget_status_invariant(
    *,
    status: SearchStatus | str,
    max_popped_states: int | None,
    context: str,
) -> None:
    if max_popped_states is not None:
        return
    name = search_status_name(status)
    if name in (
        SearchStatus.FOUND_REACHABLE_UPPER_BOUND.value,
        SearchStatus.SEARCH_TRUNCATED.value,
    ):
        raise AssertionError(f"no-budget invariant violated for {context}: status={name}")


def _project_best_state(
    dist: Dict[State, int],
    target_tail: Tail,
    accepting: set[HiddenClass],
) -> Optional[State]:
    candidates = [state for state in dist if state.tail == target_tail and state.hidden in accepting]
    if not candidates:
        return None
    return min(candidates, key=lambda state: dist[state])


def _reconstruct_plan(
    *,
    goal: State,
    target_tail: Tail,
    dist: Dict[State, int],
    seed_step: Dict[State, Step],
    predecessor: Dict[State, Step],
    p0_source_count: int,
    p2_source_count: int,
) -> SynthesisPlan:
    p3_reverse_path: List[Step] = []
    cursor = goal
    while cursor in predecessor:
        step = predecessor[cursor]
        p3_reverse_path.append(step)
        if step.src is None:
            break
        cursor = step.src

    source = seed_step[cursor]
    full_path = [source] + list(reversed(p3_reverse_path))
    remote_count = sum(1 for step in full_path if step.scope == SCOPE_INTER_MODULE_BELL)
    return SynthesisPlan(
        target_tail=target_tail,
        best_state=goal,
        best_hidden=goal.hidden,
        total_cost=dist[goal],
        projected_best_cost=dist[goal],
        p0_source_count=p0_source_count,
        p2_source_count=p2_source_count,
        remote_inter_module_count=remote_count,
        steps=full_path,
    )


def _run_search(
    target_tail: Tail,
    local_table: LocalTable,
    joint_table: JointTable,
    accepting_hidden_classes: Optional[Iterable[HiddenClass]] = None,
    max_popped_states: Optional[int] = None,
    enable_p2: bool = True,
    p2_use_cache: bool = True,
) -> SynthesisSearchResult:
    accepting = set(accepting_hidden_classes or DEFAULT_ACCEPTING_HIDDEN_CLASSES)

    p0_sources = generate_p0_sources(local_table)
    p2_sources = generate_p2_sources(local_table, joint_table, use_cache=p2_use_cache) if enable_p2 else []
    source_steps = p0_sources + p2_sources
    p0_source_hit_count = sum(
        1 for step in p0_sources if step.dst.tail == target_tail and step.dst.hidden in accepting
    )
    p2_source_hit_count = sum(
        1 for step in p2_sources if step.dst.tail == target_tail and step.dst.hidden in accepting
    )
    in_p0_source = p0_source_hit_count > 0
    in_p2_source = p2_source_hit_count > 0
    if not source_steps:
        return SynthesisSearchResult(
            status=SearchStatus.RULE_UNREACHABLE,
            plan=None,
            search_truncated=False,
            in_p0_source=in_p0_source,
            in_p2_source=in_p2_source,
            p0_source_hit_count=p0_source_hit_count,
            p2_source_hit_count=p2_source_hit_count,
            any_state_with_same_tail_in_dist=False,
            any_accepting_state_with_same_tail_in_dist=False,
            projected_best_cost_exists=False,
            projected_best_cost=None,
            popped_states=0,
        )

    dist: Dict[State, int] = {}
    seed_step: Dict[State, Step] = {}
    predecessor: Dict[State, Step] = {}

    heap: List[Tuple[int, int, State]] = []
    counter = 0
    for step in source_steps:
        state = step.dst
        cost = step.delta_cost
        if cost < dist.get(state, inf):
            dist[state] = cost
            seed_step[state] = step
            heapq.heappush(heap, (cost, counter, state))
            counter += 1

    goal: Optional[State] = None
    popped_states = 0
    search_truncated = False

    while heap:
        cur_cost, _, cur_state = heapq.heappop(heap)
        if cur_cost != dist.get(cur_state):
            continue
        popped_states += 1

        if cur_state.tail == target_tail and cur_state.hidden in accepting:
            goal = cur_state
            break

        if max_popped_states is not None and popped_states > max_popped_states:
            search_truncated = True
            break

        for step in expand_p3_edges(cur_state, local_table, joint_table):
            nxt_state = step.dst
            nxt_cost = cur_cost + step.delta_cost
            if nxt_cost < dist.get(nxt_state, inf):
                dist[nxt_state] = nxt_cost
                predecessor[nxt_state] = step
                heapq.heappush(heap, (nxt_cost, counter, nxt_state))
                counter += 1

    any_same_tail = any(state.tail == target_tail for state in dist)
    best_projected_state = _project_best_state(dist, target_tail, accepting)
    has_projected_cost = best_projected_state is not None
    projected_best_cost = None if best_projected_state is None else int(dist[best_projected_state])

    if goal is not None:
        return SynthesisSearchResult(
            status=SearchStatus.FOUND_OPTIMAL,
            plan=_reconstruct_plan(
                goal=goal,
                target_tail=target_tail,
                dist=dist,
                seed_step=seed_step,
                predecessor=predecessor,
                p0_source_count=len(p0_sources),
                p2_source_count=len(p2_sources),
            ),
            search_truncated=search_truncated,
            in_p0_source=in_p0_source,
            in_p2_source=in_p2_source,
            p0_source_hit_count=p0_source_hit_count,
            p2_source_hit_count=p2_source_hit_count,
            any_state_with_same_tail_in_dist=any_same_tail,
            any_accepting_state_with_same_tail_in_dist=has_projected_cost,
            projected_best_cost_exists=has_projected_cost,
            projected_best_cost=projected_best_cost,
            popped_states=popped_states,
        )

    if best_projected_state is not None:
        return SynthesisSearchResult(
            status=SearchStatus.FOUND_REACHABLE_UPPER_BOUND,
            plan=_reconstruct_plan(
                goal=best_projected_state,
                target_tail=target_tail,
                dist=dist,
                seed_step=seed_step,
                predecessor=predecessor,
                p0_source_count=len(p0_sources),
                p2_source_count=len(p2_sources),
            ),
            search_truncated=search_truncated,
            in_p0_source=in_p0_source,
            in_p2_source=in_p2_source,
            p0_source_hit_count=p0_source_hit_count,
            p2_source_hit_count=p2_source_hit_count,
            any_state_with_same_tail_in_dist=any_same_tail,
            any_accepting_state_with_same_tail_in_dist=True,
            projected_best_cost_exists=has_projected_cost,
            projected_best_cost=projected_best_cost,
            popped_states=popped_states,
        )

    if search_truncated:
        return SynthesisSearchResult(
            status=SearchStatus.SEARCH_TRUNCATED,
            plan=None,
            search_truncated=True,
            in_p0_source=in_p0_source,
            in_p2_source=in_p2_source,
            p0_source_hit_count=p0_source_hit_count,
            p2_source_hit_count=p2_source_hit_count,
            any_state_with_same_tail_in_dist=any_same_tail,
            any_accepting_state_with_same_tail_in_dist=False,
            projected_best_cost_exists=False,
            projected_best_cost=None,
            popped_states=popped_states,
        )

    return SynthesisSearchResult(
        status=SearchStatus.RULE_UNREACHABLE,
        plan=None,
        search_truncated=False,
        in_p0_source=in_p0_source,
        in_p2_source=in_p2_source,
        p0_source_hit_count=p0_source_hit_count,
        p2_source_hit_count=p2_source_hit_count,
        any_state_with_same_tail_in_dist=any_same_tail,
        any_accepting_state_with_same_tail_in_dist=False,
        projected_best_cost_exists=False,
        projected_best_cost=None,
        popped_states=popped_states,
    )


def synthesize_search(
    target_tail: Tail,
    local_table: LocalTable,
    joint_table: JointTable,
    accepting_hidden_classes: Optional[Iterable[HiddenClass]] = None,
    max_popped_states: Optional[int] = None,
    enable_p2: bool = True,
    p2_use_cache: bool = True,
) -> SynthesisSearchResult:
    return _run_search(
        target_tail=target_tail,
        local_table=local_table,
        joint_table=joint_table,
        accepting_hidden_classes=accepting_hidden_classes,
        max_popped_states=max_popped_states,
        enable_p2=enable_p2,
        p2_use_cache=p2_use_cache,
    )


def synthesize(
    target_tail: Tail,
    local_table: LocalTable,
    joint_table: JointTable,
    accepting_hidden_classes: Optional[Iterable[HiddenClass]] = None,
    max_popped_states: Optional[int] = None,
    enable_p2: bool = True,
    p2_use_cache: bool = True,
) -> Optional[SynthesisPlan]:
    """
    Legacy convenience wrapper that returns only the best plan.

    Prefer `synthesize_search` or `synthesize_target` when callers must distinguish
    SEARCH_TRUNCATED from RULE_UNREACHABLE.
    """
    result = synthesize_search(
        target_tail=target_tail,
        local_table=local_table,
        joint_table=joint_table,
        accepting_hidden_classes=accepting_hidden_classes,
        max_popped_states=max_popped_states,
        enable_p2=enable_p2,
        p2_use_cache=p2_use_cache,
    )
    return result.plan


def synthesize_target(
    target: Tail | str,
    local_table: LocalTable,
    joint_table: JointTable,
    accepting_hidden_classes: Optional[Iterable[HiddenClass]] = None,
    max_popped_states: Optional[int] = None,
    enable_p2: bool = True,
    p2_use_cache: bool = True,
    target_aware_p2_fallback: bool = False,
) -> SynthesisOutcome:
    _ = target_aware_p2_fallback  # compatibility only; target-aware P2 fallback is disabled by design.
    accepting = set(accepting_hidden_classes or DEFAULT_ACCEPTING_HIDDEN_CLASSES)
    classification = classify_target(target, local_table)

    if classification.frame_unreachable:
        return SynthesisOutcome(
            classification=classification,
            plan=None,
            frame_unreachable=True,
            reason=classification.reason,
            search_status=SearchStatus.RULE_UNREACHABLE,
            search_truncated=False,
            projected_best_cost_exists=False,
        )

    if classification.canonical_tail is None:
        return SynthesisOutcome(
            classification=classification,
            plan=None,
            frame_unreachable=True,
            reason="unable to canonicalize target tail",
            search_status=SearchStatus.RULE_UNREACHABLE,
            search_truncated=False,
            projected_best_cost_exists=False,
        )

    target_tail = classification.canonical_tail
    search_result = _run_search(
        target_tail=target_tail,
        local_table=local_table,
        joint_table=joint_table,
        accepting_hidden_classes=accepting,
        max_popped_states=max_popped_states,
        enable_p2=enable_p2,
        p2_use_cache=p2_use_cache,
    )

    if search_result.plan is not None:
        reason = None
        if search_result.status == SearchStatus.FOUND_REACHABLE_UPPER_BOUND:
            reason = (
                f"search reached an upper-bound witness for target {target_tail.compact()} "
                f"before optimality proof; popped_states={search_result.popped_states}"
            )
        return SynthesisOutcome(
            classification=classification,
            plan=search_result.plan,
            frame_unreachable=False,
            reason=reason,
            search_status=search_result.status,
            search_truncated=search_result.search_truncated,
            projected_best_cost_exists=search_result.projected_best_cost_exists,
        )

    if search_result.status == SearchStatus.SEARCH_TRUNCATED:
        return SynthesisOutcome(
            classification=classification,
            plan=None,
            frame_unreachable=False,
            reason=(
                f"search truncated at popped_states={search_result.popped_states}; "
                "no projected witness discovered yet"
            ),
            search_status=SearchStatus.SEARCH_TRUNCATED,
            search_truncated=True,
            projected_best_cost_exists=False,
        )

    return SynthesisOutcome(
        classification=classification,
        plan=None,
        frame_unreachable=True,
        reason=(
            f"frame {local_table.frame_id} cannot synthesize target {target_tail.compact()} "
            "under current P0/P2/P3 rules after full search; no automatic Bell fallback"
        ),
        search_status=SearchStatus.RULE_UNREACHABLE,
        search_truncated=False,
        projected_best_cost_exists=False,
    )


def _single_qubit_axis_tails(index: int) -> tuple[Tail, Tail, Tail]:
    symbols_x = ["I"] * len(Tail.identity().paulis)
    symbols_y = ["I"] * len(Tail.identity().paulis)
    symbols_z = ["I"] * len(Tail.identity().paulis)
    symbols_x[index] = "X"
    symbols_y[index] = "Y"
    symbols_z[index] = "Z"
    return Tail(tuple(symbols_x)), Tail(tuple(symbols_y)), Tail(tuple(symbols_z))


def _y_debug_info(
    target_tail: Tail,
    local_table: LocalTable,
    joint_table: JointTable,
) -> Dict[str, object] | None:
    non_identity = [(idx, symbol) for idx, symbol in enumerate(target_tail.paulis) if symbol != "I"]
    if len(non_identity) != 1:
        return None
    idx, symbol = non_identity[0]
    if symbol != "Y":
        return None

    x_tail, _, z_tail = _single_qubit_axis_tails(idx)
    x_seed_exists = any(
        step.dst.tail == x_tail and step.dst.hidden == hidden_from_head("X")
        for step in generate_p0_sources(local_table)
    )
    z_axes = [entry for entry in local_table.entries if entry.head == "Z" and entry.tail == z_tail]
    transitions: List[Step] = []
    if z_axes:
        start_state = State(hidden=hidden_from_head("X"), tail=x_tail)
        for axis in z_axes:
            transitions.extend(apply_p3_transitions(start_state, axis, joint_table))

    return {
        "x_seed_exists": x_seed_exists,
        "z_axis_exists": len(z_axes) > 0,
        "z_axis_native_ids": [axis.native_id for axis in z_axes],
        "p3_transition_count": len(transitions),
        "p3_dst_tails": [step.dst.tail.compact() for step in transitions],
        "p3_dst_hidden": [step.dst.hidden.name for step in transitions],
    }


def audit_target_execution(
    target: Tail | str,
    local_table: LocalTable,
    joint_table: JointTable,
    *,
    accepting_hidden_classes: Optional[Iterable[HiddenClass]] = None,
    max_popped_states: Optional[int] = None,
    enable_p2: bool = True,
    p2_use_cache: bool = False,
    target_aware_p2_fallback: bool = False,
) -> TargetAuditReport:
    _ = target_aware_p2_fallback  # compatibility only; target-aware P2 fallback is disabled by design.
    accepting = set(accepting_hidden_classes or DEFAULT_ACCEPTING_HIDDEN_CLASSES)
    classification = classify_target(target, local_table)
    canonical = classification.canonical_tail
    if canonical is None:
        raise ValueError(f"cannot audit target without canonical tail: {target!r}")

    if classification.frame_unreachable:
        y_debug = _y_debug_info(canonical, local_table, joint_table)
        return TargetAuditReport(
            target_tail=canonical,
            search_status=SearchStatus.RULE_UNREACHABLE,
            in_p0_source=False,
            in_p2_source=False,
            p0_source_hit_count=0,
            p2_source_hit_count=0,
            any_state_with_same_tail_in_dist=False,
            any_accepting_state_with_same_tail_in_dist=False,
            projected_best_cost_exists=False,
            projected_best_cost=None,
            popped_states=0,
            plan_cost=None,
            first_family_if_any=None,
            reason=classification.reason,
            y_debug=y_debug,
        )

    search_result = _run_search(
        target_tail=canonical,
        local_table=local_table,
        joint_table=joint_table,
        accepting_hidden_classes=accepting,
        max_popped_states=max_popped_states,
        enable_p2=enable_p2,
        p2_use_cache=p2_use_cache,
    )

    plan_cost = search_result.plan.total_cost if search_result.plan is not None else None
    first_family_if_any = None
    if search_result.plan is not None and search_result.plan.steps:
        first_family_if_any = search_result.plan.steps[0].family
    status = search_result.status
    reason: str | None = None
    projected_exists = search_result.projected_best_cost_exists

    if reason is None:
        if status == SearchStatus.FOUND_REACHABLE_UPPER_BOUND:
            reason = (
                f"search reached an upper-bound witness for target {canonical.compact()} "
                f"before optimality proof; popped_states={search_result.popped_states}"
            )
        elif status == SearchStatus.SEARCH_TRUNCATED:
            reason = (
                f"search truncated at popped_states={search_result.popped_states}; "
                "no projected witness discovered yet"
            )
        elif status == SearchStatus.RULE_UNREACHABLE:
            reason = (
                f"frame {local_table.frame_id} cannot synthesize target {canonical.compact()} "
                "under current P0/P2/P3 rules after full search; no automatic Bell fallback"
            )

    y_debug = _y_debug_info(canonical, local_table, joint_table)
    return TargetAuditReport(
        target_tail=canonical,
        search_status=status,
        in_p0_source=search_result.in_p0_source,
        in_p2_source=search_result.in_p2_source,
        p0_source_hit_count=search_result.p0_source_hit_count,
        p2_source_hit_count=search_result.p2_source_hit_count,
        any_state_with_same_tail_in_dist=search_result.any_state_with_same_tail_in_dist,
        any_accepting_state_with_same_tail_in_dist=search_result.any_accepting_state_with_same_tail_in_dist,
        projected_best_cost_exists=projected_exists,
        projected_best_cost=search_result.projected_best_cost,
        popped_states=search_result.popped_states,
        plan_cost=plan_cost,
        first_family_if_any=first_family_if_any,
        reason=reason,
        y_debug=y_debug,
    )

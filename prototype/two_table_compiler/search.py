from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum
from math import inf
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple

from .rules import Step, apply_p3_transitions, expand_p3_edges, generate_p0_sources, generate_p2_sources
from .state import DEFAULT_ACCEPTING_HIDDEN_CLASSES, N_QUBITS, HiddenClass, State, Tail, hidden_from_head
from .tables import JointTable, LocalTable, NativeEntry, SCOPE_INTER_MODULE_BELL
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
    closure_build_seconds: float | None = None
    closure_lookup_seconds: float | None = None
    closure_witness_seconds: float | None = None
    closure_cache_hit: bool | None = None


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


@dataclass(frozen=True)
class FrameClosureProfile:
    precompute_seconds: float
    dijkstra_seconds: float
    p0_source_count: int
    p2_source_count: int
    source_state_count: int
    reachable_state_count: int
    accepting_tail_count: int
    popped_states: int
    relaxed_edges: int
    p3_attempts: int
    p3_successes: int
    heap_pushes: int


@dataclass(frozen=True)
class FrameClosure:
    requested_tails: frozenset[Tail] | None
    is_exhaustive: bool
    accepting_hidden_classes: Tuple[HiddenClass, ...]
    p0_source_count: int
    p2_source_count: int
    dist: Dict[int, int]
    seed_step: Dict[int, Step]
    predecessor: Dict[int, "P3PredecessorRecord"]
    reachable_tails: frozenset[Tail]
    best_accepting_state_by_tail: Dict[Tail, int]
    best_cost_by_tail: Dict[Tail, int]
    p0_source_hit_count_by_tail: Dict[Tail, int]
    p2_source_hit_count_by_tail: Dict[Tail, int]
    profile: FrameClosureProfile


@dataclass(frozen=True)
class P3PredecessorRecord:
    src_code: int
    dst_code: int
    native_id: str
    applied_rule_name: str
    delta_cost: int


_PAULI_TO_DIGIT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_DIGIT_TO_PAULI = ("I", "X", "Y", "Z")
_HIDDEN_TO_DIGIT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_DIGIT_TO_HIDDEN = ("I", "X", "Y", "Z")
_TAIL_SPACE_SIZE = 4**N_QUBITS


def _tail_to_code(tail: Tail) -> int:
    code = 0
    for symbol in tail.paulis:
        code = (code * 4) + _PAULI_TO_DIGIT[symbol]
    return code


def _tail_from_code(code: int) -> Tail:
    digits = [0] * N_QUBITS
    cursor = code
    for index in range(N_QUBITS - 1, -1, -1):
        cursor, rem = divmod(cursor, 4)
        digits[index] = rem
    return Tail(tuple(_DIGIT_TO_PAULI[digit] for digit in digits))


def _state_to_code(state: State) -> int:
    hidden_digit = _HIDDEN_TO_DIGIT[state.hidden.name]
    return hidden_digit * _TAIL_SPACE_SIZE + _tail_to_code(state.tail)


def _state_from_code(code: int) -> State:
    hidden_digit, tail_code = divmod(code, _TAIL_SPACE_SIZE)
    hidden = hidden_from_head(_DIGIT_TO_HIDDEN[hidden_digit])
    return State(hidden=hidden, tail=_tail_from_code(tail_code))


_FRAME_CLOSURE_CACHE: Dict[Tuple[object, ...], List[FrameClosure]] = {}
_FRAME_CLOSURE_CACHE_HITS = 0
_FRAME_CLOSURE_CACHE_MISSES = 0
_FRAME_CLOSURE_BUILD_COUNT = 0
_FRAME_CLOSURE_BUILD_SECONDS_TOTAL = 0.0
_FRAME_CLOSURE_LOOKUP_COUNT = 0
_FRAME_CLOSURE_LOOKUP_SECONDS_TOTAL = 0.0
_FRAME_CLOSURE_WITNESS_COUNT = 0
_FRAME_CLOSURE_WITNESS_SECONDS_TOTAL = 0.0


def _normalize_accepting_hidden_classes(
    accepting_hidden_classes: Optional[Iterable[HiddenClass]],
) -> Tuple[HiddenClass, ...]:
    accepting = set(accepting_hidden_classes or DEFAULT_ACCEPTING_HIDDEN_CLASSES)
    return tuple(sorted(accepting, key=lambda hidden: hidden.name))


def _frame_closure_base_key(
    *,
    local_table: LocalTable,
    joint_table: JointTable,
    accepting_hidden_classes: Tuple[HiddenClass, ...],
    enable_p2: bool,
    p2_use_cache: bool,
) -> Tuple[object, ...]:
    return (
        id(local_table),
        len(local_table.entries),
        id(joint_table),
        len(joint_table.p2_rules),
        len(joint_table.p3_rules),
        tuple(hidden.name for hidden in accepting_hidden_classes),
        bool(enable_p2),
        bool(p2_use_cache),
    )


def _closure_covers_tail(closure: FrameClosure, target_tail: Tail) -> bool:
    if closure.is_exhaustive:
        return True
    if closure.requested_tails is None:
        return False
    return target_tail in closure.requested_tails


def _find_cached_closure_for_target(
    *,
    local_table: LocalTable,
    joint_table: JointTable,
    accepting_hidden_classes: Tuple[HiddenClass, ...],
    enable_p2: bool,
    p2_use_cache: bool,
    target_tail: Tail,
) -> FrameClosure | None:
    global _FRAME_CLOSURE_CACHE_HITS
    base_key = _frame_closure_base_key(
        local_table=local_table,
        joint_table=joint_table,
        accepting_hidden_classes=accepting_hidden_classes,
        enable_p2=enable_p2,
        p2_use_cache=p2_use_cache,
    )
    closures = _FRAME_CLOSURE_CACHE.get(base_key, [])
    for closure in closures:
        if _closure_covers_tail(closure, target_tail):
            _FRAME_CLOSURE_CACHE_HITS += 1
            return closure
    return None


def frame_closure_cache_stats() -> Dict[str, int]:
    total_entries = sum(len(closures) for closures in _FRAME_CLOSURE_CACHE.values())
    return {
        "cache_hits": _FRAME_CLOSURE_CACHE_HITS,
        "cache_misses": _FRAME_CLOSURE_CACHE_MISSES,
        "closure_cache_hit_count": _FRAME_CLOSURE_CACHE_HITS,
        "closure_cache_miss_count": _FRAME_CLOSURE_CACHE_MISSES,
        "cache_entries": total_entries,
        "cache_base_keys": len(_FRAME_CLOSURE_CACHE),
        "closure_build_count": _FRAME_CLOSURE_BUILD_COUNT,
        "closure_build_seconds_total": int(_FRAME_CLOSURE_BUILD_SECONDS_TOTAL * 1_000_000),
        "closure_lookup_count": _FRAME_CLOSURE_LOOKUP_COUNT,
        "closure_lookup_seconds_total": int(_FRAME_CLOSURE_LOOKUP_SECONDS_TOTAL * 1_000_000),
        "closure_witness_count": _FRAME_CLOSURE_WITNESS_COUNT,
        "closure_witness_seconds_total": int(_FRAME_CLOSURE_WITNESS_SECONDS_TOTAL * 1_000_000),
    }


def clear_frame_closure_cache() -> None:
    global _FRAME_CLOSURE_CACHE_HITS
    global _FRAME_CLOSURE_CACHE_MISSES
    global _FRAME_CLOSURE_BUILD_COUNT
    global _FRAME_CLOSURE_BUILD_SECONDS_TOTAL
    global _FRAME_CLOSURE_LOOKUP_COUNT
    global _FRAME_CLOSURE_LOOKUP_SECONDS_TOTAL
    global _FRAME_CLOSURE_WITNESS_COUNT
    global _FRAME_CLOSURE_WITNESS_SECONDS_TOTAL
    _FRAME_CLOSURE_CACHE.clear()
    _FRAME_CLOSURE_CACHE_HITS = 0
    _FRAME_CLOSURE_CACHE_MISSES = 0
    _FRAME_CLOSURE_BUILD_COUNT = 0
    _FRAME_CLOSURE_BUILD_SECONDS_TOTAL = 0.0
    _FRAME_CLOSURE_LOOKUP_COUNT = 0
    _FRAME_CLOSURE_LOOKUP_SECONDS_TOTAL = 0.0
    _FRAME_CLOSURE_WITNESS_COUNT = 0
    _FRAME_CLOSURE_WITNESS_SECONDS_TOTAL = 0.0


def precompute_frame_closure(
    local_table: LocalTable,
    joint_table: JointTable,
    *,
    accepting_hidden_classes: Optional[Iterable[HiddenClass]] = None,
    enable_p2: bool = True,
    p2_use_cache: bool = True,
    target_tails: Optional[Iterable[Tail]] = None,
) -> FrameClosure:
    global _FRAME_CLOSURE_CACHE_HITS
    global _FRAME_CLOSURE_CACHE_MISSES
    global _FRAME_CLOSURE_BUILD_COUNT
    global _FRAME_CLOSURE_BUILD_SECONDS_TOTAL
    accepting = _normalize_accepting_hidden_classes(accepting_hidden_classes)
    accepting_set = set(accepting)
    base_key = _frame_closure_base_key(
        local_table=local_table,
        joint_table=joint_table,
        accepting_hidden_classes=accepting,
        enable_p2=enable_p2,
        p2_use_cache=p2_use_cache,
    )
    requested_tails = None if target_tails is None else frozenset(target_tails)
    cached_closures = _FRAME_CLOSURE_CACHE.get(base_key, [])
    if requested_tails is None:
        for closure in cached_closures:
            if closure.is_exhaustive:
                _FRAME_CLOSURE_CACHE_HITS += 1
                return closure
    else:
        for closure in cached_closures:
            if closure.is_exhaustive:
                _FRAME_CLOSURE_CACHE_HITS += 1
                return closure
            if closure.requested_tails is not None and requested_tails.issubset(closure.requested_tails):
                _FRAME_CLOSURE_CACHE_HITS += 1
                return closure

    _FRAME_CLOSURE_CACHE_MISSES += 1
    started = perf_counter()
    p0_sources = generate_p0_sources(local_table)
    p2_sources = generate_p2_sources(local_table, joint_table, use_cache=p2_use_cache) if enable_p2 else []
    source_steps = p0_sources + p2_sources

    p0_source_hit_count_by_tail: Dict[Tail, int] = {}
    p2_source_hit_count_by_tail: Dict[Tail, int] = {}
    for step in p0_sources:
        if step.dst.hidden in accepting_set:
            p0_source_hit_count_by_tail[step.dst.tail] = p0_source_hit_count_by_tail.get(step.dst.tail, 0) + 1
    for step in p2_sources:
        if step.dst.hidden in accepting_set:
            p2_source_hit_count_by_tail[step.dst.tail] = p2_source_hit_count_by_tail.get(step.dst.tail, 0) + 1

    dist: Dict[int, int] = {}
    seed_step: Dict[int, Step] = {}
    predecessor: Dict[int, P3PredecessorRecord] = {}
    heap: List[Tuple[int, int, int]] = []
    counter = 0
    heap_pushes = 0
    reachable_tails: set[Tail] = set()

    for step in source_steps:
        state_code = _state_to_code(step.dst)
        cost = step.delta_cost
        reachable_tails.add(step.dst.tail)
        if cost < dist.get(state_code, inf):
            dist[state_code] = cost
            seed_step[state_code] = step
            heapq.heappush(heap, (cost, counter, state_code))
            counter += 1
            heap_pushes += 1

    dijkstra_started = perf_counter()
    popped_states = 0
    relaxed_edges = 0
    p3_successes = 0
    best_accepting_state_by_tail: Dict[Tail, int] = {}
    best_cost_by_tail: Dict[Tail, int] = {}
    unresolved_tails = set(requested_tails or ())
    terminated_early = requested_tails is not None and len(unresolved_tails) == 0
    while heap:
        if terminated_early:
            break
        cur_cost, _, cur_state_code = heapq.heappop(heap)
        if cur_cost != dist.get(cur_state_code):
            continue
        cur_state = _state_from_code(cur_state_code)
        popped_states += 1
        if cur_state.hidden in accepting_set and cur_state.tail not in best_cost_by_tail:
            best_cost_by_tail[cur_state.tail] = int(cur_cost)
            best_accepting_state_by_tail[cur_state.tail] = cur_state_code
            if requested_tails is not None and cur_state.tail in unresolved_tails:
                unresolved_tails.remove(cur_state.tail)
                if not unresolved_tails:
                    terminated_early = True
                    continue
        for step in expand_p3_edges(cur_state, local_table, joint_table, use_cache=False):
            relaxed_edges += 1
            nxt_state_code = _state_to_code(step.dst)
            nxt_cost = cur_cost + step.delta_cost
            reachable_tails.add(step.dst.tail)
            if nxt_cost < dist.get(nxt_state_code, inf):
                dist[nxt_state_code] = nxt_cost
                p3_successes += 1
                predecessor[nxt_state_code] = P3PredecessorRecord(
                    src_code=cur_state_code,
                    dst_code=nxt_state_code,
                    native_id=step.native_ids[0] if step.native_ids else "",
                    applied_rule_name=step.applied_rule_name or step.rule_name,
                    delta_cost=step.delta_cost,
                )
                heapq.heappush(heap, (nxt_cost, counter, nxt_state_code))
                counter += 1
                heap_pushes += 1
    dijkstra_seconds = perf_counter() - dijkstra_started

    is_exhaustive = requested_tails is None or not terminated_early
    if is_exhaustive:
        best_accepting_state_by_tail = {}
        best_cost_by_tail = {}
        for state_code, cost in dist.items():
            state = _state_from_code(state_code)
            if state.hidden not in accepting_set:
                continue
            current_best_cost = best_cost_by_tail.get(state.tail)
            if current_best_cost is None or cost < current_best_cost:
                best_cost_by_tail[state.tail] = int(cost)
                best_accepting_state_by_tail[state.tail] = state_code

    closure = FrameClosure(
        requested_tails=requested_tails,
        is_exhaustive=is_exhaustive,
        accepting_hidden_classes=accepting,
        p0_source_count=len(p0_sources),
        p2_source_count=len(p2_sources),
        dist=dist,
        seed_step=seed_step,
        predecessor=predecessor,
        reachable_tails=frozenset(reachable_tails),
        best_accepting_state_by_tail=best_accepting_state_by_tail,
        best_cost_by_tail=best_cost_by_tail,
        p0_source_hit_count_by_tail=p0_source_hit_count_by_tail,
        p2_source_hit_count_by_tail=p2_source_hit_count_by_tail,
        profile=FrameClosureProfile(
            precompute_seconds=perf_counter() - started,
            dijkstra_seconds=dijkstra_seconds,
            p0_source_count=len(p0_sources),
            p2_source_count=len(p2_sources),
            source_state_count=len(seed_step),
            reachable_state_count=len(dist),
            accepting_tail_count=len(best_accepting_state_by_tail),
            popped_states=popped_states,
            relaxed_edges=relaxed_edges,
            p3_attempts=relaxed_edges,
            p3_successes=p3_successes,
            heap_pushes=heap_pushes,
        ),
    )
    _FRAME_CLOSURE_BUILD_COUNT += 1
    _FRAME_CLOSURE_BUILD_SECONDS_TOTAL += closure.profile.precompute_seconds
    _FRAME_CLOSURE_CACHE.setdefault(base_key, []).append(closure)
    return closure


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


def _reconstruct_plan_from_closure(
    *,
    goal_code: int,
    target_tail: Tail,
    closure: FrameClosure,
    local_table: LocalTable,
    joint_table: JointTable,
) -> SynthesisPlan:
    entry_by_id: Dict[str, NativeEntry] = {entry.native_id: entry for entry in local_table.entries}
    p3_reverse_path: List[Step] = []
    cursor_code = goal_code
    while cursor_code in closure.predecessor:
        pred = closure.predecessor[cursor_code]
        src_state = _state_from_code(pred.src_code)
        dst_state = _state_from_code(pred.dst_code)
        axis = entry_by_id.get(pred.native_id)
        if axis is None:
            raise ValueError(f"closure predecessor references missing native_id={pred.native_id!r}")

        matched_step: Step | None = None
        for candidate in apply_p3_transitions(src_state, axis, joint_table):
            candidate_rule = candidate.applied_rule_name or candidate.rule_name
            if candidate_rule != pred.applied_rule_name:
                continue
            if candidate.dst != dst_state:
                continue
            if candidate.delta_cost != pred.delta_cost:
                continue
            matched_step = candidate
            break
        if matched_step is None:
            raise ValueError(
                "cannot reconstruct closure predecessor step "
                f"src={src_state} dst={dst_state} rule={pred.applied_rule_name!r} native_id={pred.native_id!r}"
            )

        p3_reverse_path.append(matched_step)
        cursor_code = pred.src_code

    source = closure.seed_step.get(cursor_code)
    if source is None:
        raise ValueError(f"closure source step missing for state_code={cursor_code}")

    full_path = [source] + list(reversed(p3_reverse_path))
    goal_state = _state_from_code(goal_code)
    total_cost = int(closure.dist[goal_code])
    remote_count = sum(1 for step in full_path if step.scope == SCOPE_INTER_MODULE_BELL)
    return SynthesisPlan(
        target_tail=target_tail,
        best_state=goal_state,
        best_hidden=goal_state.hidden,
        total_cost=total_cost,
        projected_best_cost=total_cost,
        p0_source_count=closure.p0_source_count,
        p2_source_count=closure.p2_source_count,
        remote_inter_module_count=remote_count,
        steps=full_path,
    )


def _run_search_online(
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


def _run_search_from_closure(
    *,
    target_tail: Tail,
    closure: FrameClosure,
    local_table: LocalTable,
    joint_table: JointTable,
    closure_cache_hit: bool,
) -> SynthesisSearchResult:
    global _FRAME_CLOSURE_LOOKUP_COUNT
    global _FRAME_CLOSURE_LOOKUP_SECONDS_TOTAL
    global _FRAME_CLOSURE_WITNESS_COUNT
    global _FRAME_CLOSURE_WITNESS_SECONDS_TOTAL
    lookup_started = perf_counter()
    if not _closure_covers_tail(closure, target_tail):
        raise ValueError(f"frame closure does not cover target tail {target_tail.compact()}")
    p0_source_hit_count = closure.p0_source_hit_count_by_tail.get(target_tail, 0)
    p2_source_hit_count = closure.p2_source_hit_count_by_tail.get(target_tail, 0)
    in_p0_source = p0_source_hit_count > 0
    in_p2_source = p2_source_hit_count > 0
    any_same_tail = target_tail in closure.reachable_tails

    goal_code = closure.best_accepting_state_by_tail.get(target_tail)
    if goal_code is None:
        lookup_seconds = perf_counter() - lookup_started
        _FRAME_CLOSURE_LOOKUP_COUNT += 1
        _FRAME_CLOSURE_LOOKUP_SECONDS_TOTAL += lookup_seconds
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
            popped_states=closure.profile.popped_states,
            closure_build_seconds=closure.profile.precompute_seconds,
            closure_lookup_seconds=lookup_seconds,
            closure_witness_seconds=0.0,
            closure_cache_hit=closure_cache_hit,
        )

    lookup_seconds = perf_counter() - lookup_started
    _FRAME_CLOSURE_LOOKUP_COUNT += 1
    _FRAME_CLOSURE_LOOKUP_SECONDS_TOTAL += lookup_seconds
    witness_started = perf_counter()
    plan = _reconstruct_plan_from_closure(
        goal_code=goal_code,
        target_tail=target_tail,
        closure=closure,
        local_table=local_table,
        joint_table=joint_table,
    )
    witness_seconds = perf_counter() - witness_started
    _FRAME_CLOSURE_WITNESS_COUNT += 1
    _FRAME_CLOSURE_WITNESS_SECONDS_TOTAL += witness_seconds
    return SynthesisSearchResult(
        status=SearchStatus.FOUND_OPTIMAL,
        plan=plan,
        search_truncated=False,
        in_p0_source=in_p0_source,
        in_p2_source=in_p2_source,
        p0_source_hit_count=p0_source_hit_count,
        p2_source_hit_count=p2_source_hit_count,
        any_state_with_same_tail_in_dist=any_same_tail,
        any_accepting_state_with_same_tail_in_dist=True,
        projected_best_cost_exists=True,
        projected_best_cost=closure.best_cost_by_tail.get(target_tail),
        popped_states=closure.profile.popped_states,
        closure_build_seconds=closure.profile.precompute_seconds,
        closure_lookup_seconds=lookup_seconds,
        closure_witness_seconds=witness_seconds,
        closure_cache_hit=closure_cache_hit,
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
    accepting = _normalize_accepting_hidden_classes(accepting_hidden_classes)
    if max_popped_states is None:
        closure = _find_cached_closure_for_target(
            local_table=local_table,
            joint_table=joint_table,
            accepting_hidden_classes=accepting,
            enable_p2=enable_p2,
            p2_use_cache=p2_use_cache,
            target_tail=target_tail,
        )
        closure_cache_hit = closure is not None
        if closure is None:
            closure = precompute_frame_closure(
                local_table,
                joint_table,
                accepting_hidden_classes=accepting,
                enable_p2=enable_p2,
                p2_use_cache=p2_use_cache,
            )
        result = _run_search_from_closure(
            target_tail=target_tail,
            closure=closure,
            local_table=local_table,
            joint_table=joint_table,
            closure_cache_hit=closure_cache_hit,
        )
        assert_no_budget_status_invariant(
            status=result.status,
            max_popped_states=max_popped_states,
            context=f"target={target_tail.compact()}",
        )
        return result
    return _run_search_online(
        target_tail=target_tail,
        local_table=local_table,
        joint_table=joint_table,
        accepting_hidden_classes=accepting,
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
    search_result = synthesize_search(
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

    search_result = synthesize_search(
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

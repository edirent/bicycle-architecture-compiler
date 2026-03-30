from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .state import HEADS, NON_IDENTITY_HEADS, State, Tail, hidden_from_head
from .tables import (
    JointTable,
    LocalTable,
    NativeEntry,
    P2_RELATION_ANTICOMMUTE,
    P2_RELATION_ANY,
    P2_RELATION_COMMUTE,
    P2_TAIL_OP_PRODUCT,
    P2Rule,
    SCOPE_CLOSURE_TRANSITION,
    SCOPE_CROSS_LOGICAL_BLOCK_NATIVE,
    SCOPE_INTER_MODULE_BELL,
    SCOPE_INTRA_BLOCK_NATIVE,
)

DEFAULT_BELL_PENALTY = 6
DEFAULT_CROSS_NATIVE_SHIFT = 0

_P0_QUERY_CACHE: Dict[Tuple[int, int], Tuple["Step", ...]] = {}
_P2_QUERY_CACHE: Dict[Tuple[int, int, int, int], Tuple["Step", ...]] = {}
_P3_EDGE_CACHE: Dict[Tuple[int, int, int, int, State], Tuple["Step", ...]] = {}


@dataclass(frozen=True)
class Step:
    family: str
    rule_name: str
    src: Optional[State]
    dst: State
    delta_cost: int
    native_ids: Tuple[str, ...]
    template: str
    scope: str
    support: Tuple[str, ...]
    metadata: Tuple[Tuple[str, str], ...]
    input_native_ids: Tuple[str, ...] = ()
    input_scopes: Tuple[str, ...] = ()
    applied_rule_name: str = ""


def _metadata_dict(metadata: Tuple[Tuple[str, str], ...]) -> Dict[str, str]:
    return {key: value for key, value in metadata}


def _metadata_int(metadata: Tuple[Tuple[str, str], ...], key: str, default_value: int) -> int:
    raw = _metadata_dict(metadata).get(key)
    if raw is None:
        return default_value
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid integer metadata {key}={raw!r}") from exc


def _p0_allowed_scope(scope: str) -> bool:
    return scope in (SCOPE_INTRA_BLOCK_NATIVE, SCOPE_CROSS_LOGICAL_BLOCK_NATIVE)


def _p0_cost(entry: NativeEntry) -> int:
    base = 1 if entry.head == "I" else 2
    if entry.scope == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE:
        shift = _metadata_int(entry.metadata, "native_shift_cost", DEFAULT_CROSS_NATIVE_SHIFT)
        return base + shift
    return base


def _p3_cost(base_cost: int, axis: NativeEntry) -> int:
    if axis.scope == SCOPE_INTER_MODULE_BELL:
        penalty = _metadata_int(axis.metadata, "bell_penalty", DEFAULT_BELL_PENALTY)
        return base_cost + penalty
    if axis.scope == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE:
        shift = _metadata_int(axis.metadata, "native_shift_cost", DEFAULT_CROSS_NATIVE_SHIFT)
        return base_cost + shift
    return base_cost


def _relation_class(lhs: Tail, rhs: Tail) -> str:
    return P2_RELATION_COMMUTE if lhs.commute(rhs) else P2_RELATION_ANTICOMMUTE


def _tail_op(lhs: Tail, rhs: Tail, op: str) -> Tail:
    if op == P2_TAIL_OP_PRODUCT:
        return lhs.multiply(rhs)
    raise ValueError(f"unsupported P2 tail_op: {op!r}")


def _relation_ok(required_relation: str, relation: str) -> bool:
    if required_relation == P2_RELATION_ANY:
        return True
    return required_relation == relation


def _rule_accepts_entries(rule: P2Rule, lhs: NativeEntry, rhs: NativeEntry, relation: str) -> bool:
    if lhs.scope not in rule.allowed_input_scopes:
        return False
    if rhs.scope not in rule.allowed_input_scopes:
        return False
    if rule.lhs_head is not None and lhs.head != rule.lhs_head:
        return False
    if rule.rhs_head is not None and rhs.head != rule.rhs_head:
        return False
    if rule.require_same_head and lhs.head != rhs.head:
        return False
    if rule.require_nonidentity_heads:
        if lhs.head not in NON_IDENTITY_HEADS or rhs.head not in NON_IDENTITY_HEADS:
            return False
    if not _relation_ok(rule.required_relation, relation):
        return False
    if hidden_from_head(lhs.head) != rule.out_hidden:
        return False
    return True


def _p2_cache_key(local_table: LocalTable, joint_table: JointTable) -> Tuple[int, int, int, int]:
    return (
        id(local_table),
        len(local_table.entries),
        id(joint_table),
        len(joint_table.p2_rules),
    )


def _build_p2_step(rule: P2Rule, lhs: NativeEntry, rhs: NativeEntry, relation: str) -> Step:
    out_scope = (
        SCOPE_CROSS_LOGICAL_BLOCK_NATIVE
        if SCOPE_CROSS_LOGICAL_BLOCK_NATIVE in (lhs.scope, rhs.scope)
        else SCOPE_INTRA_BLOCK_NATIVE
    )
    out_tail = _tail_op(lhs.tail, rhs.tail, rule.tail_op)
    return Step(
        family="P2",
        rule_name=rule.name,
        src=None,
        dst=State(hidden=rule.out_hidden, tail=out_tail),
        delta_cost=rule.total_cost,
        native_ids=(lhs.native_id, rhs.native_id),
        template=rule.template,
        scope=out_scope,
        support=tuple(sorted(set(lhs.support + rhs.support))),
        metadata=(
            ("lhs_scope", lhs.scope),
            ("rhs_scope", rhs.scope),
            ("relation_class", relation),
            ("tail_op", rule.tail_op),
            ("total_cost", str(rule.total_cost)),
        ),
        input_native_ids=(lhs.native_id, rhs.native_id),
        input_scopes=(lhs.scope, rhs.scope),
        applied_rule_name=rule.name,
    )


def _entries_for_rule(local_table: LocalTable, head: str | None, allowed_scopes: Tuple[str, ...]) -> List[NativeEntry]:
    if head is None:
        entries = local_table.entries
    else:
        entries = local_table.by_head().get(head, [])
    return [entry for entry in entries if entry.scope in allowed_scopes]


def _generate_p2_all(local_table: LocalTable, joint_table: JointTable) -> List[Step]:
    out: List[Step] = []
    seen: set[Tuple[str, str, str, str]] = set()

    by_head = local_table.by_head()

    for rule in joint_table.p2_rules:
        lhs_heads = [rule.lhs_head] if rule.lhs_head is not None else list(HEADS)
        rhs_heads = [rule.rhs_head] if rule.rhs_head is not None else list(HEADS)

        if rule.require_same_head:
            shared_heads = sorted(set(lhs_heads).intersection(rhs_heads))
            for head in shared_heads:
                bucket = [
                    entry
                    for entry in by_head.get(head, [])
                    if entry.scope in rule.allowed_input_scopes
                ]
                for idx in range(len(bucket)):
                    lhs = bucket[idx]
                    for jdx in range(idx + 1, len(bucket)):
                        rhs = bucket[jdx]
                        relation = _relation_class(lhs.tail, rhs.tail)
                        if not _rule_accepts_entries(rule, lhs, rhs, relation):
                            continue
                        step = _build_p2_step(rule, lhs, rhs, relation)
                        key = (
                            rule.name,
                            min(lhs.native_id, rhs.native_id),
                            max(lhs.native_id, rhs.native_id),
                            step.dst.tail.compact(),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(step)
            continue

        for lhs_head in lhs_heads:
            lhs_bucket = [
                entry
                for entry in by_head.get(lhs_head, [])
                if entry.scope in rule.allowed_input_scopes
            ]
            if not lhs_bucket:
                continue
            for rhs_head in rhs_heads:
                rhs_bucket = [
                    entry
                    for entry in by_head.get(rhs_head, [])
                    if entry.scope in rule.allowed_input_scopes
                ]
                if not rhs_bucket:
                    continue
                for lhs in lhs_bucket:
                    for rhs in rhs_bucket:
                        if lhs.native_id == rhs.native_id:
                            continue
                        relation = _relation_class(lhs.tail, rhs.tail)
                        if not _rule_accepts_entries(rule, lhs, rhs, relation):
                            continue
                        step = _build_p2_step(rule, lhs, rhs, relation)
                        key = (
                            rule.name,
                            min(lhs.native_id, rhs.native_id),
                            max(lhs.native_id, rhs.native_id),
                            step.dst.tail.compact(),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(step)

    return out


def generate_p0_sources(local_table: LocalTable) -> List[Step]:
    """
    P0 seed rule.
    - (I, P)   -> source State(I, P), cost 1
    - (Q, P), Q in {X,Y,Z} -> source State(Q, P), cost 2

    P0 is restricted to direct frame-native scopes:
      intra_block_native, cross_logical_block_native.
    """
    cache_key = (id(local_table), len(local_table.entries))
    if cache_key in _P0_QUERY_CACHE:
        return list(_P0_QUERY_CACHE[cache_key])

    steps: List[Step] = []
    for entry in local_table.entries:
        if not _p0_allowed_scope(entry.scope):
            continue
        seed_cost = _p0_cost(entry)
        steps.append(
            Step(
                family="P0",
                rule_name="P0_NATIVE_SEED",
                src=None,
                dst=State(hidden=hidden_from_head(entry.head), tail=entry.tail),
                delta_cost=seed_cost,
                native_ids=(entry.native_id,),
                template="(I,P) cost=1; (Q,P), Q in {X,Y,Z}, cost=2",
                scope=entry.scope,
                support=entry.support,
                metadata=entry.metadata,
                input_native_ids=(entry.native_id,),
                input_scopes=(entry.scope,),
                applied_rule_name="P0_NATIVE_SEED",
            )
        )
    _P0_QUERY_CACHE[cache_key] = tuple(steps)
    return steps


def generate_p2_sources(
    local_table: LocalTable,
    joint_table: JointTable,
    target_tail: Tail | None = None,
    use_cache: bool = True,
) -> List[Step]:
    """
    P2 direct composition source rule.
    Build source candidates from pairs of native measurements under JointTable protocol rules.

    - P2 is dynamic per frame inventory (no pre-baked target lookup table).
    - rule.total_cost is interpreted as full source cost (not edge delta from operand states).
    - inter_module_bell entries are excluded by rule.allowed_input_scopes validation.
    - target_tail is only a post-filter over the full P2 source set. It must not
      change which source states exist in the search graph.
    """
    cache_key = _p2_cache_key(local_table, joint_table)
    if use_cache and cache_key in _P2_QUERY_CACHE:
        all_steps = list(_P2_QUERY_CACHE[cache_key])
        if target_tail is None:
            return all_steps
        return [step for step in all_steps if step.dst.tail == target_tail]

    steps = _generate_p2_all(local_table, joint_table)

    # Keep the best total-cost step per dst state.
    best: Dict[State, Step] = {}
    for step in steps:
        prev = best.get(step.dst)
        if prev is None or step.delta_cost < prev.delta_cost:
            best[step.dst] = step
    deduped = list(best.values())

    if use_cache:
        _P2_QUERY_CACHE[cache_key] = tuple(deduped)
    if target_tail is None:
        return deduped
    return [step for step in deduped if step.dst.tail == target_tail]


def apply_p3_transitions(state: State, axis: NativeEntry, joint_table: JointTable) -> List[Step]:
    """
    P3 closure transition rule.
    If current tail anticommutes with axis tail, protocol table decides:
      (cur_hidden, axis_head) -> (next_hidden, delta_cost)
    and emits dst state (next_hidden, state.tail * axis.tail).
    """
    out: List[Step] = []
    is_anticommuting = state.tail.anticommute(axis.tail)

    for rule in joint_table.p3_rules:
        if rule.cur_hidden != state.hidden:
            continue
        if rule.axis_head != axis.head:
            continue
        if rule.anticommute_required and not is_anticommuting:
            continue

        out.append(
            Step(
                family="P3",
                rule_name=rule.name,
                src=state,
                dst=State(hidden=rule.next_hidden, tail=state.tail.multiply(axis.tail)),
                delta_cost=_p3_cost(rule.delta_cost, axis),
                native_ids=(axis.native_id,),
                template=rule.template,
                scope=axis.scope if axis.scope else SCOPE_CLOSURE_TRANSITION,
                support=axis.support,
                metadata=axis.metadata,
                input_native_ids=(axis.native_id,),
                input_scopes=(axis.scope,),
                applied_rule_name=rule.name,
            )
        )

    return out


def expand_p3_edges(state: State, local_table: LocalTable, joint_table: JointTable) -> List[Step]:
    cache_key = (
        id(local_table),
        len(local_table.entries),
        id(joint_table),
        len(joint_table.p3_rules),
        state,
    )
    if cache_key in _P3_EDGE_CACHE:
        return list(_P3_EDGE_CACHE[cache_key])

    transitions: List[Step] = []
    for axis in local_table.native_axes():
        transitions.extend(apply_p3_transitions(state, axis, joint_table))
    _P3_EDGE_CACHE[cache_key] = tuple(transitions)
    return transitions

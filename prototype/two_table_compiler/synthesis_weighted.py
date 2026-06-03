from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numba as nb
import numpy as np
from numba import types
from numba.typed import List

from .protocol_costs import (
    DIGIT_TO_HEAD,
    HEAD_TO_DIGIT,
    P2_DIRECT_SOURCE_COST,
    derive_p0_source_cost,
    derive_p2_source_cost,
    derive_p3_transition,
    validate_joint_table_matches_paper_skeleton,
)
from .state import N_QUBITS, Tail
from .tables import JointTable, LocalTable, NativeEntry

UNREACHED_COST = np.uint32(0xFFFFFFFF)
TAIL_SPACE_SIZE = 1 << (2 * N_QUBITS)
TAIL_MASK = TAIL_SPACE_SIZE - 1
STATE_SHIFT = 2 * N_QUBITS
STATE_SPACE_SIZE = 4 * TAIL_SPACE_SIZE
PLANE_MASK = (1 << N_QUBITS) - 1
BUCKET_COUNT = 7

WEIGHTED_TRANSITION_COST_TABLE = np.array(
    [
        [0, 4, 4, 4],
        [0, 6, 6, 6],
        [0, 6, 6, 6],
        [0, 6, 6, 6],
    ],
    dtype=np.uint32,
)
UNIT_EDGE_TRANSITION_COST_TABLE = np.array(
    [
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
    ],
    dtype=np.uint32,
)
NEXT_HEAD_TABLE = np.array(
    [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ],
    dtype=np.uint8,
)

DIGIT_TO_SYMBOL = ("I", "X", "Z", "Y")
SYMBOL_TO_XZ = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}
PREFIX_WIDTH = 5
SUFFIX_WIDTH = N_QUBITS - PREFIX_WIDTH
PREFIX_MASK = (1 << PREFIX_WIDTH) - 1
SUFFIX_MASK = (1 << SUFFIX_WIDTH) - 1
TAIL_PREFIXES = tuple()
TAIL_SUFFIXES = tuple()


def _segment_to_compact(segment_code: int, width: int) -> str:
    x_mask = segment_code & ((1 << width) - 1)
    z_mask = segment_code >> width
    chars = []
    for qubit in range(width):
        digit = ((z_mask >> qubit) & 1) << 1 | ((x_mask >> qubit) & 1)
        chars.append(DIGIT_TO_SYMBOL[digit])
    return "".join(chars)


TAIL_PREFIXES = tuple(_segment_to_compact(code, PREFIX_WIDTH) for code in range(1 << (2 * PREFIX_WIDTH)))
TAIL_SUFFIXES = tuple(_segment_to_compact(code, SUFFIX_WIDTH) for code in range(1 << (2 * SUFFIX_WIDTH)))


@dataclass(frozen=True)
class SourceWitness:
    family: str
    dst_head: str
    dst_tail_code: int
    cost: int
    native_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class TargetTrace:
    tail_code: int
    tail_pauli: str
    best_head: str | None
    beta_cost: int | None
    head_costs: dict[str, int | None]
    path: list[dict[str, Any]]


def tail_code_from_tail(tail: Tail) -> int:
    x_mask = 0
    z_mask = 0
    for qubit, symbol in enumerate(tail.paulis):
        x_bit, z_bit = SYMBOL_TO_XZ[symbol]
        if x_bit:
            x_mask |= 1 << qubit
        if z_bit:
            z_mask |= 1 << qubit
    return x_mask | (z_mask << N_QUBITS)


def tail_compact_from_code(code: int) -> str:
    x_mask = code & PLANE_MASK
    z_mask = code >> N_QUBITS
    prefix_code = (x_mask & PREFIX_MASK) | ((z_mask & PREFIX_MASK) << PREFIX_WIDTH)
    suffix_code = ((x_mask >> PREFIX_WIDTH) & SUFFIX_MASK) | (
        ((z_mask >> PREFIX_WIDTH) & SUFFIX_MASK) << SUFFIX_WIDTH
    )
    return TAIL_PREFIXES[prefix_code] + TAIL_SUFFIXES[suffix_code]


def state_code(head: str, tail_code: int) -> int:
    return (HEAD_TO_DIGIT[head] << STATE_SHIFT) | tail_code


def decode_state_code(code: int) -> tuple[str, int]:
    head_digit = code >> STATE_SHIFT
    tail_code = code & TAIL_MASK
    return DIGIT_TO_HEAD[head_digit], tail_code


def _same_basis_p2_rule_by_head(joint_table: JointTable) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    for rule in joint_table.p2_rules:
        if rule.lhs_head != rule.rhs_head or rule.lhs_head not in ("X", "Y", "Z"):
            continue
        if not rule.require_same_head or rule.required_relation != "commute":
            continue
        rules[rule.lhs_head] = rule
    return rules


def build_weighted_sources(local_table: LocalTable, joint_table: JointTable) -> dict[str, Any]:
    validate_joint_table_matches_paper_skeleton(joint_table)
    p0_witnesses: list[SourceWitness] = []
    p2_witnesses: list[SourceWitness] = []

    for entry in local_table.entries:
        p0_witnesses.append(
            SourceWitness(
                family="P0",
                dst_head=entry.head,
                dst_tail_code=tail_code_from_tail(entry.tail),
                cost=derive_p0_source_cost(entry),
                native_ids=(entry.native_id,),
                detail=f"direct native {entry.head}:{entry.tail.compact()}",
            )
        )

    same_basis_rules = _same_basis_p2_rule_by_head(joint_table)
    by_head = local_table.by_head()
    for head in ("X", "Y", "Z"):
        rule = same_basis_rules[head]
        bucket = list(by_head.get(head, ()))
        for idx in range(len(bucket)):
            lhs = bucket[idx]
            for jdx in range(idx + 1, len(bucket)):
                rhs = bucket[jdx]
                if not lhs.tail.commute(rhs.tail):
                    continue
                cost = derive_p2_source_cost(lhs, rhs, rule)
                out_tail_code = tail_code_from_tail(lhs.tail.multiply(rhs.tail))
                p2_witnesses.append(
                    SourceWitness(
                        family="P2",
                        dst_head="I",
                        dst_tail_code=out_tail_code,
                        cost=cost,
                        native_ids=(lhs.native_id, rhs.native_id),
                        detail=f"same-basis {head} pair {lhs.tail.compact()} xor {rhs.tail.compact()}",
                    )
                )

    deduped_by_state: dict[int, SourceWitness] = {}
    for witness in p0_witnesses + p2_witnesses:
        encoded = state_code(witness.dst_head, witness.dst_tail_code)
        incumbent = deduped_by_state.get(encoded)
        if incumbent is None or witness.cost < incumbent.cost:
            deduped_by_state[encoded] = witness

    ordered_sources = sorted(deduped_by_state.items())
    source_states = np.fromiter((encoded for encoded, _ in ordered_sources), dtype=np.uint32, count=len(ordered_sources))
    source_costs = np.fromiter((witness.cost for _, witness in ordered_sources), dtype=np.uint32, count=len(ordered_sources))

    return {
        "source_states": source_states,
        "source_costs": source_costs,
        "deduped_witness_by_state": {int(encoded): witness for encoded, witness in ordered_sources},
        "p0_source_count_raw": len(p0_witnesses),
        "p2_source_count_raw": len(p2_witnesses),
        "p0_source_count": sum(1 for witness in deduped_by_state.values() if witness.family == "P0"),
        "p2_source_count": sum(1 for witness in deduped_by_state.values() if witness.family == "P2"),
        "sample_p2_source": None if not p2_witnesses else p2_witnesses[0],
    }


def build_unit_edge_source_costs(weighted_source_costs: np.ndarray) -> np.ndarray:
    unit_costs = np.ones_like(weighted_source_costs, dtype=np.uint32)
    _ = weighted_source_costs
    return unit_costs


def build_weighted_axes(local_table: LocalTable) -> dict[str, Any]:
    axis_rows: list[tuple[int, int, int, int, str, str]] = []
    for entry in local_table.entries:
        if entry.head not in ("X", "Y", "Z"):
            continue
        axis_code = tail_code_from_tail(entry.tail)
        transition = derive_p3_transition("I", entry.head)
        _ = transition
        axis_rows.append(
            (
                HEAD_TO_DIGIT[entry.head],
                axis_code,
                axis_code & PLANE_MASK,
                axis_code >> N_QUBITS,
                entry.native_id,
                entry.tail.compact(),
            )
        )

    axis_rows.sort(key=lambda row: (row[0], row[1], row[4]))
    return {
        "axis_head_digits": np.fromiter((row[0] for row in axis_rows), dtype=np.uint8, count=len(axis_rows)),
        "axis_codes": np.fromiter((row[1] for row in axis_rows), dtype=np.uint32, count=len(axis_rows)),
        "axis_xs": np.fromiter((row[2] for row in axis_rows), dtype=np.uint32, count=len(axis_rows)),
        "axis_zs": np.fromiter((row[3] for row in axis_rows), dtype=np.uint32, count=len(axis_rows)),
        "axis_debug": [
            {"head": DIGIT_TO_HEAD[row[0]], "tail_pauli": row[5], "native_id": row[4]}
            for row in axis_rows
        ],
    }


def build_target_mask(*, limit_tails: int | None = None, max_weight: int | None = None) -> tuple[np.ndarray, int]:
    mask = np.zeros(TAIL_SPACE_SIZE, dtype=np.uint8)
    if limit_tails is not None:
        capped = min(limit_tails, TAIL_SPACE_SIZE)
        mask[:capped] = 1
        return mask, int(capped)

    count = 0
    for tail_code in range(TAIL_SPACE_SIZE):
        x_mask = tail_code & PLANE_MASK
        z_mask = tail_code >> N_QUBITS
        weight = (x_mask | z_mask).bit_count()
        if max_weight is not None and weight > max_weight:
            continue
        mask[tail_code] = 1
        count += 1
    return mask, count


def _new_bucket_ring() -> List:
    buckets = List()
    for _ in range(BUCKET_COUNT):
        buckets.append(List.empty_list(types.uint32))
    return buckets


@nb.njit(cache=True)
def _parity32(value: np.uint32) -> np.uint32:
    x = value
    x ^= x >> np.uint32(16)
    x ^= x >> np.uint32(8)
    x ^= x >> np.uint32(4)
    x &= np.uint32(0xF)
    return np.uint32((0x6996 >> x) & 1)


@nb.njit(cache=True)
def _solve_beta_table_numba(
    source_states: np.ndarray,
    source_costs: np.ndarray,
    axis_head_digits: np.ndarray,
    axis_codes: np.ndarray,
    axis_xs: np.ndarray,
    axis_zs: np.ndarray,
    transition_cost_table: np.ndarray,
    target_mask: np.ndarray,
    target_count: int,
    progress_every_popped: int,
    buckets: List,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int, int, int]:
    dist = np.full(STATE_SPACE_SIZE, UNREACHED_COST, dtype=np.uint32)
    best_cost_by_tail = np.full(TAIL_SPACE_SIZE, UNREACHED_COST, dtype=np.uint32)
    best_head_by_tail = np.zeros(TAIL_SPACE_SIZE, dtype=np.uint8)

    queued_states = 0
    queue_pushes = 0
    source_state_count = 0
    reachable_state_count = 0

    for idx in range(source_states.shape[0]):
        state_idx = source_states[idx]
        cost = source_costs[idx]
        if dist[state_idx] == UNREACHED_COST:
            source_state_count += 1
            reachable_state_count += 1
        if cost < dist[state_idx]:
            dist[state_idx] = cost
            buckets[cost % BUCKET_COUNT].append(state_idx)
            queued_states += 1
            queue_pushes += 1

    popped_states = 0
    relaxed_edges = 0
    p3_successes = 0
    reached_target_count = 0
    current_cost = 0

    while queued_states > 0 and reached_target_count < target_count:
        bucket_idx = current_cost % BUCKET_COUNT
        while len(buckets[bucket_idx]) == 0:
            current_cost += 1
            bucket_idx = current_cost % BUCKET_COUNT

        cur_state_idx = buckets[bucket_idx].pop()
        queued_states -= 1
        if dist[cur_state_idx] != current_cost:
            continue

        popped_states += 1
        cur_head_digit = cur_state_idx >> STATE_SHIFT
        cur_tail_idx = cur_state_idx & TAIL_MASK

        if cur_head_digit != 0 and target_mask[cur_tail_idx] != 0 and best_cost_by_tail[cur_tail_idx] == UNREACHED_COST:
            best_cost_by_tail[cur_tail_idx] = np.uint32(current_cost)
            best_head_by_tail[cur_tail_idx] = np.uint8(cur_head_digit)
            reached_target_count += 1
            if reached_target_count >= target_count:
                break

        cur_x = np.uint32(cur_tail_idx & PLANE_MASK)
        cur_z = np.uint32(cur_tail_idx >> N_QUBITS)
        for axis_idx in range(axis_codes.shape[0]):
            parity = _parity32((cur_x & axis_zs[axis_idx]) ^ (cur_z & axis_xs[axis_idx]))
            if parity == 0:
                continue

            relaxed_edges += 1
            next_head_digit = NEXT_HEAD_TABLE[cur_head_digit, axis_head_digits[axis_idx]]
            next_tail_idx = cur_tail_idx ^ int(axis_codes[axis_idx])
            next_state_idx = (int(next_head_digit) << STATE_SHIFT) | next_tail_idx
            next_cost = np.uint32(current_cost + int(transition_cost_table[cur_head_digit, axis_head_digits[axis_idx]]))
            old_cost = dist[next_state_idx]
            if next_cost < old_cost:
                if old_cost == UNREACHED_COST:
                    reachable_state_count += 1
                dist[next_state_idx] = next_cost
                buckets[next_cost % BUCKET_COUNT].append(np.uint32(next_state_idx))
                queued_states += 1
                queue_pushes += 1
                p3_successes += 1

        if progress_every_popped > 0 and popped_states % progress_every_popped == 0:
            print(
                "[weighted-progress]",
                popped_states,
                reached_target_count,
                target_count,
                reachable_state_count,
                current_cost,
            )

    return (
        best_cost_by_tail,
        best_head_by_tail,
        source_state_count,
        reachable_state_count,
        popped_states,
        relaxed_edges,
        p3_successes,
        queue_pushes,
        reached_target_count,
    )


def solve_beta_table(
    *,
    source_states: np.ndarray,
    source_costs: np.ndarray,
    axes: dict[str, Any],
    transition_cost_table: np.ndarray,
    target_mask: np.ndarray | None = None,
    progress_every_popped: int = 0,
) -> dict[str, Any]:
    mask = np.ones(TAIL_SPACE_SIZE, dtype=np.uint8) if target_mask is None else target_mask
    target_count = int(mask.sum())
    started = perf_counter()
    buckets = _new_bucket_ring()
    (
        best_cost_by_tail,
        best_head_by_tail,
        source_state_count,
        reachable_state_count,
        popped_states,
        relaxed_edges,
        p3_successes,
        queue_pushes,
        reached_target_count,
    ) = _solve_beta_table_numba(
        source_states=source_states,
        source_costs=source_costs,
        axis_head_digits=axes["axis_head_digits"],
        axis_codes=axes["axis_codes"],
        axis_xs=axes["axis_xs"],
        axis_zs=axes["axis_zs"],
        transition_cost_table=transition_cost_table,
        target_mask=mask,
        target_count=target_count,
        progress_every_popped=progress_every_popped,
        buckets=buckets,
    )
    elapsed = perf_counter() - started
    return {
        "best_cost_by_tail": best_cost_by_tail,
        "best_head_by_tail": best_head_by_tail,
        "profile": {
            "tail_count": TAIL_SPACE_SIZE,
            "node_count": STATE_SPACE_SIZE,
            "source_state_count": int(source_state_count),
            "reachable_state_count": int(reachable_state_count),
            "popped_states": int(popped_states),
            "relaxed_edges": int(relaxed_edges),
            "p3_successes": int(p3_successes),
            "queue_pushes": int(queue_pushes),
            "reached_target_count": int(reached_target_count),
            "target_count": int(target_count),
            "solve_seconds": elapsed,
        },
    }


def compute_histogram_and_stats(best_cost_by_tail: np.ndarray) -> dict[str, Any]:
    reachable_mask = best_cost_by_tail != UNREACHED_COST
    reachable_costs = best_cost_by_tail[reachable_mask].astype(np.uint64, copy=False)
    reachable_count = int(reachable_mask.sum())
    unreachable_count = int(best_cost_by_tail.shape[0] - reachable_count)
    if reachable_count == 0:
        return {
            "histogram": {},
            "reachable_tail_count": 0,
            "unreachable_tail_count": unreachable_count,
            "total_tail_space": int(best_cost_by_tail.shape[0]),
            "mean_cost": None,
            "min_cost": None,
            "max_cost": None,
        }

    unique_costs, counts = np.unique(reachable_costs, return_counts=True)
    histogram = {int(cost): int(count) for cost, count in zip(unique_costs, counts)}
    return {
        "histogram": histogram,
        "reachable_tail_count": reachable_count,
        "unreachable_tail_count": unreachable_count,
        "total_tail_space": int(best_cost_by_tail.shape[0]),
        "mean_cost": float(reachable_costs.mean()),
        "min_cost": int(reachable_costs.min()),
        "max_cost": int(reachable_costs.max()),
    }


def write_weighted_cost_table_csv(path: Path, best_cost_by_tail: np.ndarray, best_head_by_tail: np.ndarray) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tail_id", "tail_pauli", "cost", "best_head", "reachable"])
        for tail_code, cost in enumerate(best_cost_by_tail):
            reachable = int(cost != UNREACHED_COST)
            writer.writerow(
                [
                    tail_code,
                    tail_compact_from_code(tail_code),
                    "" if not reachable else int(cost),
                    "" if not reachable else DIGIT_TO_HEAD[int(best_head_by_tail[tail_code])],
                    reachable,
                ]
            )


def write_hist_csv(path: Path, histogram: dict[int, int]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cost_bucket", "count"])
        for cost, count in sorted(histogram.items()):
            writer.writerow([cost, count])


def build_weighted_pipeline(local_table: LocalTable, joint_table: JointTable) -> dict[str, Any]:
    sources = build_weighted_sources(local_table, joint_table)
    axes = build_weighted_axes(local_table)
    return {"sources": sources, "axes": axes}


def run_weighted_exact_build(
    *,
    local_table: LocalTable,
    joint_table: JointTable,
    progress_every_popped: int = 0,
) -> dict[str, Any]:
    pipeline = build_weighted_pipeline(local_table, joint_table)
    solved = solve_beta_table(
        source_states=pipeline["sources"]["source_states"],
        source_costs=pipeline["sources"]["source_costs"],
        axes=pipeline["axes"],
        transition_cost_table=WEIGHTED_TRANSITION_COST_TABLE,
        progress_every_popped=progress_every_popped,
    )
    stats = compute_histogram_and_stats(solved["best_cost_by_tail"])
    profile = {
        **solved["profile"],
        "p0_source_count": pipeline["sources"]["p0_source_count"],
        "p2_source_count": pipeline["sources"]["p2_source_count"],
        "p0_source_count_raw": pipeline["sources"]["p0_source_count_raw"],
        "p2_source_count_raw": pipeline["sources"]["p2_source_count_raw"],
        "axis_count": int(pipeline["axes"]["axis_codes"].shape[0]),
        "reachable_tail_count": stats["reachable_tail_count"],
        "unreachable_tail_count": stats["unreachable_tail_count"],
        "mean_cost": stats["mean_cost"],
        "min_cost": stats["min_cost"],
        "max_cost": stats["max_cost"],
    }
    return {
        "pipeline": pipeline,
        "best_cost_by_tail": solved["best_cost_by_tail"],
        "best_head_by_tail": solved["best_head_by_tail"],
        "histogram": stats["histogram"],
        "profile": profile,
    }


def run_subset_smoke(
    *,
    pipeline: dict[str, Any],
    limit_tails: int = 4096,
    progress_every_popped: int = 0,
) -> dict[str, Any]:
    target_mask, target_count = build_target_mask(limit_tails=limit_tails)
    weighted = solve_beta_table(
        source_states=pipeline["sources"]["source_states"],
        source_costs=pipeline["sources"]["source_costs"],
        axes=pipeline["axes"],
        transition_cost_table=WEIGHTED_TRANSITION_COST_TABLE,
        target_mask=target_mask,
        progress_every_popped=progress_every_popped,
    )
    unit = solve_beta_table(
        source_states=pipeline["sources"]["source_states"],
        source_costs=build_unit_edge_source_costs(pipeline["sources"]["source_costs"]),
        axes=pipeline["axes"],
        transition_cost_table=UNIT_EDGE_TRANSITION_COST_TABLE,
        target_mask=target_mask,
        progress_every_popped=progress_every_popped,
    )
    weighted_costs = weighted["best_cost_by_tail"][:limit_tails]
    unit_costs = unit["best_cost_by_tail"][:limit_tails]
    diff_tail = None
    for tail_code in range(limit_tails):
        if int(weighted_costs[tail_code]) != int(unit_costs[tail_code]):
            diff_tail = tail_code
            break
    return {
        "limit_tails": limit_tails,
        "target_count": target_count,
        "weighted_profile": weighted["profile"],
        "unit_profile": unit["profile"],
        "weighted_costs": weighted_costs,
        "unit_costs": unit_costs,
        "first_difference_tail": diff_tail,
    }


def _state_tail_parity(cur_tail_code: int, axis_code: int) -> bool:
    cur_x = cur_tail_code & PLANE_MASK
    cur_z = cur_tail_code >> N_QUBITS
    axis_x = axis_code & PLANE_MASK
    axis_z = axis_code >> N_QUBITS
    return (((cur_x & axis_z) ^ (cur_z & axis_x)).bit_count() & 1) == 1


def explain_target(
    *,
    tail_code: int,
    pipeline: dict[str, Any],
) -> TargetTrace:
    dist: dict[int, int] = {}
    pred: dict[int, dict[str, Any]] = {}
    heap: list[tuple[int, int]] = []

    witness_by_state = pipeline["sources"]["deduped_witness_by_state"]
    axis_debug = pipeline["axes"]["axis_debug"]
    axis_head_digits = pipeline["axes"]["axis_head_digits"]
    axis_codes = pipeline["axes"]["axis_codes"]

    for state_idx, witness in witness_by_state.items():
        if witness.cost < dist.get(state_idx, 1 << 60):
            dist[state_idx] = witness.cost
            pred[state_idx] = {
                "family": witness.family,
                "cost": witness.cost,
                "native_ids": witness.native_ids,
                "detail": witness.detail,
            }
            heapq.heappush(heap, (witness.cost, state_idx))

    head_costs: dict[str, int | None] = {"X": None, "Y": None, "Z": None}
    best_target: tuple[int, int] | None = None
    while heap:
        cur_cost, state_idx = heapq.heappop(heap)
        if cur_cost != dist.get(state_idx):
            continue

        cur_head, cur_tail_code = decode_state_code(state_idx)
        if cur_tail_code == tail_code and cur_head in head_costs and head_costs[cur_head] is None:
            head_costs[cur_head] = cur_cost
            if best_target is None or cur_cost < best_target[1]:
                best_target = (state_idx, cur_cost)
            if all(value is not None for value in head_costs.values()):
                break

        cur_head_digit = HEAD_TO_DIGIT[cur_head]
        for axis_idx, axis_head_digit in enumerate(axis_head_digits):
            axis_code = int(axis_codes[axis_idx])
            if not _state_tail_parity(cur_tail_code, axis_code):
                continue
            next_head_digit = int(NEXT_HEAD_TABLE[cur_head_digit, axis_head_digit])
            delta_cost = int(WEIGHTED_TRANSITION_COST_TABLE[cur_head_digit, axis_head_digit])
            next_tail_code = cur_tail_code ^ axis_code
            next_state_idx = (next_head_digit << STATE_SHIFT) | next_tail_code
            next_cost = cur_cost + delta_cost
            if next_cost >= dist.get(next_state_idx, 1 << 60):
                continue
            dist[next_state_idx] = next_cost
            pred[next_state_idx] = {
                "family": "P3",
                "cost": delta_cost,
                "axis_head": DIGIT_TO_HEAD[int(axis_head_digit)],
                "axis_tail": axis_debug[axis_idx]["tail_pauli"],
                "native_id": axis_debug[axis_idx]["native_id"],
                "prev_state": state_idx,
            }
            heapq.heappush(heap, (next_cost, next_state_idx))

    if best_target is None:
        return TargetTrace(
            tail_code=tail_code,
            tail_pauli=tail_compact_from_code(tail_code),
            best_head=None,
            beta_cost=None,
            head_costs=head_costs,
            path=[],
        )

    best_state_idx, best_cost = best_target
    path: list[dict[str, Any]] = []
    cursor = best_state_idx
    while True:
        head, cursor_tail_code = decode_state_code(cursor)
        node = pred[cursor]
        if node["family"] in ("P0", "P2"):
            path.append(
                {
                    "family": node["family"],
                    "dst_head": head,
                    "dst_tail": tail_compact_from_code(cursor_tail_code),
                    "delta_cost": node["cost"],
                    "cumulative_cost": dist[cursor],
                    "native_ids": list(node["native_ids"]),
                    "detail": node["detail"],
                }
            )
            break
        prev_state = int(node["prev_state"])
        prev_head, prev_tail_code = decode_state_code(prev_state)
        path.append(
            {
                "family": "P3",
                "src_head": prev_head,
                "src_tail": tail_compact_from_code(prev_tail_code),
                "dst_head": head,
                "dst_tail": tail_compact_from_code(cursor_tail_code),
                "axis_head": node["axis_head"],
                "axis_tail": node["axis_tail"],
                "native_id": node["native_id"],
                "delta_cost": node["cost"],
                "cumulative_cost": dist[cursor],
            }
        )
        cursor = prev_state

    path.reverse()
    best_head, _ = decode_state_code(best_state_idx)
    return TargetTrace(
        tail_code=tail_code,
        tail_pauli=tail_compact_from_code(tail_code),
        best_head=best_head,
        beta_cost=best_cost,
        head_costs=head_costs,
        path=path,
    )


def find_head_difference_example(*, pipeline: dict[str, Any], candidate_limit: int = 512) -> TargetTrace | None:
    for tail_code in range(candidate_limit):
        trace = explain_target(tail_code=tail_code, pipeline=pipeline)
        reachable = [cost for cost in trace.head_costs.values() if cost is not None]
        if len(reachable) < 2:
            continue
        if len(set(reachable)) > 1:
            return trace
    return None


def find_p3_trace_example(*, pipeline: dict[str, Any], candidate_limit: int = 1024) -> TargetTrace | None:
    for tail_code in range(candidate_limit):
        trace = explain_target(tail_code=tail_code, pipeline=pipeline)
        if any(step["family"] == "P3" for step in trace.path):
            return trace
    return None


def summarize_histogram_front(histogram: dict[int, int], limit: int = 10) -> list[tuple[int, int]]:
    return sorted(histogram.items())[:limit]


def sample_p2_source_debug(pipeline: dict[str, Any]) -> dict[str, Any] | None:
    witness = pipeline["sources"]["sample_p2_source"]
    if witness is None:
        return None
    return {
        "family": witness.family,
        "dst_head": witness.dst_head,
        "dst_tail": tail_compact_from_code(witness.dst_tail_code),
        "cost": witness.cost,
        "native_ids": list(witness.native_ids),
        "detail": witness.detail,
    }


def build_toy_debug_samples() -> dict[str, Any]:
    from .examples import build_demo_joint_table
    from .tables import LocalTable, NativeEntry

    frame_id = "weighted_debug_toy"
    local_table_with_direct_y = LocalTable(
        frame_id=frame_id,
        entries=[
            NativeEntry(native_id="I_base", frame_id=frame_id, head="I", tail=Tail.from_str("XIIIIIIIIII")),
            NativeEntry(native_id="X_axis", frame_id=frame_id, head="X", tail=Tail.from_str("ZIIIIIIIIII")),
            NativeEntry(native_id="Y_target", frame_id=frame_id, head="Y", tail=Tail.from_str("YIIIIIIIIII")),
            NativeEntry(native_id="X_left", frame_id=frame_id, head="X", tail=Tail.from_str("XIIIIIIIIII")),
            NativeEntry(native_id="X_right", frame_id=frame_id, head="X", tail=Tail.from_str("IXIIIIIIIII")),
        ],
    )
    local_table_without_direct_y = LocalTable(
        frame_id=f"{frame_id}_p3",
        entries=[
            NativeEntry(native_id="I_base", frame_id=f"{frame_id}_p3", head="I", tail=Tail.from_str("XIIIIIIIIII")),
            NativeEntry(native_id="X_axis", frame_id=f"{frame_id}_p3", head="X", tail=Tail.from_str("ZIIIIIIIIII")),
        ],
    )
    joint_table = build_demo_joint_table()
    head_pipeline = build_weighted_pipeline(local_table_with_direct_y, joint_table)
    p3_pipeline = build_weighted_pipeline(local_table_without_direct_y, joint_table)
    head_trace = explain_target(tail_code=tail_code_from_tail(Tail.from_str("YIIIIIIIIII")), pipeline=head_pipeline)
    p3_trace = explain_target(tail_code=tail_code_from_tail(Tail.from_str("YIIIIIIIIII")), pipeline=p3_pipeline)
    return {
        "head_trace": _trace_to_plain_dict(head_trace),
        "p3_trace": _trace_to_plain_dict(p3_trace),
        "sample_p2_source": sample_p2_source_debug(head_pipeline),
    }


def _trace_to_plain_dict(trace: TargetTrace) -> dict[str, Any]:
    return {
        "tail_code": trace.tail_code,
        "tail_pauli": trace.tail_pauli,
        "best_head": trace.best_head,
        "beta_cost": trace.beta_cost,
        "head_costs": trace.head_costs,
        "path": trace.path,
    }

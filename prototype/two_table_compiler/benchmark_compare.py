from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import build_assets
from .io import DEFAULT_LOCAL_TABLE_PATH, load_joint_table, load_local_table
from .search import (
    SearchStatus,
    SynthesisSearchResult,
    assert_no_budget_status_invariant,
    search_status_flags,
    search_status_name,
    search_status_note,
    synthesize_search,
)
from .state import N_QUBITS, Tail
from .tables import (
    JointTable,
    LocalTable,
    SCOPE_CROSS_LOGICAL_BLOCK_NATIVE,
    SCOPE_INTER_MODULE_BELL,
    SCOPE_INTRA_BLOCK_NATIVE,
)

PAULI_ORDER = ("I", "X", "Y", "Z")
DEFAULT_BASELINE_JSON = Path("results") / "beta_headed_min_11q.json"
DEFAULT_RESULTS_ROOT = Path("benchmark") / "results"


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    seed: int
    count: int
    max_weight: int | None
    include_identity: bool
    max_popped_states: int | None
    disable_p2: bool
    output_dir: Path
    frame_id: str | None
    jobs: int
    progress_every: int
    baseline_json: Path
    local_table_path: Path
    joint_table_path: Path


def _tail_bits_from_compact(compact: str) -> int:
    if len(compact) != N_QUBITS:
        raise ValueError(f"target must have length {N_QUBITS}, got {len(compact)}")
    x_bits = 0
    z_bits = 0
    for idx, symbol in enumerate(compact):
        p = symbol.upper()
        if p not in PAULI_ORDER:
            raise ValueError(f"invalid Pauli symbol in target {compact!r}: {symbol!r}")
        if p in ("X", "Y"):
            x_bits |= 1 << idx
        if p in ("Z", "Y"):
            z_bits |= 1 << idx
    return x_bits | (z_bits << N_QUBITS)


def _weight(compact: str) -> int:
    return sum(ch != "I" for ch in compact)


def _iter_exhaustive_targets(include_identity: bool) -> Iterator[str]:
    for symbols in _cartesian_paulis(N_QUBITS):
        target = "".join(symbols)
        if not include_identity and all(ch == "I" for ch in target):
            continue
        yield target


def _cartesian_paulis(length: int) -> Iterator[Tuple[str, ...]]:
    if length == 0:
        yield ()
        return
    stack = [0] * length
    while True:
        yield tuple(PAULI_ORDER[idx] for idx in stack)
        pos = length - 1
        while pos >= 0:
            stack[pos] += 1
            if stack[pos] < len(PAULI_ORDER):
                break
            stack[pos] = 0
            pos -= 1
        if pos < 0:
            return


def _sample_targets(seed: int, count: int, max_weight: int | None, include_identity: bool) -> List[str]:
    corpus = build_assets.build_random_corpus(
        seed=seed,
        count=count,
        n_qubits=N_QUBITS,
        max_weight_filter=max_weight,
    )
    targets = list(corpus.targets)
    if include_identity and "I" * N_QUBITS not in targets:
        targets = ["I" * N_QUBITS] + targets
    return targets


def _ensure_baseline_json(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        f"baseline file is required but missing: {path}. "
        "This benchmark reuses offline artifacts only and does not invoke cargo."
    )


def _load_baseline_costs(path: Path, *, required_indices: Iterable[int] | None = None) -> List[int] | Dict[int, int]:
    src = _ensure_baseline_json(path)
    needed = None if required_indices is None else {int(index) for index in required_indices}
    found: Dict[int, int] = {}
    costs: List[int] = []
    max_needed = None if not needed else max(needed)
    in_array = False
    index = 0

    with src.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not in_array:
                if line.endswith('"beta_headed_min": [') or line == '"beta_headed_min": [':
                    in_array = True
                continue
            if line in ("]", "],"):
                break
            if line == "":
                continue
            value = int(line.rstrip(","))
            if needed is None:
                costs.append(value)
            elif index in needed:
                found[index] = value
                if len(found) == len(needed) and max_needed is not None and index >= max_needed:
                    break
            index += 1

    if not in_array:
        raise ValueError(f"unexpected baseline JSON schema in {src}")
    if needed is None:
        if len(costs) != 4 ** N_QUBITS:
            raise ValueError(
                f"baseline length mismatch: expected {4 ** N_QUBITS}, got {len(costs)}"
            )
        return costs
    missing = sorted(needed.difference(found))
    if missing:
        raise ValueError(f"baseline indices missing from {src}: first_missing={missing[:10]}")
    return found


def _baseline_cost_at(baseline_costs: Sequence[int] | Mapping[int, int], tail_bits: int) -> int:
    if isinstance(baseline_costs, Mapping):
        return int(baseline_costs[tail_bits])
    return int(baseline_costs[tail_bits])


def _load_tables(cfg: BenchmarkConfig) -> Tuple[LocalTable, JointTable]:
    if cfg.local_table_path.exists() and cfg.joint_table_path.exists():
        local = load_local_table(cfg.local_table_path)
        joint = load_joint_table(cfg.joint_table_path)
    else:
        local, joint, _ = build_assets.ensure_cached_assets(seed=cfg.seed, count=max(cfg.count, 20))

    if cfg.frame_id is not None and local.frame_id != cfg.frame_id:
        raise ValueError(f"frame mismatch: requested {cfg.frame_id!r}, table has {local.frame_id!r}")
    return local, joint


def _gross_inventory_from_csv(csv_path: Path = build_assets.GROSS_NATIVE_SOURCE_CSV) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"gross native CSV not found: {csv_path}. Run export_native_11q first."
        )
    out: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index = int(row["index"])
            head = row["head"].strip().upper()
            tail_bits = int(row["tail_bits"])
            tail = build_assets._decode_tail_bits(tail_bits).compact()
            out.append(
                {
                    "native_id": f"gross_native_{index}",
                    "head": head,
                    "tail": tail,
                    "scope": SCOPE_INTRA_BLOCK_NATIVE,
                    "support_weight": _weight(tail),
                }
            )
    return out


def _normalize_entry_key(head: str, tail: str) -> str:
    return f"{head}:{tail}"


def _summarize_inventory(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_head: Dict[str, int] = {}
    by_scope: Dict[str, int] = {}
    by_weight: Dict[str, int] = {}
    for entry in entries:
        by_head[entry["head"]] = by_head.get(entry["head"], 0) + 1
        scope = entry.get("scope", "unknown")
        by_scope[scope] = by_scope.get(scope, 0) + 1
        weight = str(entry["support_weight"])
        by_weight[weight] = by_weight.get(weight, 0) + 1
    return {
        "total": len(entries),
        "by_head": dict(sorted(by_head.items())),
        "by_scope": dict(sorted(by_scope.items())),
        "by_support_weight": dict(sorted(by_weight.items(), key=lambda kv: int(kv[0]))),
    }


def _single_qubit_native_coverage(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tails = {entry["tail"] for entry in entries}
    expected: List[str] = []
    missing: List[str] = []
    for idx in range(N_QUBITS):
        for pauli in ("X", "Y", "Z"):
            target = ["I"] * N_QUBITS
            target[idx] = pauli
            compact = "".join(target)
            expected.append(compact)
            if compact not in tails:
                missing.append(compact)
    return {
        "expected_count": len(expected),
        "covered_count": len(expected) - len(missing),
        "missing": missing,
    }


def _inventory_diff(local_table: LocalTable) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ours_entries = [
        {
            "native_id": entry.native_id,
            "head": entry.head,
            "tail": entry.tail.compact(),
            "scope": entry.scope,
            "support_weight": _weight(entry.tail.compact()),
        }
        for entry in local_table.entries
    ]
    gross_entries = _gross_inventory_from_csv()

    ours_by_key = {
        _normalize_entry_key(entry["head"], entry["tail"]): entry
        for entry in ours_entries
    }
    gross_by_key = {
        _normalize_entry_key(entry["head"], entry["tail"]): entry
        for entry in gross_entries
    }
    all_keys = sorted(set(ours_by_key).union(gross_by_key))

    rows: List[Dict[str, Any]] = []
    only_ours = 0
    only_gross = 0
    both = 0

    for key in all_keys:
        ours = ours_by_key.get(key)
        gross = gross_by_key.get(key)
        in_ours = ours is not None
        in_gross = gross is not None
        if in_ours and in_gross:
            source_compiler = "both"
            both += 1
        elif in_ours:
            source_compiler = "ours_only"
            only_ours += 1
        else:
            source_compiler = "gross_only"
            only_gross += 1

        row = {
            "normalized_id": key,
            "head": (ours or gross)["head"],
            "tail": (ours or gross)["tail"],
            "scope": (ours or gross).get("scope", SCOPE_INTRA_BLOCK_NATIVE),
            "source_compiler": source_compiler,
            "in_ours": int(in_ours),
            "in_gross": int(in_gross),
            "ours_native_id": ours["native_id"] if ours else "",
            "gross_native_id": gross["native_id"] if gross else "",
        }
        rows.append(row)

    cross_rows = [entry for entry in ours_entries if entry["scope"] == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE]
    cross_key_hits = [
        _normalize_entry_key(entry["head"], entry["tail"])
        for entry in cross_rows
        if _normalize_entry_key(entry["head"], entry["tail"]) in gross_by_key
    ]
    relation = "modified_variant"
    if only_ours == 0 and only_gross == 0:
        relation = "equal"
    elif only_gross == 0 and only_ours > 0:
        relation = "strict_superset"
    elif only_ours == 0 and only_gross > 0:
        relation = "strict_subset"

    sanity = {
        "ours_single_qubit_coverage": _single_qubit_native_coverage(ours_entries),
        "gross_single_qubit_coverage": _single_qubit_native_coverage(gross_entries),
        "cross_logical_seed_x1p_x1p_prime": {
            "tail": "XIIIIIIXIII",
            "present_in_ours_cross_scope": any(
                entry["tail"] == "XIIIIIIXIII"
                and entry["scope"] == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE
                for entry in ours_entries
            ),
            "present_in_gross_native": any(entry["tail"] == "XIIIIIIXIII" for entry in gross_entries),
        },
    }

    summary = {
        "ours_inventory": _summarize_inventory(ours_entries),
        "gross_inventory": _summarize_inventory(gross_entries),
        "intersection_count": both,
        "only_ours_count": only_ours,
        "only_gross_count": only_gross,
        "cross_logical_scope_count": len(cross_rows),
        "cross_logical_also_in_gross_count": len(cross_key_hits),
        "cross_logical_only_in_ours_count": len(cross_rows) - len(cross_key_hits),
        "set_relation": relation,
        "sanity_checks": sanity,
    }
    return rows, summary


STATUS_VALUES = tuple(status.value for status in SearchStatus)


def _status_name(status: SearchStatus | str) -> str:
    return search_status_name(status)


def _status_flags(status: SearchStatus | str) -> Dict[str, bool]:
    return search_status_flags(status)


def _search_status_note(status: SearchStatus | str) -> str:
    return search_status_note(status)


def _assert_no_budget_invariant(
    *,
    status: SearchStatus | str,
    max_popped_states: int | None,
    target: str,
    mode: str,
) -> None:
    assert_no_budget_status_invariant(
        status=status,
        max_popped_states=max_popped_states,
        context=f"mode={mode} target={target}",
    )


def _path_signature_from_plan(
    *,
    status: SearchStatus | str,
    plan,
) -> str:
    name = _status_name(status)
    if name == SearchStatus.RULE_UNREACHABLE.value:
        return "RULE_UNREACHABLE"
    if name == SearchStatus.SEARCH_TRUNCATED.value:
        return "SEARCH_TRUNCATED"
    if plan is None or not getattr(plan, "steps", None):
        return name
    steps = list(plan.steps)
    first = steps[0]
    p3_steps = sum(1 for step in steps if step.family == "P3")
    signature = f"{first.family}[{first.scope}]"
    if p3_steps == 1:
        signature += "->P3"
    elif p3_steps > 1:
        signature += f"->P3x{p3_steps}"
    return signature


def _mode_result_from_search_result(
    result: SynthesisSearchResult,
    *,
    target: str,
    mode: str,
    max_popped_states: int | None,
) -> Dict[str, Any]:
    _assert_no_budget_invariant(
        status=result.status,
        max_popped_states=max_popped_states,
        target=target,
        mode=mode,
    )
    flags = _status_flags(result.status)
    plan = result.plan
    if flags["reachable"] and plan is None:
        raise AssertionError(f"reachable status without plan for mode={mode} target={target}")
    if flags["truncated"] and plan is not None:
        raise AssertionError(f"truncated status unexpectedly has plan for mode={mode} target={target}")

    first = plan.steps[0] if plan is not None and plan.steps else None
    p3_steps = 0 if plan is None else sum(1 for step in plan.steps if step.family == "P3")
    used_cross = False if plan is None else any(
        step.scope == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE for step in plan.steps
    )
    used_bell = False if plan is None else any(
        step.scope == SCOPE_INTER_MODULE_BELL for step in plan.steps
    )

    return {
        "search_status": result.status.value,
        "search_status_note": _search_status_note(result.status),
        "reachable": flags["reachable"],
        "optimal": flags["optimal"],
        "truncated": flags["truncated"],
        "rule_unreachable": flags["rule_unreachable"],
        "cost": None if plan is None else int(plan.total_cost),
        "projected_best_cost": None if plan is None else int(plan.projected_best_cost),
        "first_step_family": None if first is None else first.family,
        "first_step_scope": None if first is None else first.scope,
        "final_hidden": None if plan is None else plan.best_hidden.name,
        "p3_steps": p3_steps,
        "used_cross_logical_native": used_cross,
        "used_inter_module_bell": used_bell,
        "remote_inter_module_count": 0 if plan is None else int(plan.remote_inter_module_count),
        "frame_unreachable": flags["rule_unreachable"],
        "search_truncated": flags["truncated"],
        "reason": _search_status_note(result.status),
        "path_signature": _path_signature_from_plan(status=result.status, plan=plan),
    }


def _gross_mode_result(cost: int) -> Dict[str, Any]:
    flags = _status_flags(SearchStatus.FOUND_OPTIMAL)
    return {
        "search_status": SearchStatus.FOUND_OPTIMAL.value,
        "search_status_note": _search_status_note(SearchStatus.FOUND_OPTIMAL),
        "reachable": flags["reachable"],
        "optimal": flags["optimal"],
        "truncated": flags["truncated"],
        "rule_unreachable": flags["rule_unreachable"],
        "cost": int(cost),
        "projected_best_cost": int(cost),
        "first_step_family": None,
        "first_step_scope": None,
        "final_hidden": None,
        "p3_steps": 0,
        "used_cross_logical_native": False,
        "used_inter_module_bell": False,
        "remote_inter_module_count": 0,
        "frame_unreachable": False,
        "search_truncated": False,
        "reason": _search_status_note(SearchStatus.FOUND_OPTIMAL),
        "path_signature": "GROSS_BASELINE",
    }


def _mode_result_from_outcome(outcome: Any) -> Dict[str, Any]:
    if outcome is None:
        raise TypeError("benchmark/report consumers must not flatten a missing outcome to RULE_UNREACHABLE")

    raw_status = getattr(outcome, "search_status", None)
    if raw_status is None:
        raise TypeError("benchmark/report consumers require an explicit four-state search_status")
    search_status = raw_status.value if isinstance(raw_status, SearchStatus) else str(raw_status)
    flags = _status_flags(search_status)
    plan = outcome.plan
    if flags["reachable"] and plan is None:
        raise AssertionError(f"reachable status without plan for outcome status={search_status}")
    if not flags["reachable"] and plan is not None:
        raise AssertionError(f"non-reachable status unexpectedly has plan for outcome status={search_status}")
    first = plan.steps[0] if plan is not None and plan.steps else None
    first_family = first.family if first else None
    first_scope = first.scope if first else None
    used_cross = False if plan is None else any(
        step.scope == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE for step in plan.steps
    )
    used_bell = False if plan is None else any(
        step.scope == SCOPE_INTER_MODULE_BELL for step in plan.steps
    )
    p3_steps = 0 if plan is None else sum(1 for step in plan.steps if step.family == "P3")
    return {
        "reachable": flags["reachable"],
        "optimal": flags["optimal"],
        "truncated": flags["truncated"],
        "rule_unreachable": flags["rule_unreachable"],
        "cost": int(plan.total_cost) if plan is not None and flags["reachable"] else None,
        "first_step_family": first_family,
        "first_step_scope": first_scope,
        "final_hidden": plan.best_hidden.name if plan is not None else None,
        "p3_steps": p3_steps,
        "used_cross_logical_native": used_cross,
        "used_inter_module_bell": used_bell,
        "remote_inter_module_count": int(plan.remote_inter_module_count) if plan is not None else 0,
        "frame_unreachable": flags["rule_unreachable"],
        "search_status": search_status,
        "search_status_note": _search_status_note(search_status),
        "search_truncated": flags["truncated"],
        "reason": _search_status_note(search_status),
        "path_signature": _path_signature_from_plan(status=search_status, plan=plan),
    }


def _stratum_from_full_mode(result: Dict[str, Any]) -> str:
    if result.get("truncated") or result.get("search_status") == SearchStatus.SEARCH_TRUNCATED.value:
        return "search_truncated"
    if result.get("rule_unreachable") or result.get("search_status") == SearchStatus.RULE_UNREACHABLE.value:
        return "frame_unreachable"
    if result["first_step_family"] == "P2":
        return "joint_source"
    if result["first_step_family"] == "P0" and result["first_step_scope"] == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE:
        return "cross_logical_native"
    return "direct_local"


STRATA = ("all", "direct_local", "joint_source", "cross_logical_native", "frame_unreachable", "search_truncated")
MODE_COST_FIELD = {
    "gross_baseline": "gross_baseline_cost",
    "ours_no_joint": "ours_no_joint_cost",
    "ours_full": "ours_full_cost",
}
MODE_REACHABLE_FIELD = {
    "gross_baseline": "gross_baseline_reachable",
    "ours_no_joint": "ours_no_joint_reachable",
    "ours_full": "ours_full_reachable",
}
def _cost_convention_note() -> str:
    return (
        "gross_cost uses beta_headed_min; ours_*_cost uses the current-frame "
        "P0/P2 source plus P3 shortest-path cost."
    )


PER_TARGET_FIELDS = [
    "target",
    "weight",
    "target_tail_bits",
    "max_popped_states",
    "used_no_budget",
    "search_status_gross",
    "search_status_note_gross",
    "search_status_ours_no_joint",
    "search_status_note_ours_no_joint",
    "search_status_ours_full",
    "search_status_note_ours_full",
    "reachable_gross",
    "reachable_ours_no_joint",
    "reachable_ours_full",
    "truncated_gross",
    "truncated_ours_no_joint",
    "truncated_ours_full",
    "rule_unreachable_gross",
    "rule_unreachable_ours_no_joint",
    "rule_unreachable_ours_full",
    "optimal_gross",
    "optimal_ours_no_joint",
    "optimal_ours_full",
    "gross_baseline_reachable",
    "gross_baseline_cost",
    "gross_baseline_search_status",
    "ours_no_joint_reachable",
    "ours_no_joint_cost",
    "ours_no_joint_first_step_family",
    "ours_no_joint_first_step_scope",
    "ours_no_joint_final_hidden",
    "ours_no_joint_p3_steps",
    "ours_no_joint_frame_unreachable",
    "ours_no_joint_search_status",
    "ours_no_joint_search_truncated",
    "ours_no_joint_path_signature",
    "ours_no_joint_reason",
    "ours_full_reachable",
    "ours_full_cost",
    "ours_full_first_step_family",
    "ours_full_first_step_scope",
    "ours_full_final_hidden",
    "ours_full_p3_steps",
    "ours_full_frame_unreachable",
    "ours_full_search_status",
    "ours_full_search_truncated",
    "ours_full_path_signature",
    "ours_full_reason",
    "used_cross_logical_block_native",
    "used_inter_module_bell",
    "remote_inter_module_count",
    "stratum",
    "delta_vs_baseline",
    "delta_joint_gain",
    "cost_convention_note",
]


def _value_at_rank(histogram: Dict[int, int], rank: int) -> int:
    seen = 0
    for value in sorted(histogram):
        seen += histogram[value]
        if seen > rank:
            return value
    raise RuntimeError("rank outside histogram cardinality")


def _percentile_from_hist(histogram: Dict[int, int], q: float) -> Optional[float]:
    total = sum(histogram.values())
    if total <= 0:
        return None
    if total == 1:
        return float(next(iter(histogram.keys())))
    pos = q * (total - 1)
    lo_rank = int(math.floor(pos))
    hi_rank = int(math.ceil(pos))
    lo_value = _value_at_rank(histogram, lo_rank)
    hi_value = _value_at_rank(histogram, hi_rank)
    if lo_rank == hi_rank:
        return float(lo_value)
    frac = pos - lo_rank
    return lo_value * (1.0 - frac) + hi_value * frac


@dataclass
class CostStatsAccumulator:
    total_targets: int = 0
    reachable_count: int = 0
    rule_unreachable_count: int = 0
    search_truncated_count: int = 0
    optimal_count: int = 0
    sum_cost: int = 0
    min_cost: Optional[int] = None
    max_cost: Optional[int] = None
    histogram: Dict[int, int] = None  # type: ignore[assignment]
    status_counts: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.histogram is None:
            self.histogram = {}
        if self.status_counts is None:
            self.status_counts = {name: 0 for name in STATUS_VALUES}

    def add(self, reachable: bool, cost: int | None, search_status: str) -> None:
        self.total_targets += 1
        self.status_counts[search_status] = self.status_counts.get(search_status, 0) + 1
        if search_status == SearchStatus.RULE_UNREACHABLE.value:
            self.rule_unreachable_count += 1
        elif search_status == SearchStatus.SEARCH_TRUNCATED.value:
            self.search_truncated_count += 1
        elif search_status == SearchStatus.FOUND_OPTIMAL.value:
            self.optimal_count += 1
        if not reachable or cost is None:
            return
        value = int(cost)
        self.reachable_count += 1
        self.sum_cost += value
        self.min_cost = value if self.min_cost is None else min(self.min_cost, value)
        self.max_cost = value if self.max_cost is None else max(self.max_cost, value)
        self.histogram[value] = self.histogram.get(value, 0) + 1

    def to_summary(self) -> Dict[str, Any]:
        unreachable_count = self.rule_unreachable_count
        not_reached_count = self.total_targets - self.reachable_count
        if self.reachable_count == 0:
            return {
                "reachable_count": 0,
                "unreachable_count": unreachable_count,
                "search_truncated_count": self.search_truncated_count,
                "not_reached_count": not_reached_count,
                "mean_cost": None,
                "median_cost": None,
                "p10_cost": None,
                "p90_cost": None,
                "min_cost": None,
                "max_cost": None,
                "histogram": {},
                "low_cost_buckets": {},
                "counts_by_status": dict(self.status_counts),
            }
        hist_json = {str(value): self.histogram[value] for value in sorted(self.histogram)}
        limit = max(self.max_cost or 0, 9)
        low_cost_buckets = {
            str(bucket): self.histogram.get(bucket, 0)
            for bucket in range(1, limit + 1, 2)
        }
        return {
            "reachable_count": self.reachable_count,
            "unreachable_count": unreachable_count,
            "search_truncated_count": self.search_truncated_count,
            "not_reached_count": not_reached_count,
            "mean_cost": self.sum_cost / self.reachable_count,
            "median_cost": _percentile_from_hist(self.histogram, 0.50),
            "p10_cost": _percentile_from_hist(self.histogram, 0.10),
            "p90_cost": _percentile_from_hist(self.histogram, 0.90),
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "histogram": hist_json,
            "low_cost_buckets": low_cost_buckets,
            "counts_by_status": dict(self.status_counts),
        }


@dataclass
class DeltaStatsAccumulator:
    comparable_count: int = 0
    skipped_unreachable_count: int = 0
    skipped_search_truncated_count: int = 0
    skipped_not_reached_count: int = 0
    win_count: int = 0
    tie_count: int = 0
    loss_count: int = 0
    sum_delta: int = 0
    best_savings: Optional[int] = None
    worst_regression: Optional[int] = None
    histogram: Dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.histogram is None:
            self.histogram = {}

    def add(
        self,
        lhs_cost: int | None,
        rhs_cost: int | None,
        *,
        lhs_status: str | None = None,
        rhs_status: str | None = None,
    ) -> None:
        if lhs_cost is None or rhs_cost is None:
            self.skipped_not_reached_count += 1
            statuses = {lhs_status, rhs_status}
            if SearchStatus.SEARCH_TRUNCATED.value in statuses:
                self.skipped_search_truncated_count += 1
            elif SearchStatus.RULE_UNREACHABLE.value in statuses:
                self.skipped_unreachable_count += 1
            return
        delta = int(lhs_cost) - int(rhs_cost)
        self.comparable_count += 1
        self.sum_delta += delta
        self.best_savings = delta if self.best_savings is None else min(self.best_savings, delta)
        self.worst_regression = delta if self.worst_regression is None else max(self.worst_regression, delta)
        self.histogram[delta] = self.histogram.get(delta, 0) + 1
        if delta < 0:
            self.win_count += 1
        elif delta == 0:
            self.tie_count += 1
        else:
            self.loss_count += 1

    def to_summary(self) -> Dict[str, Any]:
        mean_delta = None
        if self.comparable_count > 0:
            mean_delta = self.sum_delta / self.comparable_count
        return {
            "computed_on_common_reachable_targets_only": True,
            "common_reachable_count": self.comparable_count,
            "comparable_count": self.comparable_count,
            "skipped_unreachable_count": self.skipped_unreachable_count,
            "skipped_search_truncated_count": self.skipped_search_truncated_count,
            "skipped_not_reached_count": self.skipped_not_reached_count,
            "win_count": self.win_count,
            "tie_count": self.tie_count,
            "loss_count": self.loss_count,
            "mean_delta": mean_delta,
            "median_delta": _percentile_from_hist(self.histogram, 0.50),
            "best_savings": self.best_savings,
            "worst_regression": self.worst_regression,
        }


@dataclass
class BenchmarkSummaryAccumulator:
    by_mode: Dict[str, Dict[str, CostStatsAccumulator]]
    pair_full_vs_baseline: DeltaStatsAccumulator
    pair_full_vs_no_joint: DeltaStatsAccumulator
    target_count: int = 0
    joint_only_reachability_count: int = 0
    targets_first_solved_by_p2: int = 0
    targets_solved_direct_cross_logical_native: int = 0
    targets_best_path_without_p2: int = 0
    gross_reachable_and_ours_full_rule_unreachable: int = 0
    gross_reachable_and_ours_full_truncated: int = 0
    ours_full_reachable_and_ours_no_joint_rule_unreachable: int = 0
    ours_full_reachable_and_ours_no_joint_truncated: int = 0

    @classmethod
    def new(cls) -> "BenchmarkSummaryAccumulator":
        mode_map: Dict[str, Dict[str, CostStatsAccumulator]] = {}
        for mode in MODE_COST_FIELD:
            mode_map[mode] = {stratum: CostStatsAccumulator() for stratum in STRATA}
        return cls(
            by_mode=mode_map,
            pair_full_vs_baseline=DeltaStatsAccumulator(),
            pair_full_vs_no_joint=DeltaStatsAccumulator(),
        )

    def add_row(self, row: Dict[str, Any]) -> None:
        self.target_count += 1
        stratum = str(row["stratum"])
        strata_to_update = ("all", stratum)

        for mode in MODE_COST_FIELD:
            reachable = bool(row[MODE_REACHABLE_FIELD[mode]])
            raw_cost = row[MODE_COST_FIELD[mode]]
            cost = None if raw_cost is None else int(raw_cost)
            default_status = (
                SearchStatus.FOUND_OPTIMAL.value if reachable else SearchStatus.RULE_UNREACHABLE.value
            )
            search_status = str(row.get(f"{mode}_search_status", default_status))
            for scope in strata_to_update:
                self.by_mode[mode][scope].add(reachable, cost, search_status)

        self.pair_full_vs_baseline.add(
            row["ours_full_cost"],
            row["gross_baseline_cost"],
            lhs_status=str(row["ours_full_search_status"]),
        )
        self.pair_full_vs_no_joint.add(
            row["ours_full_cost"],
            row["ours_no_joint_cost"],
            lhs_status=str(row["ours_full_search_status"]),
            rhs_status=str(row["ours_no_joint_search_status"]),
        )

        if row["ours_full_reachable"] and not row["ours_no_joint_reachable"]:
            self.joint_only_reachability_count += 1
        if row["reachable_gross"] and row["rule_unreachable_ours_full"]:
            self.gross_reachable_and_ours_full_rule_unreachable += 1
        if row["reachable_gross"] and row["truncated_ours_full"]:
            self.gross_reachable_and_ours_full_truncated += 1
        if row["reachable_ours_full"] and row["rule_unreachable_ours_no_joint"]:
            self.ours_full_reachable_and_ours_no_joint_rule_unreachable += 1
        if row["reachable_ours_full"] and row["truncated_ours_no_joint"]:
            self.ours_full_reachable_and_ours_no_joint_truncated += 1
        if row["ours_full_first_step_family"] == "P2":
            self.targets_first_solved_by_p2 += 1
        if (
            row["ours_full_first_step_family"] == "P0"
            and row["ours_full_first_step_scope"] == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE
        ):
            self.targets_solved_direct_cross_logical_native += 1
        if row["ours_full_reachable"] and row["ours_full_first_step_family"] != "P2":
            self.targets_best_path_without_p2 += 1

    def to_summary(self) -> Dict[str, Any]:
        return {
            "by_mode": {
                mode: {
                    stratum: self.by_mode[mode][stratum].to_summary()
                    for stratum in STRATA
                }
                for mode in MODE_COST_FIELD
            },
            "pairwise": {
                "ours_full_vs_gross_baseline": self.pair_full_vs_baseline.to_summary(),
                "ours_full_vs_ours_no_joint": self.pair_full_vs_no_joint.to_summary(),
                "joint_only_reachability_count": self.joint_only_reachability_count,
            },
            "counts_by_status": {
                mode: dict(self.by_mode[mode]["all"].status_counts)
                for mode in MODE_COST_FIELD
            },
            "special_counts": {
                "targets_first_solved_by_p2": self.targets_first_solved_by_p2,
                "targets_solved_direct_cross_logical_native": self.targets_solved_direct_cross_logical_native,
                "targets_best_path_without_p2": self.targets_best_path_without_p2,
                "gross_reachable_and_ours_full_rule_unreachable": (
                    self.gross_reachable_and_ours_full_rule_unreachable
                ),
                "gross_reachable_and_ours_full_truncated": self.gross_reachable_and_ours_full_truncated,
                "ours_full_reachable_and_ours_no_joint_rule_unreachable": (
                    self.ours_full_reachable_and_ours_no_joint_rule_unreachable
                ),
                "ours_full_reachable_and_ours_no_joint_truncated": (
                    self.ours_full_reachable_and_ours_no_joint_truncated
                ),
            },
        }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_target_row(
    *,
    target: str,
    baseline_costs: Sequence[int] | Mapping[int, int],
    local_table: LocalTable,
    joint_table: JointTable,
    cfg: BenchmarkConfig,
) -> Dict[str, Any]:
    tail_bits = _tail_bits_from_compact(target)
    baseline_cost = _baseline_cost_at(baseline_costs, tail_bits)
    target_tail = Tail.from_str(target)

    gross = _gross_mode_result(baseline_cost)
    ours_no_joint = _mode_result_from_search_result(
        synthesize_search(
            target_tail,
            local_table,
            joint_table,
            max_popped_states=cfg.max_popped_states,
            enable_p2=False,
        ),
        target=target,
        mode="ours_no_joint",
        max_popped_states=cfg.max_popped_states,
    )
    ours_full = _mode_result_from_search_result(
        synthesize_search(
            target_tail,
            local_table,
            joint_table,
            max_popped_states=cfg.max_popped_states,
            enable_p2=not cfg.disable_p2,
        ),
        target=target,
        mode="ours_full",
        max_popped_states=cfg.max_popped_states,
    )

    stratum = _stratum_from_full_mode(ours_full)
    full_cost = ours_full["cost"]
    no_joint_cost = ours_no_joint["cost"]

    return {
        "target": target,
        "weight": _weight(target),
        "target_tail_bits": tail_bits,
        "max_popped_states": cfg.max_popped_states,
        "used_no_budget": cfg.max_popped_states is None,
        "search_status_gross": gross["search_status"],
        "search_status_note_gross": gross["search_status_note"],
        "search_status_ours_no_joint": ours_no_joint["search_status"],
        "search_status_note_ours_no_joint": ours_no_joint["search_status_note"],
        "search_status_ours_full": ours_full["search_status"],
        "search_status_note_ours_full": ours_full["search_status_note"],
        "reachable_gross": gross["reachable"],
        "reachable_ours_no_joint": ours_no_joint["reachable"],
        "reachable_ours_full": ours_full["reachable"],
        "truncated_gross": gross["truncated"],
        "truncated_ours_no_joint": ours_no_joint["truncated"],
        "truncated_ours_full": ours_full["truncated"],
        "rule_unreachable_gross": gross["rule_unreachable"],
        "rule_unreachable_ours_no_joint": ours_no_joint["rule_unreachable"],
        "rule_unreachable_ours_full": ours_full["rule_unreachable"],
        "optimal_gross": gross["optimal"],
        "optimal_ours_no_joint": ours_no_joint["optimal"],
        "optimal_ours_full": ours_full["optimal"],
        "gross_baseline_reachable": gross["reachable"],
        "gross_baseline_cost": gross["cost"],
        "gross_baseline_search_status": gross["search_status"],
        "ours_no_joint_reachable": ours_no_joint["reachable"],
        "ours_no_joint_cost": no_joint_cost,
        "ours_no_joint_first_step_family": ours_no_joint["first_step_family"],
        "ours_no_joint_first_step_scope": ours_no_joint["first_step_scope"],
        "ours_no_joint_final_hidden": ours_no_joint["final_hidden"],
        "ours_no_joint_p3_steps": ours_no_joint["p3_steps"],
        "ours_no_joint_frame_unreachable": ours_no_joint["rule_unreachable"],
        "ours_no_joint_search_status": ours_no_joint["search_status"],
        "ours_no_joint_search_truncated": ours_no_joint["truncated"],
        "ours_no_joint_path_signature": ours_no_joint["path_signature"],
        "ours_no_joint_reason": ours_no_joint["search_status_note"],
        "ours_full_reachable": ours_full["reachable"],
        "ours_full_cost": full_cost,
        "ours_full_first_step_family": ours_full["first_step_family"],
        "ours_full_first_step_scope": ours_full["first_step_scope"],
        "ours_full_final_hidden": ours_full["final_hidden"],
        "ours_full_p3_steps": ours_full["p3_steps"],
        "ours_full_frame_unreachable": ours_full["rule_unreachable"],
        "ours_full_search_status": ours_full["search_status"],
        "ours_full_search_truncated": ours_full["truncated"],
        "ours_full_path_signature": ours_full["path_signature"],
        "ours_full_reason": ours_full["search_status_note"],
        "used_cross_logical_block_native": ours_full["used_cross_logical_native"],
        "used_inter_module_bell": ours_full["used_inter_module_bell"],
        "remote_inter_module_count": ours_full["remote_inter_module_count"],
        "stratum": stratum,
        "delta_vs_baseline": None if full_cost is None else full_cost - baseline_cost,
        "delta_joint_gain": None if full_cost is None or no_joint_cost is None else full_cost - no_joint_cost,
        "cost_convention_note": _cost_convention_note(),
    }


def _iter_target_rows(
    targets: Iterable[str],
    baseline_costs: Sequence[int] | Mapping[int, int],
    local_table: LocalTable,
    joint_table: JointTable,
    cfg: BenchmarkConfig,
) -> Iterator[Dict[str, Any]]:
    for idx, target in enumerate(targets, start=1):
        yield build_target_row(
            target=target,
            baseline_costs=baseline_costs,
            local_table=local_table,
            joint_table=joint_table,
            cfg=cfg,
        )

        if cfg.progress_every > 0 and idx % cfg.progress_every == 0:
            print(f"[progress] processed {idx} targets")


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    local_table, joint_table = _load_tables(cfg)

    if cfg.mode == "sample":
        targets = _sample_targets(
            seed=cfg.seed,
            count=cfg.count,
            max_weight=cfg.max_weight,
            include_identity=cfg.include_identity,
        )
        required_indices = {_tail_bits_from_compact(target) for target in targets}
        baseline_costs = _load_baseline_costs(cfg.baseline_json, required_indices=required_indices)
    else:
        targets = _iter_exhaustive_targets(include_identity=cfg.include_identity)
        baseline_costs = _load_baseline_costs(cfg.baseline_json)

    local_diff_rows, local_summary = _inventory_diff(local_table)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / "summary.json"
    per_target_path = cfg.output_dir / "per_target.csv"
    local_diff_path = cfg.output_dir / "local_inventory_diff.csv"
    semantic_notice_path = cfg.output_dir / "README_semantic_notice.md"

    acc = BenchmarkSummaryAccumulator.new()
    with per_target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_TARGET_FIELDS)
        writer.writeheader()
        for row in _iter_target_rows(targets, baseline_costs, local_table, joint_table, cfg):
            writer.writerow(row)
            acc.add_row(row)

    summary = acc.to_summary()
    summary["config"] = {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "count": cfg.count,
        "max_weight": cfg.max_weight,
        "include_identity": cfg.include_identity,
        "max_popped_states": cfg.max_popped_states,
        "disable_p2": cfg.disable_p2,
        "frame_id": local_table.frame_id,
        "jobs": cfg.jobs,
        "baseline_json": str(cfg.baseline_json),
        "local_table_path": str(cfg.local_table_path),
        "joint_table_path": str(cfg.joint_table_path),
    }
    summary["inventory"] = local_summary
    summary["target_count"] = acc.target_count
    summary["semantic_contract"] = {
        "status_values": list(STATUS_VALUES),
        "search_api": "synthesize_search",
        "used_no_budget": cfg.max_popped_states is None,
        "no_budget_invariant": {
            "forbidden_statuses": [
                SearchStatus.FOUND_REACHABLE_UPPER_BOUND.value,
                SearchStatus.SEARCH_TRUNCATED.value,
            ],
            "note": "No-budget runs must not emit upper-bound or truncated statuses.",
        },
        "status_meanings": {
            status.value: _status_flags(status)
            for status in SearchStatus
        },
        "note": (
            "Artifacts generated before the four-state search semantics fix are stale "
            "and not directly comparable."
        ),
        "legacy_wrapper_note": (
            "`synthesize()` is legacy compatibility only; benchmark/report/story artifacts "
            "derive reachability from synthesize_search() status."
        ),
        "rule_unreachable_note": (
            "If the rule set is theoretically complete, RULE_UNREACHABLE should be treated as "
            "an execution-layer bug signal, not as a default-acceptable outcome."
        ),
        "target_aware_p2_note": (
            "target-aware P2 remains a post-filter only and must not shrink the full source graph."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _write_csv(local_diff_path, local_diff_rows)
    semantic_notice_path.write_text(
        (
            "# Semantic Notice\n\n"
            "Artifacts generated before the four-state search semantics fix are stale and not comparable.\n"
            "Current reports distinguish FOUND_OPTIMAL, FOUND_REACHABLE_UPPER_BOUND, "
            "SEARCH_TRUNCATED, and RULE_UNREACHABLE.\n"
            "SEARCH_TRUNCATED is not RULE_UNREACHABLE.\n"
            "FOUND_REACHABLE_UPPER_BOUND means a witness exists but optimality was not yet proven.\n"
            "If the rule set is theoretically complete, RULE_UNREACHABLE should be treated as an "
            "execution-layer bug signal.\n"
            "Target-aware P2 remains post-filter only and must not shrink the full source graph.\n"
            "Benchmark/report/story consumers derive reachability from synthesize_search() status, "
            "not from the legacy synthesize() wrapper.\n"
        ),
        encoding="utf-8",
    )

    print(f"wrote summary: {summary_path}")
    print(f"wrote per-target: {per_target_path}")
    print(f"wrote local diff: {local_diff_path}")
    print(f"wrote semantic notice: {semantic_notice_path}")
    return {
        "summary_path": summary_path,
        "per_target_path": per_target_path,
        "local_diff_path": local_diff_path,
        "semantic_notice_path": semantic_notice_path,
        "summary": summary,
    }


def _timestamped_output_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / stamp


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Compare ours vs gross baseline on local/joint measurement synthesis")
    parser.add_argument("--mode", choices=("sample", "exhaustive"), default="sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-weight", type=int, default=None)
    parser.add_argument("--include-identity", action="store_true")
    parser.add_argument("--max-popped-states", type=int, default=2000)
    parser.add_argument("--disable-p2", action="store_true", help="disable P2 in ours_full mode")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--local-table-path", type=Path, default=DEFAULT_LOCAL_TABLE_PATH)
    parser.add_argument("--joint-table-path", type=Path, default=build_assets.DEFAULT_JOINT_TABLE_PATH)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = _timestamped_output_dir(DEFAULT_RESULTS_ROOT)

    max_popped_states = args.max_popped_states if args.max_popped_states > 0 else None
    return BenchmarkConfig(
        mode=args.mode,
        seed=args.seed,
        count=args.count,
        max_weight=args.max_weight,
        include_identity=args.include_identity,
        max_popped_states=max_popped_states,
        disable_p2=args.disable_p2,
        output_dir=output_dir,
        frame_id=args.frame_id,
        jobs=args.jobs,
        progress_every=args.progress_every,
        baseline_json=args.baseline_json,
        local_table_path=args.local_table_path,
        joint_table_path=args.joint_table_path,
    )


def main() -> int:
    cfg = parse_args()
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

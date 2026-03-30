from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .benchmark_compare import (
    DEFAULT_BASELINE_JSON,
    BenchmarkConfig,
    _load_baseline_costs,
    _load_tables,
    _tail_bits_from_compact,
    build_target_row,
)
from .io import DEFAULT_JOINT_TABLE_PATH, DEFAULT_LOCAL_TABLE_PATH
from .rules import generate_p2_sources
from .search import SearchStatus, search_status_flags, search_status_note
from .tables import JointTable, LocalTable, SCOPE_CROSS_LOGICAL_BLOCK_NATIVE, SCOPE_INTRA_BLOCK_NATIVE

RESULTS_ROOT = Path("benchmark") / "results"
X1P_CROSS_TAIL = "XIIIIIIXIII"
JOINT_PROBE_FALLBACKS = (
    "IIIIZIIIXII",
)
OLD_SUSPICIOUS_FALLBACKS = (
    "IIIIIIIYIII",
    "IIIIIIXIIXI",
)


@dataclass(frozen=True)
class TargetRow:
    target_tail: str
    weight: int
    strata: str
    gross_search_status: str
    gross_search_status_note: str
    gross_reachable: bool
    gross_cost: Optional[int]
    gross_optimal: bool
    ours_no_joint_search_status: str
    ours_no_joint_search_status_note: str
    ours_no_joint_reachable: bool
    ours_no_joint_cost: Optional[int]
    ours_full_search_status: str
    ours_full_search_status_note: str
    ours_full_reachable: bool
    ours_full_cost: Optional[int]
    ours_full_first_family: Optional[str]
    ours_full_first_scope: Optional[str]
    ours_full_path_signature: str
    ours_full_reason: str
    delta_ours_full_vs_gross: Optional[int]
    delta_ours_full_vs_ours_no_joint: Optional[int]
    cost_convention_note: str


@dataclass(frozen=True)
class StoryCase:
    story_label: str
    row: TargetRow
    canonical_target_label: str
    one_line_reason: str


@dataclass(frozen=True)
class CircuitSummary:
    circuit_name: str
    ordered_targets: List[str]
    gross_search_statuses: List[str]
    ours_no_joint_search_statuses: List[str]
    ours_full_search_statuses: List[str]
    ours_full_search_status_notes: List[str]
    gross_total_cost: Optional[int]
    ours_no_joint_total_cost: Optional[int]
    ours_full_total_cost: Optional[int]
    gross_reachable: bool
    ours_no_joint_reachable: bool
    ours_full_reachable: bool
    ours_full_p0_count: int
    ours_full_p2_count: int
    ours_full_rule_unreachable_count: int
    ours_full_search_truncated_count: int
    explanation: str


@dataclass
class StoryContext:
    cfg: BenchmarkConfig
    local_table: LocalTable
    joint_table: JointTable
    baseline_costs: Sequence[int] | Mapping[int, int]
    cache: Dict[str, TargetRow]


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    return text in ("1", "true", "yes")


def _parse_int_or_none(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    return int(text)


def _canonical_label(target_tail: str) -> str:
    if target_tail == X1P_CROSS_TAIL:
        return "X1P⊗X1P'"
    return target_tail


def _cost_convention_note() -> str:
    return (
        "gross_cost uses beta_headed_min; ours_*_cost uses the current-frame "
        "P0/P2 source plus P3 shortest-path cost."
    )


def _row_to_target_row(row: Dict[str, Any]) -> TargetRow:
    gross_status = str(
        row.get("search_status_gross", row.get("gross_baseline_search_status", SearchStatus.FOUND_OPTIMAL.value))
    )
    ours_no_joint_status = str(
        row.get("search_status_ours_no_joint", row.get("ours_no_joint_search_status", ""))
    )
    ours_full_status = str(row.get("search_status_ours_full", row.get("ours_full_search_status", "")))
    gross_flags = search_status_flags(gross_status)
    ours_no_joint_flags = search_status_flags(ours_no_joint_status)
    ours_full_flags = search_status_flags(ours_full_status)
    return TargetRow(
        target_tail=str(row["target"]),
        weight=int(row["weight"]),
        strata=str(row["stratum"]),
        gross_search_status=gross_status,
        gross_search_status_note=str(row.get("search_status_note_gross", search_status_note(gross_status))),
        gross_reachable=bool(gross_flags["reachable"]),
        gross_cost=_parse_int_or_none(row.get("gross_baseline_cost")) if gross_flags["reachable"] else None,
        gross_optimal=bool(gross_flags["optimal"]),
        ours_no_joint_search_status=ours_no_joint_status,
        ours_no_joint_search_status_note=str(
            row.get("search_status_note_ours_no_joint", search_status_note(ours_no_joint_status))
        ),
        ours_no_joint_reachable=bool(ours_no_joint_flags["reachable"]),
        ours_no_joint_cost=_parse_int_or_none(row.get("ours_no_joint_cost"))
        if ours_no_joint_flags["reachable"]
        else None,
        ours_full_search_status=ours_full_status,
        ours_full_search_status_note=str(
            row.get("search_status_note_ours_full", row.get("ours_full_reason", search_status_note(ours_full_status)))
        ),
        ours_full_reachable=bool(ours_full_flags["reachable"]),
        ours_full_cost=_parse_int_or_none(row.get("ours_full_cost")) if ours_full_flags["reachable"] else None,
        ours_full_first_family=(str(row.get("ours_full_first_step_family", "")).strip() or None),
        ours_full_first_scope=(str(row.get("ours_full_first_step_scope", "")).strip() or None),
        ours_full_path_signature=str(
            row.get("ours_full_path_signature", row.get("ours_full_first_step_family", "") or "")
        ),
        ours_full_reason=str(row.get("ours_full_reason", "")),
        delta_ours_full_vs_gross=_parse_int_or_none(row.get("delta_vs_baseline")),
        delta_ours_full_vs_ours_no_joint=_parse_int_or_none(row.get("delta_joint_gain")),
        cost_convention_note=str(row.get("cost_convention_note", _cost_convention_note())),
    )


def _load_per_target(path: Path) -> List[TargetRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_row_to_target_row(row) for row in csv.DictReader(handle)]


def _latest_results_dir(root: Path = RESULTS_ROOT) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"results root does not exist: {root}")
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no benchmark result directory found under {root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _context_from_input_dir(input_dir: Path, seed_rows: Sequence[TargetRow]) -> StoryContext:
    summary_path = input_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json in {input_dir}")
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary_cfg = payload.get("config", {})
    used_no_budget = bool(payload.get("semantic_contract", {}).get("used_no_budget", False))
    if not used_no_budget:
        raise ValueError(
            f"{input_dir} was not generated with no-budget search. Re-run benchmark_compare with "
            "--max-popped-states 0 before building the story."
        )

    cfg = BenchmarkConfig(
        mode=str(summary_cfg.get("mode", "sample")),
        seed=int(summary_cfg.get("seed", 7)),
        count=int(summary_cfg.get("count", max(20, len(seed_rows)))),
        max_weight=summary_cfg.get("max_weight"),
        include_identity=bool(summary_cfg.get("include_identity", False)),
        max_popped_states=None,
        disable_p2=False,
        output_dir=input_dir,
        frame_id=summary_cfg.get("frame_id"),
        jobs=int(summary_cfg.get("jobs", 1)),
        progress_every=0,
        baseline_json=Path(summary_cfg.get("baseline_json", DEFAULT_BASELINE_JSON)),
        local_table_path=Path(summary_cfg.get("local_table_path", DEFAULT_LOCAL_TABLE_PATH)),
        joint_table_path=Path(summary_cfg.get("joint_table_path", DEFAULT_JOINT_TABLE_PATH)),
    )
    local_table, joint_table = _load_tables(cfg)
    required_targets = {row.target_tail for row in seed_rows}
    required_targets.add(X1P_CROSS_TAIL)
    required_targets.update(_old_suspicious_targets())
    baseline_costs = _load_baseline_costs(
        cfg.baseline_json,
        required_indices={_tail_bits_from_compact(target) for target in required_targets},
    )
    cache = {row.target_tail: row for row in seed_rows}
    return StoryContext(
        cfg=cfg,
        local_table=local_table,
        joint_table=joint_table,
        baseline_costs=baseline_costs,
        cache=cache,
    )


def _ensure_baseline_target_cost(ctx: StoryContext, target_tail: str) -> None:
    if not isinstance(ctx.baseline_costs, dict):
        return
    tail_bits = _tail_bits_from_compact(target_tail)
    if tail_bits in ctx.baseline_costs:
        return
    loaded = _load_baseline_costs(ctx.cfg.baseline_json, required_indices={tail_bits})
    assert isinstance(loaded, dict)
    ctx.baseline_costs.update(loaded)


def _evaluate_target(ctx: StoryContext, target_tail: str) -> TargetRow:
    row = ctx.cache.get(target_tail)
    if row is None:
        _ensure_baseline_target_cost(ctx, target_tail)
        built = build_target_row(
            target=target_tail,
            baseline_costs=ctx.baseline_costs,
            local_table=ctx.local_table,
            joint_table=ctx.joint_table,
            cfg=ctx.cfg,
        )
        row = _row_to_target_row(built)
        ctx.cache[target_tail] = row
    return row


def _best_candidate(
    rows: Sequence[TargetRow],
    *,
    used_targets: set[str],
    predicate,
    score,
) -> Optional[TargetRow]:
    candidates = [row for row in rows if predicate(row) and row.target_tail not in used_targets]
    if not candidates:
        return None
    return sorted(candidates, key=score)[0]


def _direct_local_candidates(ctx: StoryContext, *, limit: int | None = None) -> List[TargetRow]:
    seen: set[str] = set()
    rows: List[TargetRow] = []
    for entry in ctx.local_table.entries:
        if entry.scope != SCOPE_INTRA_BLOCK_NATIVE or entry.head == "I":
            continue
        target = entry.tail.compact()
        if target in seen:
            continue
        seen.add(target)
        row = _evaluate_target(ctx, target)
        if (
            row.ours_full_search_status == SearchStatus.FOUND_OPTIMAL.value
            and row.ours_full_first_family == "P0"
            and row.ours_full_first_scope == SCOPE_INTRA_BLOCK_NATIVE
        ):
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _joint_source_candidates(ctx: StoryContext, *, limit: int | None = None) -> List[TargetRow]:
    seen: set[str] = set()
    rows: List[TargetRow] = []
    for step in generate_p2_sources(ctx.local_table, ctx.joint_table):
        target = step.dst.tail.compact()
        if target in seen:
            continue
        seen.add(target)
        row = _evaluate_target(ctx, target)
        if row.ours_full_first_family == "P2" and row.ours_full_reachable:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _sample_fill_pool(seed_rows: Sequence[TargetRow]) -> List[TargetRow]:
    return sorted(
        seed_rows,
        key=lambda row: (
            0 if row.ours_full_search_status == SearchStatus.FOUND_OPTIMAL.value else 1,
            row.weight,
            row.target_tail,
        ),
    )


def _old_suspicious_targets(root: Path = RESULTS_ROOT) -> List[str]:
    targets: List[str] = list(OLD_SUSPICIOUS_FALLBACKS)
    for path in sorted(root.glob("*/simple_targets.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                label = str(row.get("story_label", ""))
                target = str(row.get("target_tail", "")).strip()
                strata = str(row.get("strata", row.get("stratum", ""))).strip()
                status = str(row.get("ours_full_search_status", row.get("search_status", ""))).strip()
                reachable = row.get("ours_full_reachable", row.get("reachable", ""))
                if not target:
                    continue
                old_unreachable_like = (
                    label == "frame_unreachable_case"
                    or strata == "frame_unreachable"
                    or status in ("RULE_UNREACHABLE", "SEARCH_TRUNCATED")
                    or str(reachable).strip().lower() in ("0", "false", "no")
                )
                if old_unreachable_like and target not in targets:
                    targets.append(target)
    return targets


def _add_case(
    selected: List[StoryCase],
    used_targets: set[str],
    *,
    story_label: str,
    row: TargetRow,
    reason: str,
) -> None:
    if row.target_tail in used_targets:
        raise ValueError(f"duplicate story target selected: {row.target_tail}")
    used_targets.add(row.target_tail)
    selected.append(
        StoryCase(
            story_label=story_label,
            row=row,
            canonical_target_label=_canonical_label(row.target_tail),
            one_line_reason=reason,
        )
    )


def _is_direct_local_representative(row: TargetRow) -> bool:
    return (
        row.ours_full_reachable
        and row.ours_full_first_family == "P0"
        and row.ours_full_first_scope == SCOPE_INTRA_BLOCK_NATIVE
    )


def _pick_story_cases(seed_rows: Sequence[TargetRow], ctx: StoryContext) -> tuple[List[StoryCase], Optional[str]]:
    if not seed_rows:
        raise ValueError("per_target.csv has no rows")

    selected: List[StoryCase] = []
    used: set[str] = set()
    direct_local_candidates = _direct_local_candidates(ctx, limit=6)
    sample_pool = _sample_fill_pool(seed_rows)

    direct_local = _best_candidate(
        direct_local_candidates,
        used_targets=used,
        predicate=lambda row: row.gross_reachable and row.ours_full_reachable,
        score=lambda row: (
            abs(row.delta_ours_full_vs_gross or 0),
            row.weight,
            row.target_tail,
        ),
    )
    if direct_local is None:
        raise ValueError("could not find a direct_local control under current frame")
    _add_case(
        selected,
        used,
        story_label="control_same_direct_local",
        row=direct_local,
        reason="Both compilers already hit a direct local native measurement.",
    )

    cross = _evaluate_target(ctx, X1P_CROSS_TAIL)
    if not (
        cross.ours_full_search_status == SearchStatus.FOUND_OPTIMAL.value
        and cross.ours_full_first_family == "P0"
        and cross.ours_full_first_scope == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE
    ):
        raise ValueError("cross-logical native probe is missing or mislabeled under current frame")
    _add_case(
        selected,
        used,
        story_label="cross_logical_native_case",
        row=cross,
        reason="This is a real cross-logical native P0 case, not a placeholder.",
    )

    joint_predicate = lambda row: (  # noqa: E731
        row.ours_full_first_family == "P2"
        and row.ours_full_reachable
        and (
            (not row.ours_no_joint_reachable)
            or (
                row.ours_no_joint_cost is not None
                and row.ours_full_cost is not None
                and row.ours_full_cost < row.ours_no_joint_cost
            )
        )
    )
    joint_score = lambda row: (  # noqa: E731
        0 if not row.ours_no_joint_reachable else 1,
        -((row.ours_no_joint_cost or 10_000) - (row.ours_full_cost or 10_000)),
        row.target_tail,
    )
    joint = _best_candidate(
        [_evaluate_target(ctx, target) for target in JOINT_PROBE_FALLBACKS],
        used_targets=used,
        predicate=joint_predicate,
        score=joint_score,
    )
    if joint is None:
        joint = _best_candidate(
            _joint_source_candidates(ctx, limit=6),
            used_targets=used,
            predicate=joint_predicate,
            score=joint_score,
        )
    if joint is None:
        raise ValueError("could not find a real joint_source_win case under current frame")
    _add_case(
        selected,
        used,
        story_label="joint_source_win_case",
        row=joint,
        reason="Joint source P2 wins without relying on sample luck.",
    )

    honest_predicate = lambda row: (  # noqa: E731
        row.ours_full_reachable
        and row.gross_reachable
        and row.delta_ours_full_vs_gross is not None
        and row.delta_ours_full_vs_gross >= 0
    )
    honest_score = lambda row: (  # noqa: E731
        0 if row.delta_ours_full_vs_gross and row.delta_ours_full_vs_gross > 0 else 1,
        -(row.delta_ours_full_vs_gross or 0),
        row.weight,
        row.target_tail,
    )
    honest = _best_candidate(
        direct_local_candidates,
        used_targets=used,
        predicate=honest_predicate,
        score=honest_score,
    )
    if honest is None:
        honest = _best_candidate(
            direct_local_candidates + sample_pool + list(ctx.cache.values()),
            used_targets=used,
            predicate=honest_predicate,
            score=honest_score,
        )
    if honest is None:
        raise ValueError("could not find an honest_loss_or_tie case")
    _add_case(
        selected,
        used,
        story_label="honest_loss_or_tie_case",
        row=honest,
        reason=(
            "Gross baseline is cheaper here."
            if (honest.delta_ours_full_vs_gross or 0) > 0
            else "Both compilers tie here."
        ),
    )

    suspicious_candidates = [_evaluate_target(ctx, target) for target in _old_suspicious_targets()]
    suspicious = _best_candidate(
        suspicious_candidates,
        used_targets=used,
        predicate=lambda _row: True,
        score=lambda row: (
            0 if row.ours_full_search_status != SearchStatus.RULE_UNREACHABLE.value else 1,
            0 if row.ours_full_search_status != SearchStatus.SEARCH_TRUNCATED.value else 1,
            row.weight,
            row.target_tail,
        ),
    )
    if suspicious is None:
        raise ValueError("could not find any previously suspicious target from old story artifacts")
    _add_case(
        selected,
        used,
        story_label="previously_suspicious_case",
        row=suspicious,
        reason=(
            "This target appeared in an old unreachable-like story slot and has now been re-audited under no-budget search."
        ),
    )

    true_rule_unreachable = _best_candidate(
        sample_pool + direct_local_candidates + list(ctx.cache.values()),
        used_targets=used,
        predicate=lambda row: row.ours_full_search_status == SearchStatus.RULE_UNREACHABLE.value,
        score=lambda row: (row.weight, row.target_tail),
    )
    unreachable_note: Optional[str] = None
    if true_rule_unreachable is not None:
        _add_case(
            selected,
            used,
            story_label="frame_unreachable_case",
            row=true_rule_unreachable,
            reason="This is a true RULE_UNREACHABLE case under no-budget search.",
        )
    else:
        unreachable_note = (
            "No true RULE_UNREACHABLE case was found in the audited corpus after the four-state semantic fix."
        )

    filler_pool = sample_pool + direct_local_candidates + list(ctx.cache.values())
    filler_index = 1
    while len(selected) < 6:
        filler = _best_candidate(
            filler_pool,
            used_targets=used,
            predicate=lambda row: row.ours_full_reachable,
            score=lambda row: (
                0 if _is_direct_local_representative(row) else 1,
                row.weight,
                row.target_tail,
            ),
        )
        if filler is None:
            raise ValueError("could not fill six representative target slots without duplicates")
        _add_case(
            selected,
            used,
            story_label=f"sample_fill_case_{filler_index}",
            row=filler,
            reason="Additional audited sample row added to keep a six-target report without label conflicts.",
        )
        filler_index += 1

    if len({case.row.target_tail for case in selected}) != len(selected):
        raise ValueError("story selection produced duplicate representative targets")
    if not any(_is_direct_local_representative(case.row) for case in selected):
        raise ValueError("story selection lost the required direct_local control")
    if not any(case.story_label == "cross_logical_native_case" for case in selected):
        raise ValueError("story selection lost the required cross_logical_native case")
    if not any(case.story_label == "joint_source_win_case" for case in selected):
        raise ValueError("story selection lost the required joint_source_win case")
    if not any(case.story_label == "honest_loss_or_tie_case" for case in selected):
        raise ValueError("story selection lost the required honest_loss_or_tie case")

    return selected, unreachable_note


def _sum_mode_cost(rows: Sequence[TargetRow], *, reachable_attr: str, cost_attr: str) -> tuple[bool, Optional[int]]:
    reachable = all(getattr(row, reachable_attr) for row in rows)
    if not reachable:
        return False, None
    total = 0
    for row in rows:
        cost = getattr(row, cost_attr)
        if cost is None:
            return False, None
        total += int(cost)
    return True, total


def _build_circuits(story_cases: Sequence[StoryCase]) -> List[CircuitSummary]:
    by_label = {case.story_label: case for case in story_cases}
    representative_rows = [case.row for case in story_cases]
    by_target = {row.target_tail: row for row in representative_rows}
    direct_pool = [row for row in representative_rows if _is_direct_local_representative(row)]
    if len(direct_pool) < 2:
        raise ValueError(
            "story circuit builder needs at least two validated direct_local representative targets"
        )
    direct_targets = [row.target_tail for row in direct_pool]

    def pick_direct(exclude: set[str]) -> str:
        for target in direct_targets:
            if target not in exclude:
                return target
        raise ValueError("not enough distinct direct_local representative targets to build toy circuits")

    control = by_label["control_same_direct_local"].row.target_tail
    cross = by_label["cross_logical_native_case"].row.target_tail
    joint = by_label["joint_source_win_case"].row.target_tail
    direct_companion = pick_direct({control})

    circuit_specs = [
        (
            "all_local_control",
            [control, direct_companion],
            "All targets are validated direct_local representative controls.",
        ),
        (
            "cross_logical_plus_local",
            [cross, control],
            "Includes one real cross_logical_native target plus a validated direct_local representative control.",
        ),
        (
            "joint_matters",
            [joint, control],
            "Includes one real joint_source_win target plus a validated direct_local representative control.",
        ),
    ]

    circuits: List[CircuitSummary] = []
    for circuit_name, target_list, explanation in circuit_specs:
        rows = [by_target[target] for target in target_list]
        gross_reachable, gross_total = _sum_mode_cost(rows, reachable_attr="gross_reachable", cost_attr="gross_cost")
        no_joint_reachable, no_joint_total = _sum_mode_cost(
            rows,
            reachable_attr="ours_no_joint_reachable",
            cost_attr="ours_no_joint_cost",
        )
        full_reachable, full_total = _sum_mode_cost(
            rows,
            reachable_attr="ours_full_reachable",
            cost_attr="ours_full_cost",
        )
        circuits.append(
            CircuitSummary(
                circuit_name=circuit_name,
                ordered_targets=[row.target_tail for row in rows],
                gross_search_statuses=[row.gross_search_status for row in rows],
                ours_no_joint_search_statuses=[row.ours_no_joint_search_status for row in rows],
                ours_full_search_statuses=[row.ours_full_search_status for row in rows],
                ours_full_search_status_notes=[row.ours_full_search_status_note for row in rows],
                gross_total_cost=gross_total,
                ours_no_joint_total_cost=no_joint_total,
                ours_full_total_cost=full_total,
                gross_reachable=gross_reachable,
                ours_no_joint_reachable=no_joint_reachable,
                ours_full_reachable=full_reachable,
                ours_full_p0_count=sum(1 for row in rows if row.ours_full_first_family == "P0"),
                ours_full_p2_count=sum(1 for row in rows if row.ours_full_first_family == "P2"),
                ours_full_rule_unreachable_count=sum(
                    1 for row in rows if row.ours_full_search_status == SearchStatus.RULE_UNREACHABLE.value
                ),
                ours_full_search_truncated_count=sum(
                    1 for row in rows if row.ours_full_search_status == SearchStatus.SEARCH_TRUNCATED.value
                ),
                explanation=explanation,
            )
        )
    return circuits


def _write_simple_targets_csv(path: Path, story_cases: Sequence[StoryCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "story_label",
        "target_tail",
        "canonical_target_label",
        "strata",
        "gross_search_status",
        "gross_search_status_note",
        "ours_no_joint_search_status",
        "ours_no_joint_search_status_note",
        "ours_full_search_status",
        "ours_full_search_status_note",
        "gross_cost",
        "ours_no_joint_cost",
        "ours_full_cost",
        "ours_full_first_family",
        "ours_full_first_scope",
        "ours_full_path_signature",
        "cost_convention_note",
        "delta_ours_full_vs_gross",
        "delta_ours_full_vs_ours_no_joint",
        "one_line_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in story_cases:
            row = case.row
            writer.writerow(
                {
                    "story_label": case.story_label,
                    "target_tail": row.target_tail,
                    "canonical_target_label": case.canonical_target_label,
                    "strata": row.strata,
                    "gross_search_status": row.gross_search_status,
                    "gross_search_status_note": row.gross_search_status_note,
                    "ours_no_joint_search_status": row.ours_no_joint_search_status,
                    "ours_no_joint_search_status_note": row.ours_no_joint_search_status_note,
                    "ours_full_search_status": row.ours_full_search_status,
                    "ours_full_search_status_note": row.ours_full_search_status_note,
                    "gross_cost": row.gross_cost,
                    "ours_no_joint_cost": row.ours_no_joint_cost,
                    "ours_full_cost": row.ours_full_cost,
                    "ours_full_first_family": row.ours_full_first_family or "",
                    "ours_full_first_scope": row.ours_full_first_scope or "",
                    "ours_full_path_signature": row.ours_full_path_signature,
                    "cost_convention_note": row.cost_convention_note,
                    "delta_ours_full_vs_gross": row.delta_ours_full_vs_gross,
                    "delta_ours_full_vs_ours_no_joint": row.delta_ours_full_vs_ours_no_joint,
                    "one_line_reason": case.one_line_reason,
                }
            )


def _write_simple_circuits_csv(path: Path, circuits: Sequence[CircuitSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "circuit_name",
        "ordered_targets",
        "gross_search_statuses",
        "ours_no_joint_search_statuses",
        "ours_full_search_statuses",
        "ours_full_search_status_notes",
        "gross_total_cost",
        "ours_no_joint_total_cost",
        "ours_full_total_cost",
        "gross_reachable",
        "ours_no_joint_reachable",
        "ours_full_reachable",
        "ours_full_p0_count",
        "ours_full_p2_count",
        "ours_full_rule_unreachable_count",
        "ours_full_search_truncated_count",
        "explanation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for circuit in circuits:
            writer.writerow(
                {
                    "circuit_name": circuit.circuit_name,
                    "ordered_targets": ";".join(circuit.ordered_targets),
                    "gross_search_statuses": ";".join(circuit.gross_search_statuses),
                    "ours_no_joint_search_statuses": ";".join(circuit.ours_no_joint_search_statuses),
                    "ours_full_search_statuses": ";".join(circuit.ours_full_search_statuses),
                    "ours_full_search_status_notes": ";".join(circuit.ours_full_search_status_notes),
                    "gross_total_cost": circuit.gross_total_cost,
                    "ours_no_joint_total_cost": circuit.ours_no_joint_total_cost,
                    "ours_full_total_cost": circuit.ours_full_total_cost,
                    "gross_reachable": circuit.gross_reachable,
                    "ours_no_joint_reachable": circuit.ours_no_joint_reachable,
                    "ours_full_reachable": circuit.ours_full_reachable,
                    "ours_full_p0_count": circuit.ours_full_p0_count,
                    "ours_full_p2_count": circuit.ours_full_p2_count,
                    "ours_full_rule_unreachable_count": circuit.ours_full_rule_unreachable_count,
                    "ours_full_search_truncated_count": circuit.ours_full_search_truncated_count,
                    "explanation": circuit.explanation,
                }
            )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _write_story_markdown(
    path: Path,
    story_cases: Sequence[StoryCase],
    circuits: Sequence[CircuitSummary],
    *,
    unreachable_note: Optional[str],
) -> None:
    target_headers = [
        "story_label",
        "target_tail",
        "canonical_target_label",
        "gross_search_status",
        "ours_no_joint_search_status",
        "ours_full_search_status",
        "ours_full_search_status_note",
        "ours_full_first_family",
        "ours_full_first_scope",
        "ours_full_path_signature",
        "gross_cost",
        "ours_no_joint_cost",
        "ours_full_cost",
    ]
    target_rows = [
        [
            case.story_label,
            case.row.target_tail,
            case.canonical_target_label,
            case.row.gross_search_status,
            case.row.ours_no_joint_search_status,
            case.row.ours_full_search_status,
            case.row.ours_full_search_status_note,
            case.row.ours_full_first_family or "",
            case.row.ours_full_first_scope or "",
            case.row.ours_full_path_signature,
            case.row.gross_cost,
            case.row.ours_no_joint_cost,
            case.row.ours_full_cost,
        ]
        for case in story_cases
    ]
    circuit_headers = [
        "circuit_name",
        "ordered_targets",
        "ours_full_search_statuses",
        "ours_full_search_status_notes",
        "gross_total_cost",
        "ours_no_joint_total_cost",
        "ours_full_total_cost",
    ]
    circuit_rows = [
        [
            circuit.circuit_name,
            ";".join(circuit.ordered_targets),
            ";".join(circuit.ours_full_search_statuses),
            ";".join(circuit.ours_full_search_status_notes),
            circuit.gross_total_cost,
            circuit.ours_no_joint_total_cost,
            circuit.ours_full_total_cost,
        ]
        for circuit in circuits
    ]

    lines: List[str] = []
    lines.append("# Simple Story Benchmark")
    lines.append("")
    lines.append("## Section 1: One-paragraph takeaway")
    lines.append(
        "Gross baseline behaves like a rotation/native-lift realization, while our compiler searches over the dynamic P0/P2/P3 operation set."
    )
    lines.append(
        "This report is built from no-budget benchmark semantics: SEARCH_TRUNCATED is distinct from RULE_UNREACHABLE, and FOUND_REACHABLE_UPPER_BOUND would be called out explicitly as an unproven upper bound."
    )
    lines.append("")
    lines.append("## Section 2: Six representative targets")
    lines.append(_markdown_table(target_headers, target_rows))
    lines.append("")
    for case in story_cases:
        lines.append(
            f"- `{case.story_label}`: {case.one_line_reason} "
            f"Status={case.row.ours_full_search_status}. "
            f"Status note={case.row.ours_full_search_status_note}. "
            f"Path={case.row.ours_full_path_signature}. "
            f"Cost note: {case.row.cost_convention_note}"
        )
    if unreachable_note is not None:
        lines.append(f"- {unreachable_note}")
    lines.append("")
    lines.append("## Section 3: Three toy circuits")
    lines.append(_markdown_table(circuit_headers, circuit_rows))
    lines.append("")
    for circuit in circuits:
        lines.append(f"- `{circuit.circuit_name}`: {circuit.explanation}")
    lines.append("")
    lines.append("## Section 4: One-sentence conclusion")
    lines.append(
        "The practical difference is not that our compiler always wins, but that it now reports real status hygiene: direct native, joint source, and any true RULE_UNREACHABLE case are separated without inventing placeholder frame-unreachable rows."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_grouped_bars(
    *,
    labels: Sequence[str],
    gross: Sequence[Optional[int]],
    no_joint: Sequence[Optional[int]],
    full: Sequence[Optional[int]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = list(range(len(labels)))
    width = 0.25
    series = [
        ("gross_baseline", gross, "#4e79a7", -width),
        ("ours_no_joint", no_joint, "#f28e2b", 0.0),
        ("ours_full", full, "#59a14f", width),
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.5))
    for name, values, color, offset in series:
        numeric = [0 if value is None else value for value in values]
        bars = ax.bar([i + offset for i in x], numeric, width=width, label=name, color=color)
        for bar, value in zip(bars, values):
            label = "NA" if value is None else str(value)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.05,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_story_figures(output_dir: Path, story_cases: Sequence[StoryCase], circuits: Sequence[CircuitSummary]) -> None:
    target_labels = [case.story_label for case in story_cases]
    gross = [case.row.gross_cost for case in story_cases]
    no_joint = [case.row.ours_no_joint_cost for case in story_cases]
    full = [case.row.ours_full_cost for case in story_cases]
    _plot_grouped_bars(
        labels=target_labels,
        gross=gross,
        no_joint=no_joint,
        full=full,
        title="Representative Target Costs",
        xlabel="Story target",
        ylabel="Cost",
        output_path=output_dir / "figure_story_costs.png",
    )

    circuit_labels = [circuit.circuit_name for circuit in circuits]
    gross_c = [circuit.gross_total_cost for circuit in circuits]
    no_joint_c = [circuit.ours_no_joint_total_cost for circuit in circuits]
    full_c = [circuit.ours_full_total_cost for circuit in circuits]
    _plot_grouped_bars(
        labels=circuit_labels,
        gross=gross_c,
        no_joint=no_joint_c,
        full=full_c,
        title="Toy Circuit Total Costs",
        xlabel="Toy circuit",
        ylabel="Total cost",
        output_path=output_dir / "figure_story_circuits.png",
    )


def generate_simple_story(input_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    src = Path(input_dir)
    out = Path(output_dir) if output_dir is not None else src
    per_target_path = src / "per_target.csv"
    if not per_target_path.exists():
        raise FileNotFoundError(f"missing per_target.csv in {src}")

    seed_rows = _load_per_target(per_target_path)
    ctx = _context_from_input_dir(src, seed_rows)
    story_cases, unreachable_note = _pick_story_cases(seed_rows, ctx)
    circuits = _build_circuits(story_cases)

    targets_csv = out / "simple_targets.csv"
    circuits_csv = out / "simple_circuits.csv"
    report_md = out / "story_report.md"

    _write_simple_targets_csv(targets_csv, story_cases)
    _write_simple_circuits_csv(circuits_csv, circuits)
    _write_story_markdown(report_md, story_cases, circuits, unreachable_note=unreachable_note)
    _write_story_figures(out, story_cases, circuits)

    return {
        "simple_targets_csv": targets_csv,
        "simple_circuits_csv": circuits_csv,
        "story_report_md": report_md,
        "figure_story_costs_png": out / "figure_story_costs.png",
        "figure_story_circuits_png": out / "figure_story_circuits.png",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a story-oriented report from no-budget benchmark output")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="benchmark results directory that contains per_target.csv (default: latest under benchmark/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory for story artifacts (default: same as --input-dir)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir or _latest_results_dir()
    output = generate_simple_story(input_dir=input_dir, output_dir=args.output_dir)
    print(f"wrote: {output['simple_targets_csv']}")
    print(f"wrote: {output['simple_circuits_csv']}")
    print(f"wrote: {output['story_report_md']}")
    print(f"wrote: {output['figure_story_costs_png']}")
    print(f"wrote: {output['figure_story_circuits_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

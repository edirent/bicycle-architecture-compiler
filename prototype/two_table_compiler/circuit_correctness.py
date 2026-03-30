from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from . import rules as rules_mod
from . import build_assets
from .io import DEFAULT_JOINT_TABLE_PATH, DEFAULT_LOCAL_TABLE_PATH, load_joint_table, load_local_table
from .rules import DEFAULT_CROSS_NATIVE_SHIFT, apply_p3_transitions, generate_p2_sources
from .search import SearchStatus, synthesize_search
from .state import (
    DEFAULT_ACCEPTING_HIDDEN_CLASSES,
    NON_IDENTITY_HEADS,
    N_QUBITS,
    State,
    Tail,
    hidden_from_head,
)
from .tables import (
    JointTable,
    LocalTable,
    NativeEntry,
    P2_RELATION_ANTICOMMUTE,
    P2_RELATION_ANY,
    P2_RELATION_COMMUTE,
    SCOPE_CROSS_LOGICAL_BLOCK_NATIVE,
    SCOPE_INTER_MODULE_BELL,
    SCOPE_INTRA_BLOCK_NATIVE,
)
from .targeting import classify_target

CIRCUIT_A_NAME = "all_direct_local_control"
CIRCUIT_B_NAME = "cross_logical_native_circuit"
CIRCUIT_C_NAME = "joint_required_circuit"
CIRCUIT_D_NAME = "mixed_p0_p2_p3_circuit"
CIRCUIT_E_NAME = "exhaustive_weight1_suite"
CIRCUIT_F_NAME = "exhaustive_weight2_suite"

EXPECTED_STATUS = SearchStatus.FOUND_OPTIMAL.value
CROSS_LOGICAL_TARGET_EXPR = "X1P⊗X1P'"

DEBUG_DIR = Path(__file__).resolve().parent / "debug"
FAILURE_CSV_PATH = DEBUG_DIR / "circuit_compile_failures.csv"
FAILURE_FIELDS = [
    "circuit_name",
    "target_index",
    "target_tail",
    "search_status",
    "expected_status",
    "first_family_if_any",
    "first_scope_if_any",
    "total_cost_if_any",
    "projected_best_cost_if_any",
    "replay_failed_at_step",
    "replay_failure_reason",
    "note",
]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    failing_step_index: int | None
    reason: str | None


@dataclass(frozen=True)
class FailureRow:
    circuit_name: str
    target_index: int
    target_tail: str
    search_status: str
    expected_status: str
    first_family_if_any: str
    first_scope_if_any: str
    total_cost_if_any: int | None
    projected_best_cost_if_any: int | None
    replay_failed_at_step: int | None
    replay_failure_reason: str
    note: str


@dataclass
class TargetCompileResult:
    circuit_name: str
    target_index: int
    target_input: str
    target_tail: str
    search_status: str
    expected_status: str
    first_family_if_any: str
    first_scope_if_any: str
    total_cost_if_any: int | None
    projected_best_cost_if_any: int | None
    remote_inter_module_count_if_any: int | None
    plan_has_p3: bool
    validation: ValidationResult
    passed: bool
    note: str


@dataclass
class CircuitCheckResult:
    circuit_name: str
    target_results: List[TargetCompileResult]
    failures: List[FailureRow]
    passed: bool


@dataclass
class CorrectnessReport:
    circuit_results: Dict[str, CircuitCheckResult]
    failures: List[FailureRow]
    failure_artifact_path: Path | None
    total_circuits_checked: int
    total_targets_checked: int
    non_optimal_target_count: int
    weight1_all_passed: bool
    weight2_all_passed: bool
    all_passed: bool


def _metadata_int(metadata: tuple[tuple[str, str], ...], key: str, default_value: int) -> int:
    md = dict(metadata)
    raw = md.get(key)
    if raw is None:
        return default_value
    return int(raw)


def _p0_expected_cost(entry: NativeEntry) -> int:
    base = 1 if entry.head == "I" else 2
    if entry.scope == SCOPE_CROSS_LOGICAL_BLOCK_NATIVE:
        shift = _metadata_int(entry.metadata, "native_shift_cost", DEFAULT_CROSS_NATIVE_SHIFT)
        return base + shift
    return base


def _resolve_target_tail(target: Tail | str, local_table: LocalTable) -> Tail:
    if isinstance(target, Tail):
        return target
    raw = str(target).strip()
    if not raw:
        raise ValueError("empty target expression")
    try:
        return Tail.from_str(raw)
    except ValueError:
        classification = classify_target(raw, local_table)
        if classification.canonical_tail is None:
            reason = classification.reason or f"cannot canonicalize target {raw!r}"
            raise ValueError(reason)
        return classification.canonical_tail


def _status_name(status: SearchStatus | str) -> str:
    return status.value if isinstance(status, SearchStatus) else str(status)


def _clear_transition_cache() -> None:
    cache = getattr(rules_mod, "_P3_EDGE_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()


def _synthesize_no_budget(
    target_tail: Tail,
    local_table: LocalTable,
    joint_table: JointTable,
    *,
    enable_p2: bool = True,
) -> object:
    _clear_transition_cache()
    result = synthesize_search(
        target_tail,
        local_table,
        joint_table,
        max_popped_states=None,
        enable_p2=enable_p2,
        p2_use_cache=True,
    )
    _clear_transition_cache()
    return result


def _validate_p0_step(step, entry_by_id: Dict[str, NativeEntry]) -> str | None:
    if step.src is not None:
        return "P0 step must have src=None"
    if len(step.native_ids) != 1:
        return "P0 step must reference exactly one native_id"
    native_id = step.native_ids[0]
    entry = entry_by_id.get(native_id)
    if entry is None:
        return f"P0 native_id not found in LocalTable: {native_id}"
    if entry.scope not in (SCOPE_INTRA_BLOCK_NATIVE, SCOPE_CROSS_LOGICAL_BLOCK_NATIVE):
        return f"P0 native scope must be intra/cross logical, got {entry.scope}"
    expected_dst = State(hidden=hidden_from_head(entry.head), tail=entry.tail)
    if step.dst != expected_dst:
        return "P0 dst state does not match referenced native entry"
    expected_cost = _p0_expected_cost(entry)
    if step.delta_cost != expected_cost:
        return f"P0 delta_cost mismatch: expected {expected_cost}, got {step.delta_cost}"
    if step.scope != entry.scope:
        return f"P0 scope mismatch: expected {entry.scope}, got {step.scope}"
    if step.input_native_ids and tuple(step.input_native_ids) != (native_id,):
        return "P0 input_native_ids mismatch"
    if step.input_scopes and tuple(step.input_scopes) != (entry.scope,):
        return "P0 input_scopes mismatch"
    return None


def _validate_p2_step(step, entry_by_id: Dict[str, NativeEntry], joint_table: JointTable) -> str | None:
    if step.src is not None:
        return "P2 source step must have src=None"
    if len(step.native_ids) != 2:
        return "P2 step must reference exactly two native_ids"
    lhs_id, rhs_id = step.native_ids
    if lhs_id == rhs_id:
        return "P2 step must reference two distinct native_ids"
    lhs = entry_by_id.get(lhs_id)
    rhs = entry_by_id.get(rhs_id)
    if lhs is None or rhs is None:
        return f"P2 native_ids not found in LocalTable: {lhs_id}, {rhs_id}"
    rule_name = step.applied_rule_name or step.rule_name
    rule = next((item for item in joint_table.p2_rules if item.name == rule_name), None)
    if rule is None:
        return f"P2 rule not found in JointTable: {rule_name!r}"

    relation = P2_RELATION_COMMUTE if lhs.tail.commute(rhs.tail) else P2_RELATION_ANTICOMMUTE
    if lhs.scope not in rule.allowed_input_scopes:
        return f"P2 lhs scope {lhs.scope} not allowed by rule {rule.name}"
    if rhs.scope not in rule.allowed_input_scopes:
        return f"P2 rhs scope {rhs.scope} not allowed by rule {rule.name}"
    if rule.lhs_head is not None and lhs.head != rule.lhs_head:
        return f"P2 lhs head mismatch for rule {rule.name}"
    if rule.rhs_head is not None and rhs.head != rule.rhs_head:
        return f"P2 rhs head mismatch for rule {rule.name}"
    if rule.require_same_head and lhs.head != rhs.head:
        return f"P2 rule {rule.name} requires same head"
    if rule.require_nonidentity_heads:
        if lhs.head not in NON_IDENTITY_HEADS or rhs.head not in NON_IDENTITY_HEADS:
            return f"P2 rule {rule.name} requires non-identity heads"
    if rule.required_relation != P2_RELATION_ANY and rule.required_relation != relation:
        return f"P2 relation mismatch for rule {rule.name}: expected {rule.required_relation}, got {relation}"
    if hidden_from_head(lhs.head) != rule.out_hidden:
        return f"P2 out_hidden mismatch for rule {rule.name}"

    expected_dst = State(hidden=rule.out_hidden, tail=lhs.tail.multiply(rhs.tail))
    if step.dst != expected_dst:
        return f"P2 dst mismatch for rule {rule.name}"
    expected_scope = (
        SCOPE_CROSS_LOGICAL_BLOCK_NATIVE
        if SCOPE_CROSS_LOGICAL_BLOCK_NATIVE in (lhs.scope, rhs.scope)
        else SCOPE_INTRA_BLOCK_NATIVE
    )
    if step.scope != expected_scope:
        return f"P2 scope mismatch: expected {expected_scope}, got {step.scope}"
    if step.delta_cost != rule.total_cost:
        return f"P2 total_cost mismatch: expected {rule.total_cost}, got {step.delta_cost}"
    if step.input_native_ids and tuple(step.input_native_ids) != (lhs_id, rhs_id):
        return "P2 input_native_ids mismatch"
    if step.input_scopes and tuple(step.input_scopes) != (lhs.scope, rhs.scope):
        return "P2 input_scopes mismatch"
    return None


def _validate_p3_step(step, entry_by_id: Dict[str, NativeEntry], joint_table: JointTable) -> str | None:
    if step.src is None:
        return "P3 step must have src state"
    if len(step.native_ids) != 1:
        return "P3 step must reference exactly one native_id"
    native_id = step.native_ids[0]
    axis = entry_by_id.get(native_id)
    if axis is None:
        return f"P3 native_id not found in LocalTable: {native_id}"
    candidates = apply_p3_transitions(step.src, axis, joint_table)
    if not candidates:
        return f"P3 transitions empty for src+axis ({native_id})"

    rule_name = step.applied_rule_name or step.rule_name
    matched = False
    for candidate in candidates:
        candidate_rule = candidate.applied_rule_name or candidate.rule_name
        if candidate_rule != rule_name:
            continue
        if candidate.dst != step.dst:
            continue
        if candidate.delta_cost != step.delta_cost:
            continue
        matched = True
        break
    if not matched:
        return f"P3 step is not reproducible by apply_p3_transitions for rule {rule_name!r}"

    if step.scope != axis.scope:
        return f"P3 scope mismatch: expected {axis.scope}, got {step.scope}"
    if step.input_native_ids and tuple(step.input_native_ids) != (native_id,):
        return "P3 input_native_ids mismatch"
    if step.input_scopes and tuple(step.input_scopes) != (axis.scope,):
        return "P3 input_scopes mismatch"
    return None


def validate_synthesis_plan(
    plan,
    target_tail: Tail,
    local_table: LocalTable,
    joint_table: JointTable,
) -> ValidationResult:
    if plan is None:
        return ValidationResult(ok=False, failing_step_index=None, reason="missing synthesis plan")
    steps = list(getattr(plan, "steps", []))
    if not steps:
        return ValidationResult(ok=False, failing_step_index=None, reason="plan.steps is empty")

    entry_by_id: Dict[str, NativeEntry] = {entry.native_id: entry for entry in local_table.entries}
    accepting = set(DEFAULT_ACCEPTING_HIDDEN_CLASSES)
    total_cost = 0

    for index, step in enumerate(steps):
        total_cost += int(step.delta_cost)
        if index == 0:
            if step.family not in ("P0", "P2"):
                return ValidationResult(
                    ok=False,
                    failing_step_index=index,
                    reason="first step family must be P0/P2 source",
                )
            if step.src is not None:
                return ValidationResult(
                    ok=False,
                    failing_step_index=index,
                    reason="first source step must have src=None",
                )
        else:
            prev_step = steps[index - 1]
            if step.family != "P3":
                return ValidationResult(
                    ok=False,
                    failing_step_index=index,
                    reason="non-first step must be P3 transition",
                )
            if step.src is None:
                return ValidationResult(
                    ok=False,
                    failing_step_index=index,
                    reason="P3 step missing src state",
                )
            if step.src != prev_step.dst:
                return ValidationResult(
                    ok=False,
                    failing_step_index=index,
                    reason="step chain is not continuous (step.src != previous step.dst)",
                )

        if step.family == "P0":
            reason = _validate_p0_step(step, entry_by_id)
        elif step.family == "P2":
            reason = _validate_p2_step(step, entry_by_id, joint_table)
        elif step.family == "P3":
            reason = _validate_p3_step(step, entry_by_id, joint_table)
        else:
            reason = f"unknown step family {step.family!r}"
        if reason is not None:
            return ValidationResult(ok=False, failing_step_index=index, reason=reason)

    final_state = steps[-1].dst
    if final_state.tail != target_tail:
        return ValidationResult(ok=False, failing_step_index=None, reason="final state tail does not match target")
    if final_state.hidden not in accepting:
        return ValidationResult(ok=False, failing_step_index=None, reason="final state hidden is not accepting")
    if getattr(plan, "target_tail", None) != target_tail:
        return ValidationResult(ok=False, failing_step_index=None, reason="plan.target_tail mismatch")
    if getattr(plan, "best_state", None) != final_state:
        return ValidationResult(ok=False, failing_step_index=None, reason="plan.best_state mismatch")
    if getattr(plan, "best_hidden", None) != final_state.hidden:
        return ValidationResult(ok=False, failing_step_index=None, reason="plan.best_hidden mismatch")
    if int(getattr(plan, "total_cost", -1)) != total_cost:
        return ValidationResult(ok=False, failing_step_index=None, reason="total_cost != sum(delta_cost)")
    if int(getattr(plan, "projected_best_cost", -1)) != int(getattr(plan, "total_cost", -2)):
        return ValidationResult(
            ok=False,
            failing_step_index=None,
            reason="projected_best_cost must equal total_cost for FOUND_OPTIMAL witness",
        )
    expected_remote_count = sum(1 for step in steps if step.scope == SCOPE_INTER_MODULE_BELL)
    if int(getattr(plan, "remote_inter_module_count", -1)) != expected_remote_count:
        return ValidationResult(ok=False, failing_step_index=None, reason="remote_inter_module_count mismatch")

    return ValidationResult(ok=True, failing_step_index=None, reason=None)


def _weight1_targets() -> List[str]:
    targets: List[str] = []
    for idx in range(N_QUBITS):
        for pauli in ("X", "Y", "Z"):
            symbols = ["I"] * N_QUBITS
            symbols[idx] = pauli
            targets.append("".join(symbols))
    return targets


def _weight2_targets() -> List[str]:
    targets: List[str] = []
    for lhs, rhs in combinations(range(N_QUBITS), 2):
        for pauli_lhs in ("X", "Y", "Z"):
            for pauli_rhs in ("X", "Y", "Z"):
                symbols = ["I"] * N_QUBITS
                symbols[lhs] = pauli_lhs
                symbols[rhs] = pauli_rhs
                targets.append("".join(symbols))
    return targets


def _load_cached_tables() -> tuple[LocalTable, JointTable]:
    if DEFAULT_LOCAL_TABLE_PATH.exists() and DEFAULT_JOINT_TABLE_PATH.exists():
        return load_local_table(DEFAULT_LOCAL_TABLE_PATH), load_joint_table(DEFAULT_JOINT_TABLE_PATH)
    local, joint, _ = build_assets.ensure_cached_assets(seed=7, count=20, max_weight_filter=1)
    return local, joint


def _find_direct_local_targets(local_table: LocalTable, joint_table: JointTable, count: int) -> List[str]:
    selected: List[str] = []
    seen: set[str] = set()
    for entry in local_table.entries:
        if entry.scope != SCOPE_INTRA_BLOCK_NATIVE:
            continue
        if entry.head not in NON_IDENTITY_HEADS:
            continue
        target = entry.tail.compact()
        if target in seen:
            continue
        seen.add(target)
        result = _synthesize_no_budget(entry.tail, local_table, joint_table, enable_p2=True)
        if result.status != SearchStatus.FOUND_OPTIMAL or result.plan is None:
            continue
        if result.plan.steps[0].family != "P0":
            continue
        if result.plan.steps[0].scope != SCOPE_INTRA_BLOCK_NATIVE:
            continue
        selected.append(target)
        if len(selected) >= count:
            return selected
    raise RuntimeError(f"cannot find {count} direct_local control targets from cached LocalTable")


def _find_joint_required_target(local_table: LocalTable, joint_table: JointTable) -> str:
    candidates = sorted({step.dst.tail.compact() for step in generate_p2_sources(local_table, joint_table)})
    for target in candidates:
        tail = Tail.from_str(target)
        full = _synthesize_no_budget(tail, local_table, joint_table, enable_p2=True)
        if full.status != SearchStatus.FOUND_OPTIMAL or full.plan is None:
            continue
        if full.plan.steps[0].family != "P2":
            continue
        no_joint = _synthesize_no_budget(tail, local_table, joint_table, enable_p2=False)
        if no_joint.status != SearchStatus.FOUND_OPTIMAL or no_joint.plan is None:
            return target
        if no_joint.plan.total_cost > full.plan.total_cost:
            return target
    raise RuntimeError("cannot find a joint-required target whose optimal witness starts from P2")


def _find_p3_target(local_table: LocalTable, joint_table: JointTable) -> str:
    for target in _weight1_targets():
        tail = Tail.from_str(target)
        result = _synthesize_no_budget(tail, local_table, joint_table, enable_p2=True)
        if result.status != SearchStatus.FOUND_OPTIMAL or result.plan is None:
            continue
        if any(step.family == "P3" for step in result.plan.steps):
            return target
    raise RuntimeError("cannot find a reachable target with a P3 transition in the witness plan")


def _as_failure_row(
    *,
    circuit_name: str,
    target_index: int,
    target_tail: str,
    search_status: str,
    first_family_if_any: str,
    first_scope_if_any: str,
    total_cost_if_any: int | None,
    projected_best_cost_if_any: int | None,
    replay_failed_at_step: int | None,
    replay_failure_reason: str,
    note: str,
) -> FailureRow:
    return FailureRow(
        circuit_name=circuit_name,
        target_index=target_index,
        target_tail=target_tail,
        search_status=search_status,
        expected_status=EXPECTED_STATUS,
        first_family_if_any=first_family_if_any,
        first_scope_if_any=first_scope_if_any,
        total_cost_if_any=total_cost_if_any,
        projected_best_cost_if_any=projected_best_cost_if_any,
        replay_failed_at_step=replay_failed_at_step,
        replay_failure_reason=replay_failure_reason,
        note=note,
    )


def _compile_target(
    *,
    circuit_name: str,
    target_index: int,
    target: Tail | str,
    local_table: LocalTable,
    joint_table: JointTable,
) -> tuple[TargetCompileResult, List[FailureRow]]:
    failures: List[FailureRow] = []
    raw_target = target.compact() if isinstance(target, Tail) else str(target)

    try:
        target_tail = _resolve_target_tail(target, local_table)
    except ValueError as exc:
        result = TargetCompileResult(
            circuit_name=circuit_name,
            target_index=target_index,
            target_input=raw_target,
            target_tail="<canonicalize_failed>",
            search_status="TARGET_CANONICALIZATION_FAILED",
            expected_status=EXPECTED_STATUS,
            first_family_if_any="",
            first_scope_if_any="",
            total_cost_if_any=None,
            projected_best_cost_if_any=None,
            remote_inter_module_count_if_any=None,
            plan_has_p3=False,
            validation=ValidationResult(ok=False, failing_step_index=None, reason=str(exc)),
            passed=False,
            note="target canonicalization failed",
        )
        failures.append(
            _as_failure_row(
                circuit_name=circuit_name,
                target_index=target_index,
                target_tail=result.target_tail,
                search_status=result.search_status,
                first_family_if_any=result.first_family_if_any,
                first_scope_if_any=result.first_scope_if_any,
                total_cost_if_any=result.total_cost_if_any,
                projected_best_cost_if_any=result.projected_best_cost_if_any,
                replay_failed_at_step=None,
                replay_failure_reason=str(exc),
                note=result.note,
            )
        )
        return result, failures

    search_result = _synthesize_no_budget(target_tail, local_table, joint_table, enable_p2=True)
    status = _status_name(search_result.status)
    plan = search_result.plan
    first_family = ""
    first_scope = ""
    total_cost = None
    projected_best = None
    remote_count = None
    plan_has_p3 = False
    if plan is not None and plan.steps:
        first_family = plan.steps[0].family
        first_scope = plan.steps[0].scope
        total_cost = int(plan.total_cost)
        projected_best = int(plan.projected_best_cost)
        remote_count = int(plan.remote_inter_module_count)
        plan_has_p3 = any(step.family == "P3" for step in plan.steps)

    if status != EXPECTED_STATUS:
        validation = ValidationResult(ok=False, failing_step_index=None, reason=f"unexpected status {status}")
        result = TargetCompileResult(
            circuit_name=circuit_name,
            target_index=target_index,
            target_input=raw_target,
            target_tail=target_tail.compact(),
            search_status=status,
            expected_status=EXPECTED_STATUS,
            first_family_if_any=first_family,
            first_scope_if_any=first_scope,
            total_cost_if_any=total_cost,
            projected_best_cost_if_any=projected_best,
            remote_inter_module_count_if_any=remote_count,
            plan_has_p3=plan_has_p3,
            validation=validation,
            passed=False,
            note="search status is not FOUND_OPTIMAL under no-budget",
        )
        failures.append(
            _as_failure_row(
                circuit_name=circuit_name,
                target_index=target_index,
                target_tail=result.target_tail,
                search_status=result.search_status,
                first_family_if_any=result.first_family_if_any,
                first_scope_if_any=result.first_scope_if_any,
                total_cost_if_any=result.total_cost_if_any,
                projected_best_cost_if_any=result.projected_best_cost_if_any,
                replay_failed_at_step=result.validation.failing_step_index,
                replay_failure_reason=result.validation.reason or "",
                note=result.note,
            )
        )
        return result, failures

    validation = validate_synthesis_plan(plan, target_tail, local_table, joint_table)
    passed = validation.ok
    result = TargetCompileResult(
        circuit_name=circuit_name,
        target_index=target_index,
        target_input=raw_target,
        target_tail=target_tail.compact(),
        search_status=status,
        expected_status=EXPECTED_STATUS,
        first_family_if_any=first_family,
        first_scope_if_any=first_scope,
        total_cost_if_any=total_cost,
        projected_best_cost_if_any=projected_best,
        remote_inter_module_count_if_any=remote_count,
        plan_has_p3=plan_has_p3,
        validation=validation,
        passed=passed,
        note="" if passed else "plan replay validation failed",
    )
    if not passed:
        failures.append(
            _as_failure_row(
                circuit_name=circuit_name,
                target_index=target_index,
                target_tail=result.target_tail,
                search_status=result.search_status,
                first_family_if_any=result.first_family_if_any,
                first_scope_if_any=result.first_scope_if_any,
                total_cost_if_any=result.total_cost_if_any,
                projected_best_cost_if_any=result.projected_best_cost_if_any,
                replay_failed_at_step=validation.failing_step_index,
                replay_failure_reason=validation.reason or "",
                note=result.note,
            )
        )
    return result, failures


def _run_circuit(
    *,
    circuit_name: str,
    targets: Sequence[Tail | str],
    local_table: LocalTable,
    joint_table: JointTable,
) -> CircuitCheckResult:
    target_results: List[TargetCompileResult] = []
    failures: List[FailureRow] = []
    for idx, target in enumerate(targets):
        target_result, target_failures = _compile_target(
            circuit_name=circuit_name,
            target_index=idx,
            target=target,
            local_table=local_table,
            joint_table=joint_table,
        )
        target_results.append(target_result)
        failures.extend(target_failures)
    return CircuitCheckResult(
        circuit_name=circuit_name,
        target_results=target_results,
        failures=failures,
        passed=len(failures) == 0,
    )


def _add_circuit_level_failure(
    *,
    circuit: CircuitCheckResult,
    message: str,
    target_index: int = -1,
    target_tail: str = "",
) -> None:
    circuit.failures.append(
        _as_failure_row(
            circuit_name=circuit.circuit_name,
            target_index=target_index,
            target_tail=target_tail,
            search_status="CIRCUIT_LEVEL_CHECK_FAILED",
            first_family_if_any="",
            first_scope_if_any="",
            total_cost_if_any=None,
            projected_best_cost_if_any=None,
            replay_failed_at_step=None,
            replay_failure_reason=message,
            note=message,
        )
    )
    circuit.passed = False


def _enforce_circuit_constraints(
    *,
    circuits: Dict[str, CircuitCheckResult],
    joint_target: str,
    p3_target: str,
) -> None:
    circuit_a = circuits[CIRCUIT_A_NAME]
    for result in circuit_a.target_results:
        if result.first_family_if_any != "P0":
            _add_circuit_level_failure(
                circuit=circuit_a,
                message="Circuit A requires first family P0 for all targets",
                target_index=result.target_index,
                target_tail=result.target_tail,
            )
        if result.first_scope_if_any != SCOPE_INTRA_BLOCK_NATIVE:
            _add_circuit_level_failure(
                circuit=circuit_a,
                message="Circuit A requires first scope intra_block_native",
                target_index=result.target_index,
                target_tail=result.target_tail,
            )

    circuit_b = circuits[CIRCUIT_B_NAME]
    if not circuit_b.target_results:
        _add_circuit_level_failure(circuit=circuit_b, message="Circuit B has no targets")
    else:
        cross = circuit_b.target_results[0]
        if cross.first_family_if_any != "P0":
            _add_circuit_level_failure(
                circuit=circuit_b,
                message="Circuit B cross-logical target must start with P0",
                target_index=cross.target_index,
                target_tail=cross.target_tail,
            )
        if cross.first_scope_if_any != SCOPE_CROSS_LOGICAL_BLOCK_NATIVE:
            _add_circuit_level_failure(
                circuit=circuit_b,
                message="Circuit B cross-logical target must use cross_logical_block_native scope",
                target_index=cross.target_index,
                target_tail=cross.target_tail,
            )
        if cross.remote_inter_module_count_if_any not in (None, 0):
            _add_circuit_level_failure(
                circuit=circuit_b,
                message="Circuit B cross-logical target must not consume inter-module Bell",
                target_index=cross.target_index,
                target_tail=cross.target_tail,
            )

    circuit_c = circuits[CIRCUIT_C_NAME]
    if not circuit_c.target_results:
        _add_circuit_level_failure(circuit=circuit_c, message="Circuit C has no targets")
    else:
        joint = circuit_c.target_results[0]
        if joint.target_tail != joint_target:
            _add_circuit_level_failure(
                circuit=circuit_c,
                message="Circuit C first target is not the selected joint-required target",
                target_index=joint.target_index,
                target_tail=joint.target_tail,
            )
        if joint.first_family_if_any != "P2":
            _add_circuit_level_failure(
                circuit=circuit_c,
                message="Circuit C joint target must start with P2",
                target_index=joint.target_index,
                target_tail=joint.target_tail,
            )

    circuit_d = circuits[CIRCUIT_D_NAME]
    first_families = {item.first_family_if_any for item in circuit_d.target_results}
    if "P0" not in first_families:
        _add_circuit_level_failure(circuit=circuit_d, message="Circuit D must include at least one first-step P0")
    if "P2" not in first_families:
        _add_circuit_level_failure(circuit=circuit_d, message="Circuit D must include at least one first-step P2")
    if not any(item.plan_has_p3 for item in circuit_d.target_results):
        _add_circuit_level_failure(circuit=circuit_d, message="Circuit D must include at least one plan containing P3")
    if p3_target not in {item.target_tail for item in circuit_d.target_results}:
        _add_circuit_level_failure(circuit=circuit_d, message="Circuit D is missing selected P3-required target")

    circuit_e = circuits[CIRCUIT_E_NAME]
    if len(circuit_e.target_results) != 3 * N_QUBITS:
        _add_circuit_level_failure(
            circuit=circuit_e,
            message=f"Circuit E must cover all {3 * N_QUBITS} weight-1 targets",
        )

    circuit_f = circuits[CIRCUIT_F_NAME]
    expected_weight2 = 9 * (N_QUBITS * (N_QUBITS - 1) // 2)
    if len(circuit_f.target_results) != expected_weight2:
        _add_circuit_level_failure(
            circuit=circuit_f,
            message=f"Circuit F must cover all weight-2 targets (expected {expected_weight2})",
        )


def _write_failure_csv(rows: Sequence[FailureRow], path: Path = FAILURE_CSV_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            writer.writerow(payload)
    return path


def run_correctness_check(*, write_failure_artifact: bool = True) -> CorrectnessReport:
    local_table, joint_table = _load_cached_tables()

    direct_targets = _find_direct_local_targets(local_table, joint_table, count=5)
    joint_target = _find_joint_required_target(local_table, joint_table)
    p3_target = _find_p3_target(local_table, joint_table)
    weight1_targets = _weight1_targets()
    weight2_targets = _weight2_targets()

    circuit_defs: List[tuple[str, List[Tail | str]]] = [
        (CIRCUIT_A_NAME, [direct_targets[0], direct_targets[1], direct_targets[2], direct_targets[3]]),
        (CIRCUIT_B_NAME, [CROSS_LOGICAL_TARGET_EXPR, direct_targets[0], direct_targets[1]]),
        (CIRCUIT_C_NAME, [joint_target, direct_targets[2], direct_targets[3]]),
        (CIRCUIT_D_NAME, [direct_targets[0], CROSS_LOGICAL_TARGET_EXPR, joint_target, p3_target]),
        (CIRCUIT_E_NAME, list(weight1_targets)),
        (CIRCUIT_F_NAME, list(weight2_targets)),
    ]

    circuits: Dict[str, CircuitCheckResult] = {}
    for circuit_name, targets in circuit_defs:
        circuits[circuit_name] = _run_circuit(
            circuit_name=circuit_name,
            targets=targets,
            local_table=local_table,
            joint_table=joint_table,
        )

    _enforce_circuit_constraints(circuits=circuits, joint_target=joint_target, p3_target=p3_target)

    all_failures: List[FailureRow] = []
    non_optimal_count = 0
    total_targets = 0
    for circuit in circuits.values():
        total_targets += len(circuit.target_results)
        for target_result in circuit.target_results:
            if target_result.search_status != EXPECTED_STATUS:
                non_optimal_count += 1
        all_failures.extend(circuit.failures)
        circuit.passed = len(circuit.failures) == 0

    failure_artifact_path: Path | None = None
    if all_failures:
        if write_failure_artifact:
            failure_artifact_path = _write_failure_csv(all_failures, FAILURE_CSV_PATH)
    else:
        if FAILURE_CSV_PATH.exists():
            FAILURE_CSV_PATH.unlink()

    weight1_passed = circuits[CIRCUIT_E_NAME].passed
    weight2_passed = circuits[CIRCUIT_F_NAME].passed
    all_passed = all(circuit.passed for circuit in circuits.values())

    return CorrectnessReport(
        circuit_results=circuits,
        failures=all_failures,
        failure_artifact_path=failure_artifact_path,
        total_circuits_checked=len(circuit_defs),
        total_targets_checked=total_targets,
        non_optimal_target_count=non_optimal_count,
        weight1_all_passed=weight1_passed,
        weight2_all_passed=weight2_passed,
        all_passed=all_passed,
    )


def _print_summary(report: CorrectnessReport) -> None:
    print(f"total circuits checked: {report.total_circuits_checked}")
    print(f"total targets checked: {report.total_targets_checked}")
    if report.all_passed:
        print("result: all passed")
        return
    print(f"result: failed ({len(report.failures)} failures)")
    if report.failure_artifact_path is not None:
        print(f"failure artifact: {report.failure_artifact_path}")
    preview_count = min(10, len(report.failures))
    for row in report.failures[:preview_count]:
        print(
            f"failure[{row.circuit_name}#{row.target_index}] target={row.target_tail} "
            f"status={row.search_status} note={row.note}"
        )


def main() -> int:
    report = run_correctness_check(write_failure_artifact=True)
    _print_summary(report)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

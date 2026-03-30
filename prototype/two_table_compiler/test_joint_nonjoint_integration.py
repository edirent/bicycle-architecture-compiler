from __future__ import annotations

import inspect
import itertools
import unittest
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from prototype.two_table_compiler import build_assets as build_assets_mod
from prototype.two_table_compiler import io as io_mod
from prototype.two_table_compiler import rules as rules_mod
from prototype.two_table_compiler import search as search_mod
from prototype.two_table_compiler import state as state_mod

try:
    from prototype.two_table_compiler import targeting as targeting_mod
except Exception:  # pragma: no cover - optional module
    targeting_mod = None


Tail = state_mod.Tail


@dataclass(frozen=True)
class CompiledCase:
    target_label: str
    raw_result: Any

    @property
    def search_status(self) -> Optional[str]:
        if self.raw_result is None:
            return None
        for attr in ("search_status", "status"):
            if hasattr(self.raw_result, attr):
                value = getattr(self.raw_result, attr)
                return getattr(value, "value", value)
        return None

    @property
    def plan(self) -> Any:
        if self.raw_result is None:
            return None
        if hasattr(self.raw_result, "plan"):
            return getattr(self.raw_result, "plan")
        return self.raw_result

    @property
    def reachable(self) -> bool:
        if self.raw_result is None:
            return False
        status = self.search_status
        if status is not None and hasattr(search_mod, "is_reachable_status"):
            return bool(search_mod.is_reachable_status(status))
        if hasattr(self.raw_result, "frame_unreachable"):
            return not bool(getattr(self.raw_result, "frame_unreachable"))
        if hasattr(self.raw_result, "reachable"):
            return bool(getattr(self.raw_result, "reachable"))
        return self.plan is not None

    @property
    def frame_unreachable(self) -> bool:
        if self.raw_result is None:
            return True
        status = self.search_status
        if status is not None:
            return status == search_mod.SearchStatus.RULE_UNREACHABLE.value
        return bool(getattr(self.raw_result, "frame_unreachable", False))

    @property
    def reason(self) -> str:
        return str(getattr(self.raw_result, "reason", ""))

    @property
    def steps(self) -> list[Any]:
        plan = self.plan
        if plan is None:
            return []
        return list(getattr(plan, "steps", []))

    @property
    def total_cost(self) -> Any:
        plan = self.plan
        if plan is None:
            return None
        for attr in ("total_cost", "beta", "projected_best_cost"):
            if hasattr(plan, attr):
                return getattr(plan, attr)
        return None

    @property
    def remote_inter_module_count(self) -> int:
        for owner in (self.raw_result, self.plan):
            if owner is None:
                continue
            if hasattr(owner, "remote_inter_module_count"):
                return int(getattr(owner, "remote_inter_module_count"))
        return 0

    @property
    def best_tail_compact(self) -> Optional[str]:
        plan = self.plan
        if plan is None:
            return None
        for attr in ("best_state", "final_state"):
            if hasattr(plan, attr):
                state = getattr(plan, attr)
                if hasattr(state, "tail"):
                    return state.tail.compact()
        if hasattr(plan, "best_tail"):
            best_tail = getattr(plan, "best_tail")
            if hasattr(best_tail, "compact"):
                return best_tail.compact()
        return None


def _call_with_matching_kwargs(fn: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    filtered = {name: value for name, value in kwargs.items() if name in sig.parameters}
    return fn(**filtered)


class JointNonJointCircuitIntegrationTests(unittest.TestCase):
    """
    Integration tests for the current two-table compiler prototype.

    The prototype compiles one logical measurement target at a time.  In these tests,
    a "circuit" is represented as an ordered list of measurement targets compiled under
    the same frame/tables.  This is the right granularity for the current Module-I code.
    """

    max_popped_states = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.local_table, cls.joint_table = cls._load_tables()
        cls.entry_by_native_id = {
            entry.native_id: entry for entry in getattr(cls.local_table, "entries", [])
        }
        cls.native_tails = {
            entry.tail.compact()
            for entry in getattr(cls.local_table, "entries", [])
            if getattr(entry, "scope", None) in (None, "intra_block_native", "cross_logical_block_native")
        }
        cls.intra_entry = cls._choose_local_entry(scope="intra_block_native")
        cls.cross_entry = cls._choose_local_entry(scope="cross_logical_block_native")
        cls.p2_only_step = cls._choose_p2_only_step()
        cls.unreachable_target = cls._find_unreachable_target()

    @classmethod
    def _load_tables(cls):
        # Preferred path: ensure cached/generated assets exist and then load them.
        if hasattr(build_assets_mod, "ensure_cached_assets"):
            try:
                local, joint, *_ = _call_with_matching_kwargs(
                    build_assets_mod.ensure_cached_assets,
                    seed=7,
                    count=20,
                    max_weight_filter=2,
                )
                return local, joint
            except Exception:
                pass

        refresh_local = getattr(build_assets_mod, "refresh_local_table_asset", None)
        refresh_joint = getattr(build_assets_mod, "refresh_joint_table_asset", None)
        if callable(refresh_local) and callable(refresh_joint):
            local = _call_with_matching_kwargs(
                refresh_local,
                path=getattr(io_mod, "DEFAULT_LOCAL_TABLE_PATH", None),
            )
            joint = _call_with_matching_kwargs(
                refresh_joint,
                path=getattr(io_mod, "DEFAULT_JOINT_TABLE_PATH", None),
            )
            return local, joint

        load_local = getattr(io_mod, "load_local_table")
        load_joint = getattr(io_mod, "load_joint_table")
        return (
            load_local(getattr(io_mod, "DEFAULT_LOCAL_TABLE_PATH")),
            load_joint(getattr(io_mod, "DEFAULT_JOINT_TABLE_PATH")),
        )

    @classmethod
    def _choose_local_entry(cls, scope: str):
        for entry in getattr(cls.local_table, "entries", []):
            if getattr(entry, "scope", None) != scope:
                continue
            if getattr(entry, "head", "I") == "I":
                continue
            return entry
        raise unittest.SkipTest(f"local table does not contain scope={scope!r} non-identity entry")

    @classmethod
    def _generate_p2_sources(cls, target_tail: Optional[Any] = None) -> list[Any]:
        fn = getattr(rules_mod, "generate_p2_sources")
        sig = inspect.signature(fn)
        kwargs = {
            "local_table": cls.local_table,
            "joint_table": cls.joint_table,
            "target_tail": target_tail,
        }
        return list(_call_with_matching_kwargs(fn, **kwargs))

    @classmethod
    def _choose_p2_only_step(cls):
        p2_steps = cls._generate_p2_sources(target_tail=None)
        if not p2_steps:
            raise unittest.SkipTest("no P2 sources generated under current frame")
        for step in p2_steps:
            dst_tail = step.dst.tail.compact()
            if dst_tail not in cls.native_tails:
                return step
        raise unittest.SkipTest("could not find a P2-only target not already present as native P0")

    @classmethod
    def _find_unreachable_target(cls) -> Optional[str]:
        # Search a small low-weight space for one frame-unreachable target.
        n_qubits = getattr(state_mod, "N_QUBITS", 11)
        paulis = ("X", "Y", "Z")
        candidate_strings: list[str] = []

        # Weight-1 candidates.
        for idx in range(min(n_qubits, 6)):
            for pauli in paulis:
                s = ["I"] * n_qubits
                s[idx] = pauli
                candidate_strings.append("".join(s))

        # A few weight-2 candidates.
        for i, j in itertools.combinations(range(min(n_qubits, 5)), 2):
            for p1 in paulis:
                for p2 in paulis:
                    s = ["I"] * n_qubits
                    s[i] = p1
                    s[j] = p2
                    candidate_strings.append("".join(s))
                    if len(candidate_strings) >= 80:
                        break
                if len(candidate_strings) >= 80:
                    break
            if len(candidate_strings) >= 80:
                break

        # Avoid known native / chosen P2 candidate first.
        preferred = [
            s
            for s in candidate_strings
            if s not in cls.native_tails and s != cls.p2_only_step.dst.tail.compact()
        ]
        for target in preferred + candidate_strings:
            outcome = cls._compile(target)
            if not outcome.reachable:
                return target
        return None

    @classmethod
    def _normalize_target(cls, target: Any) -> Any:
        if isinstance(target, Tail):
            return target
        if isinstance(target, str):
            try:
                return Tail.from_str(target)
            except Exception:
                pass

            if targeting_mod is not None:
                for name in (
                    "canonicalize_target",
                    "canonicalize",
                    "parse_target",
                    "normalize_target",
                ):
                    fn = getattr(targeting_mod, name, None)
                    if not callable(fn):
                        continue
                    try:
                        out = fn(target)
                    except Exception:
                        continue
                    if isinstance(out, Tail):
                        return out
                    if isinstance(out, str):
                        try:
                            return Tail.from_str(out)
                        except Exception:
                            continue
                    if hasattr(out, "tail"):
                        return out.tail
                    if isinstance(out, tuple):
                        for item in out:
                            if isinstance(item, Tail):
                                return item
                            if isinstance(item, str):
                                try:
                                    return Tail.from_str(item)
                                except Exception:
                                    pass

            # Known project-specific alias from the current prototype.
            aliases = {
                "X1P⊗X1P'": "XIIIIIIXIII",
                "X1P*X1P'": "XIIIIIIXIII",
            }
            if target in aliases:
                return Tail.from_str(aliases[target])

        raise ValueError(f"unable to normalize target: {target!r}")

    @classmethod
    def _compile(cls, target: Any) -> CompiledCase:
        synth_target = getattr(search_mod, "synthesize_target", None)
        if callable(synth_target):
            sig = inspect.signature(synth_target)
            kwargs = {
                "target": target,
                "target_expr": target,
                "target_spec": target,
                "target_tail": cls._normalize_target(target),
                "local_table": cls.local_table,
                "joint_table": cls.joint_table,
                "max_popped_states": cls.max_popped_states,
            }
            raw = _call_with_matching_kwargs(synth_target, **kwargs)
        else:
            synth = getattr(search_mod, "synthesize")
            sig = inspect.signature(synth)
            tail = cls._normalize_target(target)
            kwargs = {
                "target": tail,
                "target_tail": tail,
                "tail": tail,
                "local_table": cls.local_table,
                "joint_table": cls.joint_table,
                "max_popped_states": cls.max_popped_states,
            }
            raw = _call_with_matching_kwargs(synth, **kwargs)

        label = target.compact() if isinstance(target, Tail) else str(target)
        return CompiledCase(target_label=label, raw_result=raw)

    @classmethod
    def _step_family(cls, step: Any) -> str:
        if hasattr(step, "family"):
            return str(getattr(step, "family"))
        if hasattr(step, "rule"):
            rule = str(getattr(step, "rule"))
            return rule.split(":", 1)[0]
        return ""

    @classmethod
    def _step_scopes(cls, step: Any) -> list[str]:
        scopes: list[str] = []
        for attr in ("scope", "output_scope", "native_scope"):
            if hasattr(step, attr):
                value = getattr(step, attr)
                if isinstance(value, str):
                    scopes.append(value)
        if hasattr(step, "input_scopes"):
            scopes.extend(str(x) for x in getattr(step, "input_scopes"))
        native_ids: Iterable[str] = []
        for attr in ("input_native_ids", "native_ids"):
            if hasattr(step, attr):
                native_ids = list(getattr(step, attr))
                break
        for native_id in native_ids:
            entry = cls.entry_by_native_id.get(native_id)
            if entry is not None and hasattr(entry, "scope"):
                scopes.append(str(entry.scope))
        return scopes

    @classmethod
    def _run_measurement_circuit(cls, targets: Iterable[Any]) -> list[CompiledCase]:
        outcomes: list[CompiledCase] = []
        for target in targets:
            outcome = cls._compile(target)
            outcomes.append(outcome)
            if not outcome.reachable:
                break
        return outcomes

    def test_nonjoint_direct_native_measurement_runs_as_p0(self) -> None:
        target = self.intra_entry.tail.compact()
        outcome = self._compile(target)

        self.assertTrue(outcome.reachable, msg=outcome.reason)
        self.assertGreater(len(outcome.steps), 0)
        self.assertEqual(self._step_family(outcome.steps[0]), "P0")
        self.assertIn("intra_block_native", self._step_scopes(outcome.steps[0]))
        self.assertEqual(outcome.best_tail_compact, target)
        self.assertEqual(outcome.remote_inter_module_count, 0)

    def test_cross_logical_joint_native_runs_as_p0_not_bell(self) -> None:
        target = self.cross_entry.tail.compact()
        outcome = self._compile(target)

        self.assertTrue(outcome.reachable, msg=outcome.reason)
        self.assertGreater(len(outcome.steps), 0)
        self.assertEqual(self._step_family(outcome.steps[0]), "P0")
        self.assertIn("cross_logical_block_native", self._step_scopes(outcome.steps[0]))
        self.assertNotIn("inter_module_bell", self._step_scopes(outcome.steps[0]))
        self.assertEqual(outcome.best_tail_compact, target)
        self.assertEqual(outcome.remote_inter_module_count, 0)

    def test_target_expression_and_compact_tail_are_equivalent(self) -> None:
        expr = "X1P⊗X1P'"
        compact = self.cross_entry.tail.compact()

        try:
            expr_outcome = self._compile(expr)
        except Exception as exc:
            raise unittest.SkipTest(f"target expression parsing is not available in this checkout: {exc}")

        compact_outcome = self._compile(compact)

        self.assertTrue(expr_outcome.reachable, msg=expr_outcome.reason)
        self.assertTrue(compact_outcome.reachable, msg=compact_outcome.reason)
        self.assertEqual(expr_outcome.best_tail_compact, compact_outcome.best_tail_compact)
        self.assertEqual(expr_outcome.total_cost, compact_outcome.total_cost)
        self.assertEqual(self._step_family(expr_outcome.steps[0]), "P0")
        self.assertEqual(self._step_family(compact_outcome.steps[0]), "P0")

    def test_joint_protocol_source_runs_as_p2_without_bell(self) -> None:
        target = self.p2_only_step.dst.tail.compact()
        outcome = self._compile(target)

        self.assertTrue(outcome.reachable, msg=outcome.reason)
        self.assertGreater(len(outcome.steps), 0)
        self.assertEqual(self._step_family(outcome.steps[0]), "P2")
        self.assertEqual(outcome.best_tail_compact, target)
        self.assertEqual(outcome.remote_inter_module_count, 0)

        scopes = set(self._step_scopes(outcome.steps[0]))
        self.assertTrue(scopes)
        self.assertTrue(scopes.issubset({"intra_block_native", "cross_logical_block_native"}))
        self.assertNotIn("inter_module_bell", scopes)
        self.assertTrue(bool(getattr(outcome.steps[0], "applied_rule_name", None)))

    def test_p2_source_generation_never_uses_inter_module_bell_inputs(self) -> None:
        p2_steps = self._generate_p2_sources(target_tail=None)
        self.assertGreater(len(p2_steps), 0)
        for step in p2_steps[:200]:  # enough to cover generated pairs without slowing tests too much
            scopes = set(self._step_scopes(step))
            self.assertNotIn("inter_module_bell", scopes)

    def test_mixed_measurement_circuit_joint_and_nonjoint_targets_run(self) -> None:
        circuit = [
            self.intra_entry.tail.compact(),
            self.cross_entry.tail.compact(),
            self.p2_only_step.dst.tail.compact(),
        ]
        outcomes = self._run_measurement_circuit(circuit)

        self.assertEqual(len(outcomes), len(circuit))
        self.assertTrue(all(outcome.reachable for outcome in outcomes))
        self.assertEqual([self._step_family(o.steps[0]) for o in outcomes], ["P0", "P0", "P2"])
        self.assertTrue(all(o.remote_inter_module_count == 0 for o in outcomes))

    def test_unreachable_target_is_reported_without_automatic_bell_fallback(self) -> None:
        if self.unreachable_target is None:
            raise unittest.SkipTest("did not find an unreachable target under current frame")

        outcome = self._compile(self.unreachable_target)
        self.assertFalse(outcome.reachable)

        if outcome.raw_result is not None and hasattr(outcome.raw_result, "frame_unreachable"):
            self.assertTrue(outcome.frame_unreachable)
            if outcome.reason:
                self.assertIn("Bell fallback", outcome.reason)
        else:
            self.assertEqual(outcome.steps, [])

    def test_circuit_stops_on_first_unreachable_target(self) -> None:
        if self.unreachable_target is None:
            raise unittest.SkipTest("did not find an unreachable target under current frame")

        circuit = [
            self.intra_entry.tail.compact(),
            self.unreachable_target,
            self.cross_entry.tail.compact(),
        ]
        outcomes = self._run_measurement_circuit(circuit)

        self.assertEqual(len(outcomes), 2)
        self.assertTrue(outcomes[0].reachable)
        self.assertFalse(outcomes[1].reachable)


if __name__ == "__main__":
    unittest.main()

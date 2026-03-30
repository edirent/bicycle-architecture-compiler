from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prototype.two_table_compiler.build_assets import (
    refresh_joint_table_asset,
    refresh_local_table_asset,
    refresh_random_corpus_asset,
)
from prototype.two_table_compiler.demo import run as run_demo
from prototype.two_table_compiler.examples import (
    DEMO_JOINT_TABLE,
    DEMO_LOCAL_TABLE,
    TARGET_REACHABLE,
    TARGET_UNREACHABLE,
)
from prototype.two_table_compiler.io import (
    DEFAULT_JOINT_TABLE_PATH,
    DEFAULT_LOCAL_TABLE_PATH,
    RandomCorpus,
    load_joint_table,
    load_local_table,
    load_random_corpus,
    random_corpus_path,
    save_joint_table,
    save_local_table,
    save_random_corpus,
)
from prototype.two_table_compiler.pauli import anticommute, commute, single_product, tail_product
from prototype.two_table_compiler.rules import apply_p3_transitions, generate_p0_sources, generate_p2_sources
from prototype.two_table_compiler.search import (
    SearchStatus,
    synthesize,
    synthesize_search,
    synthesize_target,
)
from prototype.two_table_compiler.state import State, Tail, hidden_from_head
from prototype.two_table_compiler.tables import (
    LocalTable,
    NativeEntry,
    SCOPE_CROSS_LOGICAL_BLOCK_NATIVE,
    SCOPE_INTER_MODULE_BELL,
)


class TwoTableCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        refresh_local_table_asset(DEFAULT_LOCAL_TABLE_PATH)
        refresh_joint_table_asset(DEFAULT_JOINT_TABLE_PATH)
        local = load_local_table(DEFAULT_LOCAL_TABLE_PATH)
        allowed_targets = {
            entry.tail.compact()
            for entry in local.entries
            if entry.head in ("X", "Y", "Z")
        }
        cls.test_corpus_path = random_corpus_path(7).with_name("random_corpus_seed7_test.json")
        refresh_random_corpus_asset(
            seed=7,
            count=10,
            path=cls.test_corpus_path,
            max_weight_filter=1,
            allowed_targets=allowed_targets,
        )
        cls.cached_local = local
        cls.cached_joint = load_joint_table(DEFAULT_JOINT_TABLE_PATH)
        cls.cached_corpus = load_random_corpus(cls.test_corpus_path)

    def _native(self, native_id: str):
        return next(entry for entry in DEMO_LOCAL_TABLE.entries if entry.native_id == native_id)

    def test_pauli_product(self) -> None:
        self.assertEqual(single_product("X", "Y"), "Z")
        self.assertEqual(single_product("Z", "X"), "Y")

        lhs = Tail.from_str("X1")
        rhs = Tail.from_str("Z1")
        product = Tail(tail_product(lhs.paulis, rhs.paulis))
        self.assertEqual(product, Tail.from_str("Y1"))

    def test_tail_commutation(self) -> None:
        x1 = Tail.from_str("X1")
        z1 = Tail.from_str("Z1")
        x8 = Tail.from_str("X8")

        self.assertFalse(commute(x1.paulis, z1.paulis))
        self.assertTrue(anticommute(x1.paulis, z1.paulis))
        self.assertTrue(x1.commute(x8))

    def test_p0_seed_costs(self) -> None:
        p0_steps = generate_p0_sources(DEMO_LOCAL_TABLE)

        identity_seed = next(step for step in p0_steps if step.native_ids == ("N_I_Z2",))
        non_identity_seed = next(step for step in p0_steps if step.native_ids == ("N_X_X1",))

        self.assertEqual(identity_seed.delta_cost, 1)
        self.assertEqual(non_identity_seed.delta_cost, 2)
        self.assertEqual(identity_seed.dst.hidden, hidden_from_head("I"))
        self.assertEqual(non_identity_seed.dst.hidden, hidden_from_head("X"))

    def test_p2_generates_product_tail_with_cost_4(self) -> None:
        p2_steps = generate_p2_sources(DEMO_LOCAL_TABLE, DEMO_JOINT_TABLE)
        step = next(step for step in p2_steps if set(step.native_ids) == {"N_X_X1", "N_X_X8"})

        self.assertEqual(step.delta_cost, 4)
        self.assertEqual(step.dst.tail, TARGET_REACHABLE)

    def test_target_tail_p2_query_is_filter_over_full_source_set(self) -> None:
        all_steps = generate_p2_sources(self.cached_local, self.cached_joint, use_cache=False)
        self.assertGreater(len(all_steps), 0)
        target_tail = all_steps[0].dst.tail

        filtered = {
            (
                step.dst.hidden.name,
                step.dst.tail.compact(),
                step.delta_cost,
                tuple(sorted(step.native_ids)),
            )
            for step in all_steps
            if step.dst.tail == target_tail
        }
        targeted = {
            (
                step.dst.hidden.name,
                step.dst.tail.compact(),
                step.delta_cost,
                tuple(sorted(step.native_ids)),
            )
            for step in generate_p2_sources(
                self.cached_local,
                self.cached_joint,
                target_tail=target_tail,
                use_cache=False,
            )
        }

        self.assertEqual(targeted, filtered)

    def test_p3_requires_anticommutation(self) -> None:
        start = State(hidden=hidden_from_head("X"), tail=Tail.from_str("X8"))

        axis_commuting = self._native("N_X_X1")
        axis_anticommuting = self._native("N_Z_Z8")

        commuting_steps = apply_p3_transitions(start, axis_commuting, DEMO_JOINT_TABLE)
        anticomm_steps = apply_p3_transitions(start, axis_anticommuting, DEMO_JOINT_TABLE)

        self.assertEqual(commuting_steps, [])
        self.assertGreater(len(anticomm_steps), 0)
        self.assertTrue(all(step.family == "P3" for step in anticomm_steps))
        self.assertTrue(all(step.dst.tail == Tail.from_str("Y8") for step in anticomm_steps))

    def test_search_reachable_target_returns_projected_cost(self) -> None:
        plan = synthesize(TARGET_REACHABLE, DEMO_LOCAL_TABLE, DEMO_JOINT_TABLE)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.projected_best_cost, 4)
        self.assertEqual(plan.total_cost, 4)
        self.assertEqual(plan.best_state.tail, TARGET_REACHABLE)
        self.assertEqual(plan.steps[0].family, "P2")

    def test_search_unreachable_target_returns_none(self) -> None:
        plan = synthesize(TARGET_UNREACHABLE, DEMO_LOCAL_TABLE, DEMO_JOINT_TABLE)
        self.assertIsNone(plan)

    def test_search_api_preserves_truncated_vs_rule_unreachable(self) -> None:
        budgeted = synthesize_search(
            TARGET_UNREACHABLE,
            DEMO_LOCAL_TABLE,
            DEMO_JOINT_TABLE,
            max_popped_states=0,
        )
        full = synthesize_search(
            TARGET_UNREACHABLE,
            DEMO_LOCAL_TABLE,
            DEMO_JOINT_TABLE,
            max_popped_states=None,
        )

        self.assertIsNone(budgeted.plan)
        self.assertEqual(budgeted.status, SearchStatus.SEARCH_TRUNCATED)
        self.assertIsNone(full.plan)
        self.assertEqual(full.status, SearchStatus.RULE_UNREACHABLE)

    def test_budgeted_seeded_target_returns_upper_bound_before_pop(self) -> None:
        outcome = synthesize_target(
            TARGET_REACHABLE,
            DEMO_LOCAL_TABLE,
            DEMO_JOINT_TABLE,
            max_popped_states=0,
        )

        self.assertEqual(outcome.search_status, SearchStatus.FOUND_REACHABLE_UPPER_BOUND)
        self.assertFalse(outcome.frame_unreachable)
        self.assertIsNotNone(outcome.plan)
        assert outcome.plan is not None
        self.assertEqual(outcome.plan.total_cost, 4)
        self.assertEqual(outcome.plan.steps[0].family, "P2")

    def test_demo_run_reports_search_truncated_not_unreachable(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = run_demo(
                "YIIIIIIIIII",
                DEMO_LOCAL_TABLE,
                DEMO_JOINT_TABLE,
                max_popped_states=0,
            )

        output = stdout.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn("search status: SEARCH_TRUNCATED", output)
        self.assertIn("reachable verdict: SEARCH_TRUNCATED", output)
        self.assertNotIn("reachable verdict: UNREACHABLE", output)

    def test_hidden_class_transition_consistency(self) -> None:
        for rule in DEMO_JOINT_TABLE.p3_rules:
            expected_next = single_product(rule.cur_hidden.name, rule.axis_head)
            self.assertEqual(rule.next_hidden.name, expected_next)

    def test_local_table_roundtrip_json(self) -> None:
        roundtrip_table = LocalTable(
            frame_id="roundtrip_frame",
            entries=[
                NativeEntry(
                    native_id="cross_1",
                    frame_id="roundtrip_frame",
                    head="X",
                    tail=Tail.from_str("XIIIIIIXIII"),
                    scope=SCOPE_CROSS_LOGICAL_BLOCK_NATIVE,
                    support=("sheet:P:q1", "sheet:P':q1"),
                    metadata={"logical_canonical": "X1P⊗X1P'"},
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "local.json"
            save_local_table(roundtrip_table, path)
            loaded = load_local_table(path)

        self.assertEqual(loaded.frame_id, roundtrip_table.frame_id)
        self.assertEqual(len(loaded.entries), 1)
        self.assertEqual(loaded.entries[0].scope, SCOPE_CROSS_LOGICAL_BLOCK_NATIVE)
        self.assertEqual(loaded.entries[0].support, ("sheet:P:q1", "sheet:P':q1"))
        self.assertEqual(dict(loaded.entries[0].metadata)["logical_canonical"], "X1P⊗X1P'")

    def test_joint_table_roundtrip_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "joint.json"
            save_joint_table(DEMO_JOINT_TABLE, path)
            loaded = load_joint_table(path)

        self.assertEqual(len(loaded.p2_rules), len(DEMO_JOINT_TABLE.p2_rules))
        self.assertEqual(len(loaded.p3_rules), len(DEMO_JOINT_TABLE.p3_rules))
        self.assertEqual(loaded.p2_rules[0], DEMO_JOINT_TABLE.p2_rules[0])

    def test_random_corpus_roundtrip_json(self) -> None:
        corpus = RandomCorpus(
            seed=7,
            n_qubits=11,
            sampling_rule="uniform IXYZ + all-identity rejection",
            source="unit-test",
            count=3,
            targets=["XIIIIIIIIII", "IXIIIIIIIII", "IIXIIIIIIII"],
            max_weight_filter=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "corpus.json"
            save_random_corpus(corpus, path)
            loaded = load_random_corpus(path)

        self.assertEqual(loaded.seed, corpus.seed)
        self.assertEqual(loaded.targets, corpus.targets)
        self.assertEqual(loaded.max_weight_filter, corpus.max_weight_filter)

    def test_random_corpus_targets_parse(self) -> None:
        self.assertGreater(len(self.cached_corpus.targets), 0)
        for target in self.cached_corpus.targets:
            parsed = Tail.from_str(target)
            self.assertEqual(parsed.compact(), target)

    def test_random_corpus_targets_run_synthesis(self) -> None:
        accepting_seed_tails = {
            entry.tail.compact()
            for entry in self.cached_local.entries
            if entry.head in ("X", "Y", "Z")
        }
        selected_targets = [t for t in self.cached_corpus.targets if t in accepting_seed_tails]

        self.assertGreater(len(selected_targets), 0)
        target = selected_targets[0]
        result = synthesize_search(
            Tail.from_str(target),
            self.cached_local,
            self.cached_joint,
            max_popped_states=2_000,
        )
        self.assertEqual(result.status, SearchStatus.FOUND_OPTIMAL)
        self.assertIsNotNone(result.plan)
        assert result.plan is not None
        self.assertEqual(result.plan.best_state.tail.compact(), target)

    def test_random_corpus_smoke_summary(self) -> None:
        status_counts: dict[str, int] = {}
        for target in self.cached_corpus.targets:
            result = synthesize_search(
                Tail.from_str(target),
                self.cached_local,
                self.cached_joint,
                max_popped_states=300,
            )
            key = result.status.value
            status_counts[key] = status_counts.get(key, 0) + 1

        self.assertEqual(sum(status_counts.values()), len(self.cached_corpus.targets))
        self.assertNotIn(SearchStatus.RULE_UNREACHABLE.value, status_counts)

    def test_cross_logical_native_target_prefers_p0_no_bell(self) -> None:
        outcome = synthesize_target(
            "X1P⊗X1P'",
            self.cached_local,
            self.cached_joint,
            max_popped_states=2_000,
        )
        self.assertFalse(outcome.frame_unreachable)
        self.assertIsNotNone(outcome.plan)
        assert outcome.plan is not None
        self.assertTrue(outcome.classification.is_logical_notation)
        self.assertEqual(outcome.classification.canonical_tail, Tail.from_str("XIIIIIIXIII"))
        self.assertEqual(outcome.plan.steps[0].family, "P0")
        self.assertEqual(outcome.plan.steps[0].scope, SCOPE_CROSS_LOGICAL_BLOCK_NATIVE)
        self.assertNotEqual(outcome.plan.steps[0].scope, SCOPE_INTER_MODULE_BELL)
        self.assertEqual(outcome.plan.remote_inter_module_count, 0)
        self.assertEqual(outcome.plan.total_cost, 2)
        self.assertEqual(outcome.plan.projected_best_cost, 2)

    def test_joint_composed_target_uses_p2_source(self) -> None:
        frame = LocalTable(
            frame_id="joint_case_frame",
            entries=[
                NativeEntry(
                    native_id="joint_x1",
                    frame_id="joint_case_frame",
                    head="X",
                    tail=Tail.from_str("XIIIIIIIIII"),
                ),
                NativeEntry(
                    native_id="joint_x8",
                    frame_id="joint_case_frame",
                    head="X",
                    tail=Tail.from_str("IIIIIIIXIII"),
                ),
            ],
        )
        target = Tail.from_str("XIIIIIIXIII")
        self.assertEqual(frame.native_family_lookup(target), [])

        outcome = synthesize_target(target, frame, DEMO_JOINT_TABLE, max_popped_states=200)
        self.assertFalse(outcome.frame_unreachable)
        self.assertIsNotNone(outcome.plan)
        assert outcome.plan is not None
        self.assertEqual(outcome.plan.steps[0].family, "P2")
        self.assertIn(outcome.plan.steps[0].scope, ("intra_block_native", "cross_logical_block_native"))
        self.assertEqual(outcome.plan.steps[0].input_native_ids, ("joint_x1", "joint_x8"))
        self.assertEqual(outcome.plan.remote_inter_module_count, 0)
        self.assertEqual(outcome.plan.total_cost, 4)

    def test_cross_logical_native_frame_miss_is_not_bell_fallback(self) -> None:
        entries_without_cross = [
            entry for entry in self.cached_local.entries if entry.scope != SCOPE_CROSS_LOGICAL_BLOCK_NATIVE
        ]
        frame_without_cross = LocalTable(frame_id=self.cached_local.frame_id, entries=entries_without_cross)
        outcome = synthesize_target(
            "X1P⊗X1P'",
            frame_without_cross,
            self.cached_joint,
            max_popped_states=2_000,
        )
        self.assertTrue(outcome.frame_unreachable)
        self.assertIsNone(outcome.plan)
        self.assertIsNotNone(outcome.reason)
        assert outcome.reason is not None
        self.assertIn("no automatic Bell fallback", outcome.reason)
        self.assertEqual(outcome.classification.matched_native_ids(), ())

    def test_frame_unreachable_when_neither_native_nor_joint(self) -> None:
        frame = LocalTable(
            frame_id="no_joint_frame",
            entries=[
                NativeEntry(
                    native_id="only_x1",
                    frame_id="no_joint_frame",
                    head="X",
                    tail=Tail.from_str("XIIIIIIIIII"),
                ),
            ],
        )
        outcome = synthesize_target("XIIIIIIXIII", frame, DEMO_JOINT_TABLE, max_popped_states=200)
        self.assertTrue(outcome.frame_unreachable)
        self.assertIsNone(outcome.plan)
        self.assertIsNotNone(outcome.reason)
        assert outcome.reason is not None
        self.assertIn("no automatic Bell fallback", outcome.reason)

    def test_demo_cli_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        target = self.cached_corpus.targets[0]
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "prototype.two_table_compiler.demo",
                "--target",
                target,
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertIn(proc.returncode, (0, 1), msg=proc.stdout + "\n" + proc.stderr)
        self.assertIn(f"target input: {target}", proc.stdout)
        self.assertIn(f"target tail: {target}", proc.stdout)
        self.assertIn("local table summary:", proc.stdout)
        self.assertIn("joint/protocol table summary:", proc.stdout)
        self.assertIn("final projected result:", proc.stdout)


if __name__ == "__main__":
    unittest.main()

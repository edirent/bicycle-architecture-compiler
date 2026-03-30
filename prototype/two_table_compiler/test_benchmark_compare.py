from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prototype.two_table_compiler.benchmark_compare import (
    BenchmarkConfig,
    DEFAULT_BASELINE_JSON,
    _mode_result_from_outcome,
    _stratum_from_full_mode,
    run_benchmark,
)
from prototype.two_table_compiler.io import DEFAULT_JOINT_TABLE_PATH, DEFAULT_LOCAL_TABLE_PATH
from prototype.two_table_compiler.search import SearchStatus, SynthesisOutcome
from prototype.two_table_compiler.state import Tail
from prototype.two_table_compiler.targeting import TargetClassification


class BenchmarkCompareSmokeTests(unittest.TestCase):
    def test_sample_benchmark_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "bench_out"
            cfg = BenchmarkConfig(
                mode="sample",
                seed=7,
                count=1,
                max_weight=1,
                include_identity=False,
                max_popped_states=None,
                disable_p2=True,
                output_dir=out_dir,
                frame_id=None,
                jobs=1,
                progress_every=0,
                baseline_json=DEFAULT_BASELINE_JSON,
                local_table_path=DEFAULT_LOCAL_TABLE_PATH,
                joint_table_path=DEFAULT_JOINT_TABLE_PATH,
            )
            with patch(
                "prototype.two_table_compiler.benchmark_compare._sample_targets",
                return_value=["IXIIIIIIIII"],
            ):
                result = run_benchmark(cfg)

            summary_path = Path(result["summary_path"])
            per_target_path = Path(result["per_target_path"])
            local_diff_path = Path(result["local_diff_path"])

            self.assertTrue(summary_path.exists())
            self.assertTrue(per_target_path.exists())
            self.assertTrue(local_diff_path.exists())

            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertIn("by_mode", summary)
            self.assertIn("pairwise", summary)
            self.assertIn("inventory", summary)
            self.assertIn("counts_by_status", summary)
            self.assertTrue(summary["semantic_contract"]["used_no_budget"])
            self.assertEqual("synthesize_search", summary["semantic_contract"]["search_api"])
            self.assertIn("rule_unreachable_note", summary["semantic_contract"])
            self.assertIn("target_aware_p2_note", summary["semantic_contract"])
            self.assertIn("ours_full", summary["by_mode"])
            self.assertIn("gross_baseline", summary["by_mode"])
            self.assertIn(
                "gross_reachable_and_ours_full_rule_unreachable",
                summary["special_counts"],
            )
            self.assertIn(
                "gross_reachable_and_ours_full_truncated",
                summary["special_counts"],
            )
            self.assertIn(
                "ours_full_reachable_and_ours_no_joint_rule_unreachable",
                summary["special_counts"],
            )
            self.assertIn(
                "ours_full_reachable_and_ours_no_joint_truncated",
                summary["special_counts"],
            )
            self.assertTrue(
                summary["pairwise"]["ours_full_vs_gross_baseline"]["computed_on_common_reachable_targets_only"]
            )
            self.assertEqual(
                0,
                summary["counts_by_status"]["ours_full"].get(SearchStatus.SEARCH_TRUNCATED.value, 0),
            )
            self.assertEqual(
                0,
                summary["counts_by_status"]["ours_full"].get(SearchStatus.FOUND_REACHABLE_UPPER_BOUND.value, 0),
            )

            with per_target_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("target", rows[0])
            self.assertIn("used_no_budget", rows[0])
            self.assertIn("search_status_gross", rows[0])
            self.assertIn("search_status_note_ours_full", rows[0])
            self.assertIn("reachable_ours_full", rows[0])
            self.assertIn("truncated_ours_full", rows[0])
            self.assertIn("rule_unreachable_ours_full", rows[0])
            self.assertIn("optimal_ours_full", rows[0])
            self.assertIn("gross_baseline_cost", rows[0])
            self.assertIn("ours_full_cost", rows[0])
            self.assertIn("ours_no_joint_cost", rows[0])
            self.assertIn("ours_full_search_status", rows[0])
            self.assertIn("stratum", rows[0])

            with local_diff_path.open("r", encoding="utf-8", newline="") as handle:
                diff_rows = list(csv.DictReader(handle))
            self.assertGreater(len(diff_rows), 0)
            self.assertIn("normalized_id", diff_rows[0])
            self.assertIn("in_ours", diff_rows[0])
            self.assertIn("in_gross", diff_rows[0])

    def test_search_truncated_row_is_not_labeled_frame_unreachable(self) -> None:
        outcome = SynthesisOutcome(
            classification=TargetClassification(
                input_target="XIIIIIIIIII",
                canonical_tail=Tail.from_str("XIIIIIIIIII"),
                canonical_logical=None,
                matched_native_entries=(),
                is_logical_notation=False,
                frame_unreachable=False,
                reason=None,
            ),
            plan=None,
            frame_unreachable=False,
            reason="search truncated",
            search_status=SearchStatus.SEARCH_TRUNCATED,
            search_truncated=True,
            projected_best_cost_exists=False,
        )

        row = _mode_result_from_outcome(outcome)
        self.assertFalse(row["frame_unreachable"])
        self.assertEqual(row["search_status"], SearchStatus.SEARCH_TRUNCATED.value)
        self.assertEqual(_stratum_from_full_mode(row), "search_truncated")

    def test_mode_result_from_outcome_rejects_none(self) -> None:
        with self.assertRaises(TypeError):
            _mode_result_from_outcome(None)


if __name__ == "__main__":
    unittest.main()

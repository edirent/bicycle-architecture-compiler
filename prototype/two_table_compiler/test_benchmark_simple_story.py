from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prototype.two_table_compiler.benchmark_compare import (
    BenchmarkConfig,
    DEFAULT_BASELINE_JSON,
    run_benchmark,
)
from prototype.two_table_compiler.benchmark_simple_story import generate_simple_story
from prototype.two_table_compiler.io import DEFAULT_JOINT_TABLE_PATH, DEFAULT_LOCAL_TABLE_PATH
from prototype.two_table_compiler.search import SearchStatus


class BenchmarkSimpleStorySmokeTests(unittest.TestCase):
    def test_story_artifacts_from_real_benchmark_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "bench_out"
            cfg = BenchmarkConfig(
                mode="sample",
                seed=7,
                count=2,
                max_weight=1,
                include_identity=False,
                max_popped_states=None,
                disable_p2=False,
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
                return_value=["IXIIIIIIIII", "IIIIIIXIIII"],
            ):
                run_benchmark(cfg)
            paths = generate_simple_story(input_dir=out_dir, output_dir=out_dir)

            targets_csv = paths["simple_targets_csv"]
            circuits_csv = paths["simple_circuits_csv"]
            story_md = paths["story_report_md"]
            costs_png = paths["figure_story_costs_png"]
            circuits_png = paths["figure_story_circuits_png"]

            self.assertTrue(targets_csv.exists())
            self.assertTrue(circuits_csv.exists())
            self.assertTrue(story_md.exists())
            self.assertTrue(costs_png.exists())
            self.assertTrue(circuits_png.exists())

            with targets_csv.open("r", encoding="utf-8", newline="") as handle:
                target_rows = list(csv.DictReader(handle))
            self.assertEqual(len(target_rows), 6)
            self.assertIn("story_label", target_rows[0])
            self.assertIn("target_tail", target_rows[0])
            self.assertIn("gross_search_status", target_rows[0])
            self.assertIn("ours_no_joint_search_status", target_rows[0])
            self.assertIn("ours_full_search_status", target_rows[0])
            self.assertIn("ours_full_search_status_note", target_rows[0])
            self.assertIn("ours_full_path_signature", target_rows[0])
            self.assertIn("one_line_reason", target_rows[0])
            self.assertEqual(len({row["target_tail"] for row in target_rows}), len(target_rows))

            by_label = {row["story_label"]: row for row in target_rows}
            self.assertIn("control_same_direct_local", by_label)
            self.assertIn("cross_logical_native_case", by_label)
            self.assertIn("joint_source_win_case", by_label)
            self.assertIn("honest_loss_or_tie_case", by_label)
            self.assertIn("previously_suspicious_case", by_label)
            self.assertEqual("XIIIIIIXIII", by_label["cross_logical_native_case"]["target_tail"])
            self.assertEqual("P0", by_label["cross_logical_native_case"]["ours_full_first_family"])
            self.assertEqual(
                "cross_logical_block_native",
                by_label["cross_logical_native_case"]["ours_full_first_scope"],
            )
            self.assertNotEqual(
                SearchStatus.SEARCH_TRUNCATED.value,
                by_label["cross_logical_native_case"]["ours_full_search_status"],
            )

            with circuits_csv.open("r", encoding="utf-8", newline="") as handle:
                circuit_rows = list(csv.DictReader(handle))
            self.assertEqual(len(circuit_rows), 3)
            self.assertIn("circuit_name", circuit_rows[0])
            self.assertIn("ours_full_search_statuses", circuit_rows[0])
            self.assertIn("ours_full_search_status_notes", circuit_rows[0])
            self.assertIn("ours_full_total_cost", circuit_rows[0])
            self.assertIn("ours_full_p2_count", circuit_rows[0])
            self.assertIn("ours_full_search_truncated_count", circuit_rows[0])

            circuits_by_name = {row["circuit_name"]: row for row in circuit_rows}
            self.assertIn("cross_logical_plus_local", circuits_by_name)
            self.assertIn("joint_matters", circuits_by_name)
            self.assertIn("XIIIIIIXIII", circuits_by_name["cross_logical_plus_local"]["ordered_targets"])
            self.assertIn(
                by_label["joint_source_win_case"]["target_tail"],
                circuits_by_name["joint_matters"]["ordered_targets"],
            )
            if "frame_unreachable_case" not in by_label:
                self.assertIn(
                    "No true RULE_UNREACHABLE case was found in the audited corpus after the four-state semantic fix.",
                    story_md.read_text(encoding="utf-8"),
                )

            report_text = story_md.read_text(encoding="utf-8")
            self.assertIn("Section 1: One-paragraph takeaway", report_text)
            self.assertIn("Section 2: Six representative targets", report_text)
            self.assertIn("Section 3: Three toy circuits", report_text)
            self.assertIn("Section 4: One-sentence conclusion", report_text)
            self.assertIn("gross_search_status", report_text)
            self.assertIn("ours_no_joint_search_status", report_text)
            self.assertIn("upper bound found, not yet proven optimal", report_text)
            self.assertNotIn("No cross_logical_native target appears in this benchmark corpus", report_text)


if __name__ == "__main__":
    unittest.main()

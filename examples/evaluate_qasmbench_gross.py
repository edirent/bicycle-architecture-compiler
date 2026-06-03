#!/usr/bin/env python3
"""Evaluate LGM accounting and PBC weight stats on results/qasmbench_gross."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lgm_compiler.dataset_eval import aggregate_reports, evaluate_dataset, markdown_table, write_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/qasmbench_gross"))
    parser.add_argument(
        "--use-qiskit",
        action="store_true",
        help="Reserved for optional future Qiskit parsing; the MVP parser is used in this implementation.",
    )
    args = parser.parse_args()

    reports = evaluate_dataset(args.root)
    json_path, csv_path = write_reports(reports, args.root)
    aggregate = aggregate_reports(reports)

    print("Per-benchmark LGM vs PBC summary:")
    print(markdown_table(reports))
    print()
    print("Aggregate totals:")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print()
    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote CSV summary: {csv_path}")

    if args.use_qiskit:
        print()
        print("--use-qiskit was requested, but Qiskit parsing is optional and not required here; used MVP parser.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright contributors to the Bicycle Architecture Compiler project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Count measurements and related instructions in compiled Gross ISA JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


COUNT_FIELDS = [
    "logical_qubits",
    "gross_modules",
    "in_module_measure",
    "inter_module_code_code",
    "t_injection",
    "inter_module_including_factory",
    "automorphism",
    "joint_measure_halves_raw",
]


def _instruction_name(instruction: Any) -> str | None:
    if not isinstance(instruction, dict) or len(instruction) != 1:
        return None
    return next(iter(instruction))


def _iter_instructions(step: Any) -> Iterable[tuple[int | None, str]]:
    if not isinstance(step, list):
        return
    for addressed in step:
        if (
            isinstance(addressed, list)
            and len(addressed) == 2
            and isinstance(addressed[1], dict)
        ):
            module = addressed[0] if isinstance(addressed[0], int) else None
            name = _instruction_name(addressed[1])
            if name is not None:
                yield module, name


def count_isa(path: Path, logical_qubits: int | None = None) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "in_module_measure": 0,
        "inter_module_code_code": 0,
        "t_injection": 0,
        "inter_module_including_factory": 0,
        "automorphism": 0,
        "joint_measure_halves_raw": 0,
        "isa_lines": 0,
        "isa_steps": 0,
        "parse_errors": 0,
        "warnings": [],
    }

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            counts["isa_lines"] += 1
            try:
                chunk = json.loads(stripped)
            except json.JSONDecodeError as exc:
                counts["parse_errors"] += 1
                counts["warnings"].append(
                    f"{path}:{line_number}: JSONDecodeError: {exc.msg}"
                )
                continue

            if not isinstance(chunk, list):
                counts["warnings"].append(f"{path}:{line_number}: expected list chunk")
                continue

            for step in chunk:
                counts["isa_steps"] += 1
                names = [name for _, name in _iter_instructions(step)]
                if not names:
                    continue
                counts["in_module_measure"] += names.count("Measure")
                counts["joint_measure_halves_raw"] += names.count("JointMeasure")
                counts["t_injection"] += names.count("TGate")
                counts["automorphism"] += names.count("Automorphism")
                if "JointMeasure" in names:
                    counts["inter_module_code_code"] += 1

    counts["inter_module_including_factory"] = (
        counts["inter_module_code_code"] + counts["t_injection"]
    )

    if logical_qubits is not None:
        counts["logical_qubits"] = int(logical_qubits)
        counts["gross_modules"] = int(math.ceil(logical_qubits / 11))

    expected_halves = 2 * counts["inter_module_code_code"]
    if counts["joint_measure_halves_raw"] != expected_halves:
        counts["warnings"].append(
            "joint_measure_halves_raw "
            f"({counts['joint_measure_halves_raw']}) != "
            f"2 * inter_module_code_code ({expected_halves})"
        )

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read one compiled Gross ISA JSONL file and print measurement counts.",
    )
    parser.add_argument("isa_jsonl", type=Path)
    parser.add_argument("--logical-qubits", type=int)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = count_isa(args.isa_jsonl, args.logical_qubits)
    warnings = counts.get("warnings") or []
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=COUNT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(counts)
    else:
        print(json.dumps(counts, indent=2, sort_keys=True))
    return 1 if counts.get("parse_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())

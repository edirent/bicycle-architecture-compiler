#!/usr/bin/env python3
"""Generate boss validation reports from 11Q comparison outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_TOTAL = 4**11


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--commands-log",
        type=Path,
        default=Path("results/reports/boss_commands.log"),
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--all-commands-passed", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, dict):
        return loaded
    return None


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def load_hist(path: Path) -> dict[int, int] | None:
    raw = load_json(path)
    if raw is None:
        return None
    out: dict[int, int] = {}
    for k, v in raw.items():
        out[int(k)] = int(v)
    return out


def hist_total(hist: dict[int, int] | None) -> int | None:
    if hist is None:
        return None
    return sum(hist.values())


def value_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        )
        return out.strip()
    except Exception:
        return "UNKNOWN"


def read_commands(path: Path) -> list[str]:
    if not path.exists():
        return []
    commands: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            commands.append(line)
    return commands


def benchmark_entry_available(path: Path, marker: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return marker in text


def local_hist_keys(summary: dict[str, Any] | None) -> set[int]:
    if summary is None:
        return set()
    raw = summary.get("local_measurement_count_hist")
    if not isinstance(raw, dict):
        return set()
    return {int(k) for k in raw}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()

    hist_dir = root / "results/histograms"
    report_dir = root / "results/reports"
    fig_dir = root / "results/figures"
    bench_dir = root / "results/benchmarks"

    paper_gross_hist_path = hist_dir / "paper_baseline_gross_exact_hist.json"
    paper_two_hist_path = hist_dir / "paper_baseline_two_gross_exact_hist.json"
    ours_gross_hist_path = hist_dir / "ours_single_shot_gross_exact_hist.json"
    ours_two_hist_path = hist_dir / "ours_single_shot_two_gross_exact_hist.json"
    ours_fixed_hist_path = hist_dir / "ours_single_shot_fixed_pivot_exact_hist.json"
    ours_best_hist_path = hist_dir / "ours_single_shot_best_single_pivot_exact_hist.json"

    paper_gross_summary_path = first_existing(
        [
            report_dir / "paper_baseline_gross_summary.json",
            hist_dir / "paper_baseline_gross_summary.json",
        ]
    )
    paper_two_summary_path = first_existing(
        [
            report_dir / "paper_baseline_two_gross_summary.json",
            hist_dir / "paper_baseline_two_gross_summary.json",
        ]
    )
    ours_gross_summary_path = first_existing(
        [
            report_dir / "ours_single_shot_gross_summary.json",
            hist_dir / "ours_single_shot_gross_summary.json",
        ]
    )
    ours_two_summary_path = first_existing(
        [
            report_dir / "ours_single_shot_two_gross_summary.json",
            hist_dir / "ours_single_shot_two_gross_summary.json",
        ]
    )
    ours_best_summary_path = report_dir / "ours_single_shot_best_single_pivot_summary.json"

    bench_paper_path = bench_dir / "paper_baseline_bench.txt"
    bench_ours_path = bench_dir / "ours_single_shot_bench.txt"

    figure_paths = [
        fig_dir / "paper_vs_ours_11q_5bin.png",
        fig_dir / "compare_11q_single_shot_exact_gross.png",
        fig_dir / "compare_11q_single_shot_binned_gross.png",
        fig_dir / "compare_paper_fixed_best_single_binned.png",
    ]

    paper_gross_hist = load_hist(paper_gross_hist_path)
    ours_fixed_hist = load_hist(ours_fixed_hist_path)

    paper_gross_summary = load_json(paper_gross_summary_path)
    paper_two_summary = load_json(paper_two_summary_path)
    ours_gross_summary = load_json(ours_gross_summary_path)
    ours_two_summary = load_json(ours_two_summary_path)
    ours_best_summary = load_json(ours_best_summary_path)

    local_keys = local_hist_keys(ours_gross_summary)
    local_hist_present = isinstance(
        (ours_gross_summary or {}).get("local_measurement_count_hist"),
        dict,
    )
    allowed_local_keys = {1, 2, 4, 6}

    checks = [
        CheckResult(
            name="paper_total_is_4_pow_11",
            passed=hist_total(paper_gross_hist) == EXPECTED_TOTAL,
            detail=f"paper_total={hist_total(paper_gross_hist)} expected={EXPECTED_TOTAL}",
        ),
        CheckResult(
            name="ours_total_is_4_pow_11",
            passed=hist_total(ours_fixed_hist) == EXPECTED_TOTAL,
            detail=f"ours_total={hist_total(ours_fixed_hist)} expected={EXPECTED_TOTAL}",
        ),
        CheckResult(
            name="local_measurement_count_hist_present",
            passed=local_hist_present,
            detail=f"present={local_hist_present}",
        ),
        CheckResult(
            name="local_measurement_count_keys_subset",
            passed=local_keys.issubset(allowed_local_keys),
            detail=f"local_keys={sorted(local_keys)} allowed={sorted(allowed_local_keys)}",
        ),
        CheckResult(
            name="local_measurement_count_no_5",
            passed=5 not in local_keys,
            detail=f"local_keys={sorted(local_keys)}",
        ),
        CheckResult(
            name="local_measurement_count_no_ge_7",
            passed=all(k < 7 for k in local_keys),
            detail=f"local_keys={sorted(local_keys)}",
        ),
        CheckResult(
            name="fixed_vs_best_outputs_separated",
            passed=ours_fixed_hist_path != ours_best_hist_path
            and ours_fixed_hist_path.exists()
            and ours_best_hist_path.exists(),
            detail=(
                f"fixed={ours_fixed_hist_path} exists={ours_fixed_hist_path.exists()} ; "
                f"best={ours_best_hist_path} exists={ours_best_hist_path.exists()}"
            ),
        ),
        CheckResult(
            name="benchmark_entry_paper_available",
            passed=benchmark_entry_available(bench_paper_path, "benchmark=paper_baseline_11q"),
            detail=f"path={bench_paper_path}",
        ),
        CheckResult(
            name="benchmark_entry_ours_available",
            passed=benchmark_entry_available(bench_ours_path, "benchmark=ours_single_shot_11q"),
            detail=f"path={bench_ours_path}",
        ),
    ]

    all_passed = all(c.passed for c in checks)
    commands = read_commands((root / args.commands_log).resolve() if not args.commands_log.is_absolute() else args.commands_log)
    commit = git_commit(root)

    paper_mean_gross = (paper_gross_summary or {}).get("mean")
    paper_mean_two = (paper_two_summary or {}).get("mean")
    ours_mean_gross = (ours_gross_summary or {}).get("mean")
    ours_mean_two = (ours_two_summary or {}).get("mean")
    ours_fixed_mean = (ours_best_summary or {}).get("fixed_pivot_mean")
    ours_best_mean = (ours_best_summary or {}).get("best_single_pivot_mean")

    payload = {
        "git_commit": commit,
        "mode": args.mode,
        "all_commands_passed": args.all_commands_passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands_run": commands,
        "all_checks_passed": all_passed,
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in checks
        ],
        "means": {
            "paper_baseline": {
                "gross": paper_mean_gross,
                "two_gross": paper_mean_two,
            },
            "ours_single_shot": {
                "gross": ours_mean_gross,
                "two_gross": ours_mean_two,
                "fixed_pivot": ours_fixed_mean,
                "best_single_pivot": ours_best_mean,
            },
        },
        "histogram_paths": [
            str(path.relative_to(root))
            for path in [
                paper_gross_hist_path,
                paper_two_hist_path,
                ours_gross_hist_path,
                ours_two_hist_path,
                ours_fixed_hist_path,
                ours_best_hist_path,
            ]
        ],
        "figure_paths": [
            {
                "path": str(path.relative_to(root)),
                "exists": path.exists(),
            }
            for path in figure_paths
        ],
        "benchmark_outputs": [
            str(bench_paper_path.relative_to(root)),
            str(bench_ours_path.relative_to(root)),
        ],
        "benchmark_entry_available": {
            "paper_baseline_11q": checks[7].passed,
            "ours_single_shot_11q": checks[8].passed,
        },
    }

    out_json = report_dir / "boss_validation.json"
    out_md = report_dir / "boss_validation.md"
    key_figure_path = fig_dir / "paper_vs_ours_11q_5bin.png"
    write_json(out_json, payload)

    md_lines = [
        "# Boss Validation",
        "",
        f"- git commit: `{commit}`",
        f"- mode: `{args.mode}`",
        f"- all commands passed: `{args.all_commands_passed}`",
        f"- generated at (UTC): `{payload['generated_at_utc']}`",
        f"- all checks passed: `{all_passed}`",
        "",
        "## Means",
        "",
        f"- paper baseline mean (gross): `{value_or_na(paper_mean_gross)}`",
        f"- paper baseline mean (two-gross): `{value_or_na(paper_mean_two)}`",
        f"- ours single-shot mean (gross): `{value_or_na(ours_mean_gross)}`",
        f"- ours single-shot mean (two-gross): `{value_or_na(ours_mean_two)}`",
        f"- ours fixed-pivot mean: `{value_or_na(ours_fixed_mean)}`",
        f"- ours best-single-pivot mean: `{value_or_na(ours_best_mean)}`",
        "",
        "## Checks",
        "",
    ]
    for c in checks:
        md_lines.append(f"- [{'PASS' if c.passed else 'FAIL'}] `{c.name}`: {c.detail}")

    md_lines.extend(
        [
            "",
            "## Paths",
            "",
            "- `results/histograms/paper_baseline_gross_exact_hist.json`",
            "- `results/histograms/paper_baseline_two_gross_exact_hist.json`",
            "- `results/histograms/ours_single_shot_fixed_pivot_exact_hist.json`",
            "- `results/histograms/ours_single_shot_best_single_pivot_exact_hist.json`",
            "- figures:",
        ]
    )
    for figure_path in figure_paths:
        md_lines.append(
            f"- `{figure_path.relative_to(root)}` (exists={figure_path.exists()})"
        )
    md_lines.extend(
        [
            "- benchmark outputs:",
            f"- `{bench_paper_path.relative_to(root)}`",
            f"- `{bench_ours_path.relative_to(root)}`",
            "",
            "## Commands Run",
            "",
        ]
    )
    if commands:
        md_lines.extend([f"- `{cmd}`" for cmd in commands])
    else:
        md_lines.append("- *(no commands log found)*")

    write_md(out_md, md_lines)

    print(f"git commit hash: {commit}")
    print(f"all commands passed: {str(args.all_commands_passed).lower()}")
    print(f"all checks passed: {str(all_passed).lower()}")
    print(f"paper baseline mean (gross): {value_or_na(paper_mean_gross)}")
    print(f"paper baseline mean (two-gross): {value_or_na(paper_mean_two)}")
    print(f"ours fixed-pivot mean: {value_or_na(ours_fixed_mean)}")
    print(f"ours best-single-pivot mean: {value_or_na(ours_best_mean)}")
    print("key histogram paths:")
    print(f"  {paper_gross_hist_path.relative_to(root)}")
    print(f"  {paper_two_hist_path.relative_to(root)}")
    print(f"  {ours_fixed_hist_path.relative_to(root)}")
    print(f"  {ours_best_hist_path.relative_to(root)}")
    print("key figure paths:")
    print(f"  {key_figure_path.relative_to(root)} (exists={key_figure_path.exists()})")
    print(f"  {(fig_dir / 'compare_paper_fixed_best_single_binned.png').relative_to(root)} (exists={(fig_dir / 'compare_paper_fixed_best_single_binned.png').exists()})")
    print("key report paths:")
    print(f"  {out_md.relative_to(root)}")
    print(f"  {out_json.relative_to(root)}")
    print(
        "benchmark entry available: "
        f"paper_baseline_11q={checks[7].passed}, "
        f"ours_single_shot_11q={checks[8].passed}"
    )

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

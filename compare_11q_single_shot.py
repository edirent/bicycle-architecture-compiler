#!/usr/bin/env python3
"""Compare 11-qubit exact histograms for paper baseline vs ours_single_shot.

It prints totals / means / medians / support sets, then writes:
- exact histogram comparisons
- paper-style 5-bin comparisons
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict

HIST_DIR = Path("results/histograms")
REPORT_DIR = Path("results/reports")
FIG_DIR = Path("results/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Regenerate exact hist files via Rust binary before plotting.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--compare-fixed-safe",
        action="store_true",
        help="Compare fixed-pivot vs safe-pivot exact histograms.",
    )
    parser.add_argument(
        "--compare-paper-fixed-safe",
        action="store_true",
        help="Compare paper baseline vs fixed-pivot vs safe-pivot exact histograms.",
    )
    parser.add_argument(
        "--compare-paper-fixed-best-single",
        action="store_true",
        help="Compare paper baseline vs fixed-pivot vs empirical best-single-pivot.",
    )
    parser.add_argument("--ours-gross", type=Path, default=HIST_DIR / "ours_single_shot_gross_exact_hist.json")
    parser.add_argument(
        "--ours-two",
        "--ours-two-gross",
        dest="ours_two",
        type=Path,
        default=HIST_DIR / "ours_single_shot_two_gross_exact_hist.json",
    )
    parser.add_argument("--paper-gross", type=Path, default=HIST_DIR / "paper_baseline_gross_exact_hist.json")
    parser.add_argument(
        "--paper-two",
        "--paper-two-gross",
        dest="paper_two",
        type=Path,
        default=HIST_DIR / "paper_baseline_two_gross_exact_hist.json",
    )
    parser.add_argument("--fixed-hist", type=Path, default=HIST_DIR / "ours_single_shot_fixed_pivot_exact_hist.json")
    parser.add_argument("--safe-hist", type=Path, default=HIST_DIR / "ours_safe_pivot_exact_hist.json")
    parser.add_argument(
        "--best-single-hist",
        type=Path,
        default=HIST_DIR / "ours_single_shot_best_single_pivot_exact_hist.json",
    )
    parser.add_argument("--paper-hist", type=Path, default=HIST_DIR / "paper_baseline_gross_exact_hist.json")
    return parser.parse_args()


def ensure_output_dirs(repo_root: Path) -> None:
    (repo_root / HIST_DIR).mkdir(parents=True, exist_ok=True)
    (repo_root / REPORT_DIR).mkdir(parents=True, exist_ok=True)
    (repo_root / FIG_DIR).mkdir(parents=True, exist_ok=True)


def move_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dst)


def relocate_generated_paper_outputs(repo_root: Path) -> None:
    move_if_exists(
        repo_root / "paper_baseline_gross_exact_hist.json",
        repo_root / HIST_DIR / "paper_baseline_gross_exact_hist.json",
    )
    move_if_exists(
        repo_root / "paper_baseline_gross_summary.json",
        repo_root / REPORT_DIR / "paper_baseline_gross_summary.json",
    )
    move_if_exists(
        repo_root / "paper_baseline_two_gross_exact_hist.json",
        repo_root / HIST_DIR / "paper_baseline_two_gross_exact_hist.json",
    )
    move_if_exists(
        repo_root / "paper_baseline_two_gross_summary.json",
        repo_root / REPORT_DIR / "paper_baseline_two_gross_summary.json",
    )


def run_generate(repo_root: Path) -> None:
    ensure_output_dirs(repo_root)
    generate_commands = [
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "bicycle_compiler",
            "--",
            "--paper-beta-report",
            "--csv",
            "native_dictionary_gross.csv",
        ],
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "bicycle_compiler",
            "--",
            "--paper-beta-report",
            "--csv",
            "native_dictionary_two_gross.csv",
        ],
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "ours_single_shot",
            "--",
            "--csv",
            "native_dictionary_gross.csv",
            "--out",
            str(HIST_DIR / "ours_single_shot_gross"),
        ],
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "ours_single_shot",
            "--",
            "--csv",
            "native_dictionary_two_gross.csv",
            "--out",
            str(HIST_DIR / "ours_single_shot_two_gross"),
        ],
    ]
    for cmd in generate_commands:
        subprocess.run(cmd, cwd=repo_root, check=True)
    relocate_generated_paper_outputs(repo_root)


def run_generate_safe(repo_root: Path, include_paper: bool) -> None:
    ensure_output_dirs(repo_root)
    generate_commands = []
    if include_paper:
        generate_commands.append(
            [
                "cargo",
                "run",
                "--release",
                "--bin",
                "bicycle_compiler",
                "--",
                "--paper-beta-report",
                "--csv",
                "native_dictionary_gross.csv",
            ]
        )
    generate_commands.append(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "ours_single_shot",
            "--",
            "--csv",
            "native_dictionary_gross.csv",
            "--ours-safe-pivot-report",
            "--safe-pivot-certification-out",
            str(REPORT_DIR / "safe_pivot_certification.json"),
            "--ours-fixed-pivot-hist-out",
            str(HIST_DIR / "ours_fixed_pivot_exact_hist.json"),
            "--ours-safe-pivot-hist-out",
            str(HIST_DIR / "ours_safe_pivot_exact_hist.json"),
            "--ours-safe-pivot-summary-out",
            str(REPORT_DIR / "ours_safe_pivot_summary.json"),
        ]
    )
    for cmd in generate_commands:
        subprocess.run(cmd, cwd=repo_root, check=True)
    if include_paper:
        relocate_generated_paper_outputs(repo_root)


def run_generate_pivot_scan(repo_root: Path, include_paper: bool) -> None:
    ensure_output_dirs(repo_root)
    generate_commands = []
    if include_paper:
        generate_commands.append(
            [
                "cargo",
                "run",
                "--release",
                "--bin",
                "bicycle_compiler",
                "--",
                "--paper-beta-report",
                "--csv",
                "native_dictionary_gross.csv",
            ]
        )
    generate_commands.append(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "ours_single_shot",
            "--",
            "--csv",
            "native_dictionary_gross.csv",
            "--ours-pivot-scan-report",
            "--pivot-scan-summary-out",
            str(REPORT_DIR / "pivot_scan_summary.json"),
            "--ours-single-shot-fixed-pivot-hist-out",
            str(HIST_DIR / "ours_single_shot_fixed_pivot_exact_hist.json"),
            "--ours-single-shot-best-single-pivot-hist-out",
            str(HIST_DIR / "ours_single_shot_best_single_pivot_exact_hist.json"),
            "--ours-single-shot-best-single-pivot-summary-out",
            str(REPORT_DIR / "ours_single_shot_best_single_pivot_summary.json"),
            "--ours-single-shot-pivot-prefix",
            str(HIST_DIR / "ours_single_shot_pivot"),
        ]
    )
    for cmd in generate_commands:
        subprocess.run(cmd, cwd=repo_root, check=True)
    if include_paper:
        relocate_generated_paper_outputs(repo_root)
    for summary_file in (repo_root / HIST_DIR).glob("ours_single_shot_pivot_*_summary.json"):
        move_if_exists(summary_file, repo_root / REPORT_DIR / summary_file.name)


def load_hist(path: Path) -> Dict[int, int]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    hist = {int(k): int(v) for k, v in raw.items()}
    return dict(sorted(hist.items()))


def stats(hist: Dict[int, int]) -> dict:
    total = sum(hist.values())
    if total == 0:
        return {
            "total": 0,
            "mean": math.nan,
            "median": None,
            "support": [],
        }

    weighted = sum(cost * count for cost, count in hist.items())
    mean = weighted / total

    rank = (total + 1) // 2
    running = 0
    median = None
    for cost in sorted(hist):
        running += hist[cost]
        if running >= rank:
            median = cost
            break

    return {
        "total": total,
        "mean": mean,
        "median": median,
        "support": sorted(hist.keys()),
    }


def bin5(hist: Dict[int, int]) -> Dict[int, int]:
    bins = [
        (1, 6, 1),
        (7, 12, 7),
        (13, 18, 13),
        (19, 24, 19),
        (25, 30, 25),
    ]
    out = Counter({center: 0 for _, _, center in bins})
    for cost, count in hist.items():
        for lo, hi, center in bins:
            if lo <= cost <= hi:
                out[center] += count
                break
    return dict(sorted(out.items()))


def plot_exact(path: Path, ours: Dict[int, int], paper: Dict[int, int], title: str) -> None:
    import matplotlib.pyplot as plt

    xs = sorted(set(ours) | set(paper))
    ours_y = [ours.get(x, 0) for x in xs]
    paper_y = [paper.get(x, 0) for x in xs]

    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([x - width / 2 for x in xs], paper_y, width=width, label="paper baseline")
    ax.bar([x + width / 2 for x in xs], ours_y, width=width, label="ours_single_shot")
    ax.set_xlabel("Exact total cost")
    ax.set_ylabel("Target count")
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_binned(path: Path, ours: Dict[int, int], paper: Dict[int, int], title: str) -> None:
    import matplotlib.pyplot as plt

    ours_b = bin5(ours)
    paper_b = bin5(paper)
    xs = [1, 7, 13, 19, 25]
    ours_y = [ours_b.get(x, 0) for x in xs]
    paper_y = [paper_b.get(x, 0) for x in xs]

    width = 2.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([x - width / 2 for x in xs], paper_y, width=width, label="paper baseline")
    ax.bar([x + width / 2 for x in xs], ours_y, width=width, label="ours_single_shot")
    ax.set_xlabel("Paper-style 5-bin center")
    ax.set_ylabel("Target count")
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_exact_multi(path: Path, series: list[tuple[str, Dict[int, int]]], title: str) -> None:
    import matplotlib.pyplot as plt

    xs = sorted({x for _, hist in series for x in hist})
    if not xs:
        xs = [1]

    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    start = -(len(series) - 1) / 2
    for idx, (label, hist) in enumerate(series):
        offset = (start + idx) * width
        ys = [hist.get(x, 0) for x in xs]
        ax.bar([x + offset for x in xs], ys, width=width, label=label)
    ax.set_xlabel("Exact total cost")
    ax.set_ylabel("Target count")
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_binned_multi(path: Path, series: list[tuple[str, Dict[int, int]]], title: str) -> None:
    import matplotlib.pyplot as plt

    centers = [1, 7, 13, 19, 25]
    width = 2.2 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    start = -(len(series) - 1) / 2
    for idx, (label, hist) in enumerate(series):
        hist_b = bin5(hist)
        ys = [hist_b.get(x, 0) for x in centers]
        offset = (start + idx) * width
        ax.bar([x + offset for x in centers], ys, width=width, label=label)
    ax.set_xlabel("Paper-style 5-bin center")
    ax.set_ylabel("Target count")
    ax.set_title(title)
    ax.set_xticks(centers)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_stats(label: str, hist: Dict[int, int]) -> None:
    s = stats(hist)
    print(f"[{label}]")
    print(f"  total targets: {s['total']}")
    print(f"  mean: {s['mean']:.6f}")
    print(f"  median: {s['median']}")
    print(f"  support: {s['support']}")


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()

    comparison_modes = [
        args.compare_fixed_safe,
        args.compare_paper_fixed_safe,
        args.compare_paper_fixed_best_single,
    ]
    if sum(bool(mode) for mode in comparison_modes) > 1:
        raise SystemExit(
            "Choose only one mode: --compare-fixed-safe, --compare-paper-fixed-safe, or --compare-paper-fixed-best-single"
        )

    if args.compare_paper_fixed_best_single:
        if args.generate:
            run_generate_pivot_scan(root, include_paper=True)

        paper_hist = load_hist(root / args.paper_hist)
        fixed_hist = load_hist(root / args.fixed_hist)
        best_single_hist = load_hist(root / args.best_single_hist)
        print_stats("paper-baseline", paper_hist)
        print_stats("fixed-pivot", fixed_hist)
        print_stats("best-single-pivot", best_single_hist)

        series = [
            ("paper baseline", paper_hist),
            ("fixed pivot", fixed_hist),
            ("best single pivot", best_single_hist),
        ]
        exact_path = root / FIG_DIR / "compare_paper_fixed_best_single_exact.png"
        binned_path = root / FIG_DIR / "compare_paper_fixed_best_single_binned.png"
        ensure_output_dirs(root)
        try:
            plot_exact_multi(
                exact_path,
                series,
                "11Q exact histogram: paper vs fixed vs best-single-pivot",
            )
            plot_binned_multi(
                binned_path,
                series,
                "11Q paper-style 5-bin: paper vs fixed vs best-single-pivot",
            )
            print("Wrote exact plot:")
            print(f"  {exact_path}")
            print("Wrote binned plot:")
            print(f"  {binned_path}")
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("matplotlib is not installed; skipped plot generation.")
            print("Install matplotlib to enable exact and 5-bin figure output.")

        return 0

    if args.compare_fixed_safe or args.compare_paper_fixed_safe:
        include_paper = args.compare_paper_fixed_safe
        if args.generate:
            run_generate_safe(root, include_paper=include_paper)

        fixed_hist = load_hist(root / args.fixed_hist)
        safe_hist = load_hist(root / args.safe_hist)
        print_stats("fixed-pivot", fixed_hist)
        print_stats("safe-pivot", safe_hist)

        series = [("fixed pivot", fixed_hist), ("safe pivot", safe_hist)]
        exact_path = root / FIG_DIR / "compare_safe_pivot_exact.png"
        binned_path = root / FIG_DIR / "compare_safe_pivot_binned.png"
        title_exact = "11Q exact histogram: fixed-pivot vs safe-pivot"
        title_binned = "11Q paper-style 5-bin: fixed-pivot vs safe-pivot"

        if include_paper:
            paper_hist = load_hist(root / args.paper_hist)
            print_stats("paper-baseline", paper_hist)
            series = [("paper baseline", paper_hist), ("fixed pivot", fixed_hist), ("safe pivot", safe_hist)]
            exact_path = root / FIG_DIR / "compare_paper_fixed_safe_exact.png"
            binned_path = root / FIG_DIR / "compare_paper_fixed_safe_binned.png"
            title_exact = "11Q exact histogram: paper vs fixed vs safe"
            title_binned = "11Q paper-style 5-bin: paper vs fixed vs safe"

        try:
            ensure_output_dirs(root)
            plot_exact_multi(exact_path, series, title_exact)
            plot_binned_multi(binned_path, series, title_binned)
            print("Wrote exact plot:")
            print(f"  {exact_path}")
            print("Wrote binned plot:")
            print(f"  {binned_path}")
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("matplotlib is not installed; skipped plot generation.")
            print("Install matplotlib to enable exact and 5-bin figure output.")

        return 0

    if args.generate:
        run_generate(root)

    ours_gross = load_hist(root / args.ours_gross)
    ours_two = load_hist(root / args.ours_two)
    paper_gross = load_hist(root / args.paper_gross)
    paper_two = load_hist(root / args.paper_two)

    print_stats("gross/paper", paper_gross)
    print_stats("gross/ours", ours_gross)
    print_stats("two-gross/paper", paper_two)
    print_stats("two-gross/ours", ours_two)

    try:
        ensure_output_dirs(root)
        plot_exact(
            root / FIG_DIR / "compare_11q_single_shot_exact_gross.png",
            ours_gross,
            paper_gross,
            "11Q exact histogram: gross",
        )
        plot_exact(
            root / FIG_DIR / "compare_11q_single_shot_exact_two_gross.png",
            ours_two,
            paper_two,
            "11Q exact histogram: two-gross",
        )

        plot_binned(
            root / FIG_DIR / "compare_11q_single_shot_binned_gross.png",
            ours_gross,
            paper_gross,
            "11Q paper-style 5-bin comparison: gross",
        )
        plot_binned(
            root / FIG_DIR / "compare_11q_single_shot_binned_two_gross.png",
            ours_two,
            paper_two,
            "11Q paper-style 5-bin comparison: two-gross",
        )

        print("Wrote exact plots:")
        print(f"  {(root / FIG_DIR / 'compare_11q_single_shot_exact_gross.png')}")
        print(f"  {(root / FIG_DIR / 'compare_11q_single_shot_exact_two_gross.png')}")
        print("Wrote binned plots:")
        print(f"  {(root / FIG_DIR / 'compare_11q_single_shot_binned_gross.png')}")
        print(f"  {(root / FIG_DIR / 'compare_11q_single_shot_binned_two_gross.png')}")
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipped plot generation.")
        print("Install matplotlib to enable exact and 5-bin figure output.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

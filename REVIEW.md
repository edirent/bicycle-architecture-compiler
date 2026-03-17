# Review Quickstart

## What Is Compared

- `paper baseline` (11Q paper-style beta cost)
- `ours single-shot` (11Q fixed-pivot single-shot, plus best-single-pivot scan)

Both are exposed through the shared Rust interface:

- `SingleShot11QAlgorithm`
- `PaperBaseline11Q`
- `OursSingleShot11Q`

## One-Command Validation

Run from repo root:

```bash
./scripts/check.sh
```

Default mode is **smoke** (fast review path): formatting, lint, smoke tests, checks, histogram generation, comparison plots, split benchmarks, and boss reports.

For full/heavy validation:

```bash
./scripts/full_validation.sh
# or: ./scripts/check.sh --full
```

## Where Outputs Appear

- Histograms: `results/histograms/`
- Figures: `results/figures/`
- Reports: `results/reports/`
- Benchmarks: `results/benchmarks/`

Main boss-facing report files:

- `results/reports/validation.md`
- `results/reports/validation.json`

## Paper Comparison Files

- Paper hist: `results/histograms/paper_baseline_gross_exact_hist.json`
- Ours fixed-pivot hist: `results/histograms/ours_single_shot_fixed_pivot_exact_hist.json`
- Ours best-single-pivot hist: `results/histograms/ours_single_shot_best_single_pivot_exact_hist.json`
- 5-bin figure: `results/figures/paper_vs_ours_11q_5bin.png`

## Benchmark Entry Points

- Paper baseline only: `crates/bicycle_cliffords/benches/paper_baseline_11q.rs`
- Ours single-shot only (fixed-pivot by default): `crates/bicycle_cliffords/benches/ours_single_shot_11q.rs`

Optional heavy pivot scan benchmark:

- `cargo bench --package bicycle_cliffords --bench ours_single_shot_11q -- --csv <path> --include-pivot-scan`

## Known Limits

- Figure PNG generation depends on `matplotlib` in Python; if unavailable, comparison still runs and reports figure paths with `exists=false`.

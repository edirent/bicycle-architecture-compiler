# Boss Validation

- git commit: `0a0c712d17574fa1204d0be28352874150cebe84`
- mode: `smoke`
- all commands passed: `True`
- generated at (UTC): `2026-03-17T15:50:24.569864+00:00`
- all checks passed: `True`

## Means

- paper baseline mean (gross): `18.526206`
- paper baseline mean (two-gross): `18.481711`
- ours single-shot mean (gross): `12.952657`
- ours single-shot mean (two-gross): `12.953588`
- ours fixed-pivot mean: `12.952657`
- ours best-single-pivot mean: `9.352265`

## Checks

- [PASS] `paper_total_is_4_pow_11`: paper_total=4194304 expected=4194304
- [PASS] `ours_total_is_4_pow_11`: ours_total=4194304 expected=4194304
- [PASS] `local_measurement_count_hist_present`: present=True
- [PASS] `local_measurement_count_keys_subset`: local_keys=[4, 6] allowed=[1, 2, 4, 6]
- [PASS] `local_measurement_count_no_5`: local_keys=[4, 6]
- [PASS] `local_measurement_count_no_ge_7`: local_keys=[4, 6]
- [PASS] `fixed_vs_best_outputs_separated`: fixed=/home/edirent/bicycle-architecture-compiler/results/histograms/ours_single_shot_fixed_pivot_exact_hist.json exists=True ; best=/home/edirent/bicycle-architecture-compiler/results/histograms/ours_single_shot_best_single_pivot_exact_hist.json exists=True
- [PASS] `benchmark_entry_paper_available`: path=/home/edirent/bicycle-architecture-compiler/results/benchmarks/paper_baseline_bench.txt
- [PASS] `benchmark_entry_ours_available`: path=/home/edirent/bicycle-architecture-compiler/results/benchmarks/ours_single_shot_bench.txt

## Paths

- `results/histograms/paper_baseline_gross_exact_hist.json`
- `results/histograms/paper_baseline_two_gross_exact_hist.json`
- `results/histograms/ours_single_shot_fixed_pivot_exact_hist.json`
- `results/histograms/ours_single_shot_best_single_pivot_exact_hist.json`
- figures:
- `results/figures/paper_vs_ours_11q_5bin.png` (exists=False)
- `results/figures/compare_11q_single_shot_exact_gross.png` (exists=False)
- `results/figures/compare_11q_single_shot_binned_gross.png` (exists=False)
- `results/figures/compare_paper_fixed_best_single_binned.png` (exists=False)
- benchmark outputs:
- `results/benchmarks/paper_baseline_bench.txt`
- `results/benchmarks/ours_single_shot_bench.txt`

## Commands Run

- `cargo fmt --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --workspace -- --nocapture --skip decomposition::tests::test_gross_table --skip decomposition::tests::test_twogross_table --skip draft_single_shot::tests::local_delta_hygiene_holds_for_all_valid_pivots_in_scan --skip draft_single_shot::tests::pivot_scan_rejects_invalid_candidate_and_keeps_pivot1_valid --skip tests::qubit_measurements_are_native --skip compile::tests::measurement::compile_multiblock --skip compile::tests::measurement::compile_native_joint_measurement --skip compile::tests::rotation::compile_multiblock --skip compile::tests::rotation::compile_native_rotation --skip test::integration_test_rotation`
- `cargo check --workspace`
- `cargo run --release --bin bicycle_compiler -- --paper-beta-report --csv native.csv --paper-hist-out results/histograms/paper_baseline_gross_exact_hist.json --paper-summary-out results/reports/paper_baseline_gross_summary.json`
- `cargo run --release --bin ours_single_shot -- --csv native.csv --out results/histograms/ours_single_shot_gross`
- `mv -f results/histograms/ours_single_shot_gross_summary.json results/reports/ours_single_shot_gross_summary.json`
- `cargo run --release --bin bicycle_compiler -- two-gross --paper-beta-report --paper-hist-out results/histograms/paper_baseline_two_gross_exact_hist.json --paper-summary-out results/reports/paper_baseline_two_gross_summary.json`
- `cargo run --release --bin ours_single_shot -- --code two-gross --out results/histograms/ours_single_shot_two_gross`
- `mv -f results/histograms/ours_single_shot_two_gross_summary.json results/reports/ours_single_shot_two_gross_summary.json`
- `cargo run --release --bin ours_single_shot -- --csv native.csv --ours-pivot-scan-report --pivot-scan-summary-out results/reports/pivot_scan_summary.json --ours-single-shot-fixed-pivot-hist-out results/histograms/ours_single_shot_fixed_pivot_exact_hist.json --ours-single-shot-best-single-pivot-hist-out results/histograms/ours_single_shot_best_single_pivot_exact_hist.json --ours-single-shot-best-single-pivot-summary-out results/reports/ours_single_shot_best_single_pivot_summary.json --ours-single-shot-pivot-prefix results/histograms/ours_single_shot_pivot`
- `python3 compare_11q_single_shot.py --repo-root /home/edirent/bicycle-architecture-compiler`
- `python3 compare_11q_single_shot.py --repo-root /home/edirent/bicycle-architecture-compiler --compare-paper-fixed-best-single --paper-hist results/histograms/paper_baseline_gross_exact_hist.json --fixed-hist results/histograms/ours_single_shot_fixed_pivot_exact_hist.json --best-single-hist results/histograms/ours_single_shot_best_single_pivot_exact_hist.json`
- `cargo bench --package bicycle_cliffords --bench paper_baseline_11q -- --csv native.csv --pivot-index 0 > results/benchmarks/paper_baseline_bench.txt`
- `cargo bench --package bicycle_cliffords --bench ours_single_shot_11q -- --csv native.csv --pivot-index 0 > results/benchmarks/ours_single_shot_bench.txt`
- `python3 scripts/generate_boss_validation.py --repo-root /home/edirent/bicycle-architecture-compiler --commands-log results/reports/boss_commands.log --mode smoke --all-commands-passed`

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="smoke"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: ./scripts/boss_check.sh [--full]" >&2
  exit 1
fi

mkdir -p results/histograms results/figures results/reports results/benchmarks

COMMAND_LOG="results/reports/boss_commands.log"
: > "${COMMAND_LOG}"

run_cmd() {
  echo "$*" | tee -a "${COMMAND_LOG}"
  "$@"
}

resolve_existing_csv() {
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Neither python nor python3 is available in PATH." >&2
  exit 1
fi

GROSS_CSV="$(resolve_existing_csv "${BOSS_CSV_GROSS:-}" "native_dictionary_gross.csv" "native.csv" || true)"
TWO_GROSS_CSV="$(resolve_existing_csv "${BOSS_CSV_TWO_GROSS:-}" "native_dictionary_two_gross.csv" || true)"
BENCH_CSV="$(resolve_existing_csv "${BOSS_BENCH_CSV:-}" "${GROSS_CSV:-}" "native.csv" || true)"

if [[ -z "${BENCH_CSV}" ]]; then
  echo "No CSV available for benchmarks. Set BOSS_BENCH_CSV or provide native.csv." >&2
  exit 1
fi

run_cmd cargo fmt --check
run_cmd cargo clippy --all-targets --all-features -- -D warnings
if [[ "${MODE}" == "full" ]]; then
  run_cmd cargo test --workspace -- --nocapture
else
  run_cmd cargo test --workspace -- --nocapture \
    --skip decomposition::tests::test_gross_table \
    --skip decomposition::tests::test_twogross_table \
    --skip draft_single_shot::tests::local_delta_hygiene_holds_for_all_valid_pivots_in_scan \
    --skip draft_single_shot::tests::pivot_scan_rejects_invalid_candidate_and_keeps_pivot1_valid \
    --skip tests::qubit_measurements_are_native \
    --skip compile::tests::measurement::compile_multiblock \
    --skip compile::tests::measurement::compile_native_joint_measurement \
    --skip compile::tests::rotation::compile_multiblock \
    --skip compile::tests::rotation::compile_native_rotation \
    --skip test::integration_test_rotation
fi
run_cmd cargo check --workspace

if [[ -n "${GROSS_CSV}" ]]; then
  run_cmd cargo run --release --bin bicycle_compiler -- --paper-beta-report --csv "${GROSS_CSV}" --paper-hist-out results/histograms/paper_baseline_gross_exact_hist.json --paper-summary-out results/reports/paper_baseline_gross_summary.json
  run_cmd cargo run --release --bin ours_single_shot -- --csv "${GROSS_CSV}" --out results/histograms/ours_single_shot_gross
else
  run_cmd cargo run --release --bin bicycle_compiler -- gross --paper-beta-report --paper-hist-out results/histograms/paper_baseline_gross_exact_hist.json --paper-summary-out results/reports/paper_baseline_gross_summary.json
  run_cmd cargo run --release --bin ours_single_shot -- --code gross --out results/histograms/ours_single_shot_gross
fi
run_cmd mv -f results/histograms/ours_single_shot_gross_summary.json results/reports/ours_single_shot_gross_summary.json

if [[ -n "${TWO_GROSS_CSV}" ]]; then
  run_cmd cargo run --release --bin bicycle_compiler -- --paper-beta-report --csv "${TWO_GROSS_CSV}" --paper-hist-out results/histograms/paper_baseline_two_gross_exact_hist.json --paper-summary-out results/reports/paper_baseline_two_gross_summary.json
  run_cmd cargo run --release --bin ours_single_shot -- --csv "${TWO_GROSS_CSV}" --out results/histograms/ours_single_shot_two_gross
else
  run_cmd cargo run --release --bin bicycle_compiler -- two-gross --paper-beta-report --paper-hist-out results/histograms/paper_baseline_two_gross_exact_hist.json --paper-summary-out results/reports/paper_baseline_two_gross_summary.json
  run_cmd cargo run --release --bin ours_single_shot -- --code two-gross --out results/histograms/ours_single_shot_two_gross
fi
run_cmd mv -f results/histograms/ours_single_shot_two_gross_summary.json results/reports/ours_single_shot_two_gross_summary.json

if [[ -n "${GROSS_CSV}" ]]; then
  run_cmd cargo run --release --bin ours_single_shot -- --csv "${GROSS_CSV}" --ours-pivot-scan-report --pivot-scan-summary-out results/reports/pivot_scan_summary.json --ours-single-shot-fixed-pivot-hist-out results/histograms/ours_single_shot_fixed_pivot_exact_hist.json --ours-single-shot-best-single-pivot-hist-out results/histograms/ours_single_shot_best_single_pivot_exact_hist.json --ours-single-shot-best-single-pivot-summary-out results/reports/ours_single_shot_best_single_pivot_summary.json --ours-single-shot-pivot-prefix results/histograms/ours_single_shot_pivot
else
  run_cmd cargo run --release --bin ours_single_shot -- --code gross --ours-pivot-scan-report --pivot-scan-summary-out results/reports/pivot_scan_summary.json --ours-single-shot-fixed-pivot-hist-out results/histograms/ours_single_shot_fixed_pivot_exact_hist.json --ours-single-shot-best-single-pivot-hist-out results/histograms/ours_single_shot_best_single_pivot_exact_hist.json --ours-single-shot-best-single-pivot-summary-out results/reports/ours_single_shot_best_single_pivot_summary.json --ours-single-shot-pivot-prefix results/histograms/ours_single_shot_pivot
fi
for pivot_summary in results/histograms/ours_single_shot_pivot_*_summary.json; do
  [[ -e "${pivot_summary}" ]] || break
  run_cmd mv -f "${pivot_summary}" "results/reports/$(basename "${pivot_summary}")"
done

run_cmd "${PYTHON_BIN}" compare_11q_single_shot.py --repo-root "${REPO_ROOT}"
run_cmd "${PYTHON_BIN}" compare_11q_single_shot.py --repo-root "${REPO_ROOT}" --compare-paper-fixed-best-single --paper-hist results/histograms/paper_baseline_gross_exact_hist.json --fixed-hist results/histograms/ours_single_shot_fixed_pivot_exact_hist.json --best-single-hist results/histograms/ours_single_shot_best_single_pivot_exact_hist.json

PAPER_BENCH_CMD=(
  cargo bench --package bicycle_cliffords --bench paper_baseline_11q -- --csv "${BENCH_CSV}" --pivot-index 0
)
echo "${PAPER_BENCH_CMD[*]} > results/benchmarks/paper_baseline_bench.txt" | tee -a "${COMMAND_LOG}"
"${PAPER_BENCH_CMD[@]}" > results/benchmarks/paper_baseline_bench.txt

OURS_BENCH_CMD=(
  cargo bench --package bicycle_cliffords --bench ours_single_shot_11q -- --csv "${BENCH_CSV}" --pivot-index 0
)
if [[ "${MODE}" == "full" ]]; then
  OURS_BENCH_CMD+=(--include-pivot-scan)
fi
echo "${OURS_BENCH_CMD[*]} > results/benchmarks/ours_single_shot_bench.txt" | tee -a "${COMMAND_LOG}"
"${OURS_BENCH_CMD[@]}" > results/benchmarks/ours_single_shot_bench.txt

run_cmd "${PYTHON_BIN}" scripts/generate_boss_validation.py --repo-root "${REPO_ROOT}" --commands-log "${COMMAND_LOG}" --mode "${MODE}" --all-commands-passed

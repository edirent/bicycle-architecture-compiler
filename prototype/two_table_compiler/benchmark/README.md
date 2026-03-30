# Benchmark Compare

`prototype/two_table_compiler/benchmark_compare.py` compares:

1. `gross_baseline`: previous gross/rotation baseline
2. `ours_no_joint`: our Module-I prototype with `P2` disabled (`P0 + P3`)
3. `ours_full`: our Module-I prototype with full `P0 + P2 + P3`

The benchmark does not change core search semantics (`State(hidden, tail)`, Dijkstra).

## Baseline and Inventory Sources

- Baseline per-tail costs:
  - file: `results/beta_headed_min_11q.json`
  - exporter: `crates/bicycle_cliffords/src/bin/export_beta_baseline_11q.rs`
  - generation path: `BetaBaseline::compute_exact()`
- Gross native inventory:
  - file: `results/native_11q_real.csv`
  - exporter: `crates/bicycle_cliffords/src/bin/export_native_11q.rs`

The Python benchmark reuses offline artifacts only. If a required JSON/CSV file is missing, the
benchmark fails fast and does not invoke `cargo run`.

## Four-State Search Semantics

All benchmark, report, and story artifacts must be interpreted through `synthesize_search()`:

- `FOUND_OPTIMAL`
  - reachable
  - optimal
- `FOUND_REACHABLE_UPPER_BOUND`
  - reachable
  - upper bound found, not yet proven optimal
- `SEARCH_TRUNCATED`
  - not reachable yet under the current budget
  - not rule-unreachable
- `RULE_UNREACHABLE`
  - only this status may be reported as `frame-unreachable` / `rule-unreachable`

No-budget vs finite-budget:

- `--max-popped-states 0` means no budget limit
- no-budget runs must not emit `FOUND_REACHABLE_UPPER_BOUND` or `SEARCH_TRUNCATED`
- finite-budget runs may emit `FOUND_REACHABLE_UPPER_BOUND` or `SEARCH_TRUNCATED`

Semantic hygiene constraints:

- `SEARCH_TRUNCATED != RULE_UNREACHABLE`
- `FOUND_REACHABLE_UPPER_BOUND` must not be flattened to unreachable
- report code must derive reachability from `search_status`, never from `plan is None`
- `synthesize()` remains a legacy compatibility wrapper only; benchmark/report/story code must use `synthesize_search()`
- `target-aware P2` remains post-filter only and must not change the full source graph
- if the rule set is theoretically complete, `RULE_UNREACHABLE` should be treated as an execution bug signal

## Cost Unit

- `gross_baseline`: `beta_headed_min` (in-module bicycle measurement count)
- `ours_*`: prototype shortest-path `total_cost` / `beta(t)`

## Commands

Sample run:

```bash
python -m prototype.two_table_compiler.benchmark_compare \
  --mode sample \
  --seed 7 \
  --count 100 \
  --max-weight 3 \
  --max-popped-states 0
```

Finite-budget smoke run:

```bash
python -m prototype.two_table_compiler.benchmark_compare \
  --mode sample \
  --seed 7 \
  --count 100 \
  --max-weight 3 \
  --max-popped-states 2000 \
  --progress-every 100
```

Ablation (`ours_full` without `P2`):

```bash
python -m prototype.two_table_compiler.benchmark_compare \
  --mode sample \
  --seed 7 \
  --count 100 \
  --disable-p2
```

## Output Artifacts

Results are written to `benchmark/results/<timestamp>/`:

- `summary.json`
- `per_target.csv`
- `local_inventory_diff.csv`

`per_target.csv` includes per target:

- `search_status_gross`, `search_status_ours_no_joint`, `search_status_ours_full`
- `reachable_*`, `truncated_*`, `rule_unreachable_*`, `optimal_*`
- `gross_baseline_cost`, `ours_no_joint_cost`, `ours_full_cost`
- first-step family/scope, final hidden class, P3 step count
- cross-logical/native and inter-module-bell usage flags
- `delta_vs_baseline`, `delta_joint_gain`
- `max_popped_states`, `used_no_budget`

`summary.json` includes:

- per-mode statistics by stratum:
  - `all`, `direct_local`, `joint_source`, `cross_logical_native`, `frame_unreachable`, `search_truncated`
- per-mode `counts_by_status` for all four statuses
- pairwise comparisons:
  - computed only on common-reachable targets
  - `ours_full_vs_gross_baseline`
  - `ours_full_vs_ours_no_joint`
- special counts:
  - first solved by `P2`
  - solved by `cross_logical_block_native`
  - best path without `P2`
  - `gross_reachable_and_ours_full_rule_unreachable`
  - `gross_reachable_and_ours_full_truncated`
  - `ours_full_reachable_and_ours_no_joint_rule_unreachable`
  - `ours_full_reachable_and_ours_no_joint_truncated`
- local inventory overlap/delta summary
- semantic notice that old pre-four-state artifacts are stale and not comparable

## Simple Story Report

After a benchmark run, generate a small narrative report from the real `per_target.csv`:

```bash
python -m prototype.two_table_compiler.benchmark_simple_story \
  --input-dir benchmark/results/<timestamp>
```

This writes:

- `simple_targets.csv` (6 representative targets selected as mandatory probes + audited sample fill)
- `simple_circuits.csv` (3 tiny toy circuits built only from verified representative targets)
- `story_report.md` (short narrative markdown)
- `figure_story_costs.png`
- `figure_story_circuits.png`

Story-specific guarantees:

- includes a real `cross_logical_native_case` (`X1P⊗X1P'` when available)
- includes a real `joint_source_win_case`
- never invents a fake `frame_unreachable_case`
- if no true `RULE_UNREACHABLE` appears under no-budget search, the report says so explicitly

## Audits

Replay old unreachable-like labels under no-budget four-state search:

```bash
python -m prototype.two_table_compiler.audit_old_unreachable \
  --input benchmark/results/<old-run>/per_target.csv
```

Run the strict small-weight canary:

```bash
python -m unittest prototype.two_table_compiler.test_weight2_canary
```

If theory says the current rules cover all `4^11`, then `weight<=2` should be all `FOUND_OPTIMAL`.
Any `RULE_UNREACHABLE` there should be treated as an execution bug signal and investigated with the
generated debug CSV.

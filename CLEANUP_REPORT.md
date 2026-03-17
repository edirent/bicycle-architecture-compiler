# Repository Cleanup Report

## Scope and safety
This pass was conservative and focused on cache/build/output hygiene only.
No Rust source in `crates/`, no Python experiment scripts, and no raw input CSV data were deleted.

## Summary
- Directories created:
  - `results/`
  - `results/figures/`
  - `results/histograms/`
  - `results/reports/`
  - `scratch/`
- Cache/build directories deleted:
  - `target/`
  - `__pycache__/`
  - `notebooks/__pycache__/`
  - `pylib/__pycache__/`
- Generated artifacts moved from repo root into canonical locations:
  - figures (`*.svg`) -> `results/figures/`
  - histogram JSONs (`*_exact_hist.json`, `*_5bin_hist.json`) -> `results/histograms/`
  - summary/certification/report JSONs -> `results/reports/`
- `.gitignore` updated for recurring local junk and temporary outputs.

## Large-file scan
Threshold scan after cleanup (`>5MB`) found only local virtualenv files:

| Path | Size (bytes) | Category | Action |
|---|---:|---|---|
| `.venv/lib/python3.13/site-packages/numpy.libs/libscipy_openblas64_-2e5251cf.so` | 28,699,769 | local environment | kept (ambiguous runtime dependency; `.venv` is gitignored) |
| `.venv/lib/python3.13/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/inject_dll_x86.pdb` | 5,902,336 | local environment | kept |
| `.venv/lib/python3.13/site-packages/numpy/_core/_multiarray_umath.cpython-313-aarch64-linux-gnu.so` | 5,787,449 | local environment | kept |
| `.venv/lib/python3.13/site-packages/debugpy/_vendored/pydevd/pydevd_attach_to_process/inject_dll_amd64.pdb` | 5,779,456 | local environment | kept |

Repo size reduced from approximately `1.8G` to `196M` after cache/build cleanup.

## .gitignore changes
Added or corrected ignore coverage for:
- `.DS_Store`
- `*.pyc`
- `.ipynb_checkpoints/`
- `*.swp`, `*.swo`
- `tmp/`, `temp/`, `cache/`, `scratch/`
- `results/generated_tmp/`

Also removed broad `*.csv` ignore to avoid suppressing important raw input CSV datasets.

## Sanity checks
- `python3 -m py_compile compare_11q_single_shot.py` passed.
- `python3 compare_11q_single_shot.py --compare-paper-fixed-best-single --repo-root .` passed (reads from canonical `results/` paths).
- `cargo check -p bicycle_cliffords --bin ours_single_shot` passed before final cache purge.

## Manifest
Machine-readable manifest written to `cleanup_report.json` with:
- `deleted_files`
- `moved_files`
- `kept_large_files`
- `ignored_patterns_added`
- `directories_created`

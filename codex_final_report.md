# Codex Final Report

## 1. What was built

- `export_logical_natives.py` exports all gross-code logical native measurements in logical 12-qubit symplectic space and writes `logical_natives_gross.json`.
- `export_gross_lift.py` reproduces the gross physical CSS checks and the matching physical logical basis from the notebook `system_3` construction and writes `gross_physical_lift.json`.
- `stab_hit_test_lifted.py` performs the lifted physical-space stabilizer-equivalence test and writes `lifted_hit_test_result.json`.

## 2. Exact files created

- `export_logical_natives.py`
- `logical_natives_gross.json`
- `export_gross_lift.py`
- `gross_physical_lift.json`
- `stab_hit_test_lifted.py`
- `lifted_hit_test_result.json`
- `codex_audit_report.md`
- `codex_final_report.md`

No tracked repo files were modified.

## 3. Exact commands run

- `rg -n "GROSS_MEASUREMENT|CodeMeasurement|NativeMeasurement|measures" .`
- `rg -n "gross_code_automorphisms|get_block_basis_with_pivot|get_block_basis_with_shifts|A\\.mat|B\\.mat" .`
- `rg -n "Polynomial\\.mat|Polynomial\\.vec" .`
- `rg -n "Hx|Hz|stabilizer|logical basis|automorphism" .`
- `cargo test -p bicycle_cliffords all_native -- --nocapture`
- `cargo run -p bicycle_cliffords -- --help`
- `python3 export_logical_natives.py`
- `python3 export_gross_lift.py`
- `python3 stab_hit_test_lifted.py --target "XIIIIIIXIIII" --logical-natives logical_natives_gross.json --physical-lift gross_physical_lift.json --show-cert`
- `python3 stab_hit_test_lifted.py --target "XIIIIIXIIIII" --logical-natives logical_natives_gross.json --physical-lift gross_physical_lift.json --show-cert --out /tmp/native_hit_check.json`

## 4. Whether the lifted experiment is now valid

Yes, with the following explicit conventions:

- Logical native measurements live in logical 12-qubit symplectic space `(x1..x12 | z1..z12)`.
- Physical checks and lifted representatives live in physical CSS / physical symplectic space over 144 physical qubits.
- Logical Pauli strings in the new scripts and JSON artifacts use left-to-right logical qubit order `1..12`.
- The lift basis is the gross notebook `system_3` basis, because the saved notebook output `notebooks/gross_code_automorphisms.ipynb:383-410` matches the Rust gross automorphism matrices in `crates/bicycle_cliffords/src/measurement.rs:71-88`.

The physical lift basis validates cleanly:

- `rank(Hx) = 66`
- `rank(Hz) = 66`
- `k = 144 - 66 - 66 = 12`
- CSS commutation, logical-vs-stabilizer commutation, and logical `X/Z` symplectic pairing checks all pass.

## 5. Result for target `XIIIIIIXIIII`

- `lifted_hit_test_result.json` reports `num_hits = 0`.
- The target string `XIIIIIIXIIII` is not present in `logical_natives_gross.json`.
- A sanity check on the native target `XIIIIIXIIIII` returns exactly one hit, itself, with an empty stabilizer certificate because the lifted delta is zero.

## 6. Remaining limitations

- The workspace Python does not have `numpy`, so the new scripts reproduce the repo-backed algebra in pure Python instead of importing `pylib/automor.py` or executing notebook helpers directly.
- The exported physical basis is tied to the gross `system_3` notebook construction. That choice is evidence-backed, but the repo does not provide a standalone library module for this lift, so the basis derivation still originates from notebook logic.

## 7. Unresolved repo gaps

- The raw source of `notebooks/gross_code_automorphisms.ipynb` currently defaults its top configuration cell to `two-gross` (`notebooks/gross_code_automorphisms.ipynb:52-69`), while the saved output for cell 11 is a gross-code `system_3` run (`notebooks/gross_code_automorphisms.ipynb:383-410`). The gross lift here relies on that saved output plus the `system_3` source cell (`notebooks/gross_code_automorphisms.ipynb:415-453`) because those matrices match the Rust gross matrices exactly.
- There is no existing repo module that exposes the physical lift as a normal API. The new export script is therefore a notebook-to-script reconstruction rather than a thin wrapper over an existing module.

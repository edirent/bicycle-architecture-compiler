# Codex Audit Report

## Compiler entrypoints and selector-string evidence

- `bootstrap_code.py:1` contains only `code = "gross"`, which matches the user-supplied constraint that notebook state uses a selector string rather than a Python code object.
- `notebooks/custom_circuits.ipynb:72-72` sets `code = "gross"`, `notebooks/custom_circuits.ipynb:90-98` builds `measurement_table_path = f"table_{code}.dat"`, and `notebooks/custom_circuits.ipynb:120-127` defines `compile_pbc_circuit(...)` by shelling out to `../target/release/bicycle_compiler`, passing `code_` and `--measurement-table table_{code_}.dat`.
- `crates/bicycle_compiler/src/main.rs:38-44` defines the CLI code selector and optional `--measurement-table`; `crates/bicycle_compiler/src/main.rs:70-143` either generates or loads the measurement table based on that selector. This is the repo-backed evidence that compilation is driven by the code string plus table file, not by a Python-resident code object.

## Logical native-measurement generation

- `crates/bicycle_common/src/lib.rs:186-208` defines `TwoBases` as a pair `(p1, p7)` with only `(I, I)` excluded.
- `crates/bicycle_cliffords/src/native_measurement.rs:39-72` constructs all native measurements by iterating all 36 shift automorphisms and all 15 valid `TwoBases` choices.
- `crates/bicycle_cliffords/src/measurement.rs:29-68` defines the gross logical-space measurement map as a 24-bit symplectic vector `(map_x1, map_x7, map_z1, map_z7)` using `x_action = action(inv)` and `z_action = action(...).transpose()`.
- `crates/bicycle_cliffords/src/measurement.rs:71-88` gives the authoritative gross `mx` and `my` matrices.
- `crates/bicycle_cliffords/src/measurement.rs:268-281` asserts that the 540 enumerated gross native measurements remain 540 after deduplication, so the expected final gross native count is repo-backed.
- `pylib/automor.py:57-119` reproduces the same logical-space map in Python and hard-codes the same gross `mx` and `my` matrices, confirming the Rust/Python agreement for logical native measurements.

## Physical gross-code basis and stabilizer source

- `notebooks/gross_code_automorphisms.ipynb:49-60` gives the toric gross-code order `(12, 6)` and the gross CSS polynomials `A = 1 + y + x**3*y**-1` and `B = 1 + x + x**-1*y**-3`.
- `notebooks/gross_code_automorphisms.ipynb:84-86` defines anticommutation as `1 in Lx * Lz.T + Rx * Rz.T`, i.e. the physical-support overlap rule used throughout the notebook.
- `notebooks/gross_code_automorphisms.ipynb:102-106` defines `check_logical_qubit(...)`, explicitly checking that candidate logical operators anticommute with each other and commute with the X and Z stabilizer families generated from `A`, `B`, `B.T`, and `A.T`.
- `notebooks/gross_code_automorphisms.ipynb:187-210` defines `get_block_basis_with_shifts(...)`, including the documented ZX-duality rule for generating the dual block basis from the primal block basis.
- `notebooks/gross_code_automorphisms.ipynb:415-453` defines the gross `system_3` toric logical operators and the exact `xshifts` and `zshifts` used to build the basis.
- `notebooks/gross_code_automorphisms.ipynb:383-410` is the saved notebook output for `get_automorphisms_for_system_3()`. Its printed `x` and `y` matrices exactly match `crates/bicycle_cliffords/src/measurement.rs:71-88`, so this notebook cell is the repo-backed physical-basis source that aligns with the compiler’s gross logical basis.

## Polynomial and matrix helpers

- `notebooks/polynomial.py:323-381` supplies the notebook’s monomial iteration order plus `mat()` and `vec()` conversions, which are the physical-support-to-matrix/vector conversion path used by the notebook.
- `notebooks/matrix_utils.py:19-90` supplies the row-echelon and decomposition helpers used by the notebook’s pivot-search basis path. The final gross implementation here did not need that path because the matching gross basis comes from `system_3` plus fixed shifts, but these helpers are still part of the repo-backed derivation toolkit.

## Audit conclusion

The repo cleanly separates logical native measurements from physical CSS structure. The authoritative logical native map comes from `measurement.rs` / `pylib/automor.py`, while the physical lift basis comes from the notebook’s gross `system_3` construction. The one noteworthy repo ambiguity is that the top notebook configuration cell currently defaults to `two-gross` in raw source (`notebooks/gross_code_automorphisms.ipynb:52-69`), but the saved output for cell 11 is a gross-code `system_3` run whose matrices exactly match the Rust gross matrices. That saved output is therefore the matching evidence used for the gross lift.

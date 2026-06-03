# Bicycle Architecture Compiler

[Tour de Gross arXiv:2506.03094](https://arxiv.org/abs/2506.03094) presents the Bicycle Architecture - a modular architecture for fault-tolerant quantum computation based on bivariate bicycle codes. The analysis features:

 - A universal Instruction Set Archirecture (ISA) of logical gates in bivariate bicyle codes
 - Circuit-level noise benchmarks of all the ISA instructions
 - An end-to-end compilation scheme from Pauli-Based Computation (PBC) to the bicycle ISA

This compiler prototype implements and benchmarks this end-to-end compilation scheme so that it may inform future Qiskit development for fault-tolerant architectures.
Some example use cases are:
 - Random PBC circuits as a form of worst-case Clifford+T quantum circuits. See [`./notebooks/random_circuits.ipynb`](./notebooks/random_circuits.ipynb)
 - User-specified PBC circuits supporting arbitrary-angle rotations, see [`./notebooks/custom_circuits.ipynb`](./notebooks/custom_circuits.ipynb)
 - Qiskit circuits to Gross code ISA, see [`./scripts/qiskit_demo.sh`](./scripts/qiskit_demo.sh)

## Installation

This software is tested on some Linux and macOS platforms.

### Rust

We recommend using [rustup](https://www.rust-lang.org/tools/install) to install a rust toolchain.
Then run
```sh
shell> cargo build --release
```
Binaries are built and placed in `./target/release/`.
In the documentation we use these binaries without including their (relative) path.
Alternatively, you can use `cargo run --release --package <packageName> -- <arguments>`.

### Gridsynth

The compiler depends on the `gridsynth` binary for synthesizing angles other than $\pm\pi/4$.
Gridsynth can be installed by following [the installation instructions](https://www.mathstat.dal.ca/~selinger/newsynth/).
Please ensure `gridsynth` is available on your path.

To test your installation, the following command should succeed
```sh
shell> gridsynth -p --epsilon 1e-4 0.1
```
and print something like (the exact output may differ)
```
HTSHTSHTSHTSHTHTSHTSHTSHTSHTHTHTSHTHTSHTHTSHTSHTSHTHTSHTSHTHTHTSHTSHTHTHTSHTHTHTHTSHTSHTS
```

### Optional dependencies

To run the notebooks,
we require the packages `numpy`, `matplotlib`, and `jupyter`.
These can be installed via
```sh
shell> pip install -r optional_dependencies.txt
```

Furthermore, the following commandline applications are helpful and may be required for some functionality:
* `jq` - commandline JSON processor, or parsing JSON to a newline-delimited input.
* GNU `parallel` in order to
  run [./scripts/run_random_numerics.sh](./scripts/run_random_numerics.sh) in parallel.

## Usage

This Rust workspace consists of many Rust crates (packages) that can be used either as library
or, in many cases, also as a binary.
The contents of the repository as summarized as follows.

```
pbc-compiler/
├── analysis/                      # Python modules for postprocessing and steering
├── scripts/                       # Scripts for generating data (has some overlap with analysis)
├── notebooks/                     # Notebook for running and plotting a random circuit experiment.
├── data/                          # Cached measurement tables, and random circuit data
└── crates/
    ├── bicycle_common/            # Common definitions. Bicycle instructions.
    ├── bicycle_benchmark/         # Random generation of PBC circuits
    ├── bicycle_cliffords/         # Clifford gate implementation via search
    ├── bicycle_compiler/          # PBC to bicycle circuit compiler
    ├── bicycle_numerics/          # Additive noise estimates and stats collection
    └── bicycle_random_numerics/   # Benchmarking via random PBC circuits
```

Each crate has more info in their respective READMEs.

Many binary crates can be used via their compiled binaries (obtained by `cargo build` or `cargo run --package <package>`).
For an example workflow that generates benchmarks see [scripts/README.md](scripts/),
and more advanced examples are illustrated by Jupyter notebooks in `./notebooks/`.

## Local Gate-Measurement MVP

The `lgm_compiler` Python package is a research MVP front end for local
gate-measurement compilation. It is not a full Pauli-based computation compiler
and does not implement LDPC surgery or bicycle instruction scheduling.

The MVP keeps only local single-qubit Clifford frames. Local Clifford gates
update those frames, while `T`, `Tdg`, `RX`, `RY`, and `RZ` emit structured
Pauli-axis `Rot` operations under the current local frame. `CNOT` and `TOFFOLI`
consume their local frames and remain structured `FramedCNOT` and
`FramedToffoli` operations instead of being commuted into a global Clifford
frame. This avoids eagerly materializing high-weight Pauli strings for later
non-Clifford gates.

This is useful for gross-code experiments because direct `T` injection is
available on pivot logical indices 1 and 7, while arbitrary high-weight Pauli
measurement synthesis is expensive. The included cost model is intentionally a
stub: it counts direct pivot injections, non-pivot rotations, X/Z two-qubit
measurements, structured CNOT/Toffoli macros, and explicit PBC fallbacks.

Run the Python MVP tests and demo with:

```sh
pytest -q
python examples/demo_screenshot.py
python examples/count_cliffords.py
```


### Testing

To test in release mode, run

```sh
shell> cargo test --release
```

The script [./scripts/local_QA.sh](./scripts/local_QA.sh) runs quality assurance tests locally.
This includes test, rustfmt, and clippy.

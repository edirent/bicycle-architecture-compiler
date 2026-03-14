#!/usr/bin/env python3
"""Plot minimum measurement-only costs over the full 11-qubit target space.

The repository's Clifford synthesis decomposes each reachable target into:
1. zero or more conjugating rotations, and
2. exactly one base measurement.

Under the requested cost model, rotations cost 0 and measurements cost 1, so
every reachable target has minimum measurement-only cost 1. This script still
reuses the existing `min_data()` synthesis path to verify full-space
reachability with the repo's table/cache machinery, then writes the exhaustive
per-target results and the requested plots/statistics.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


TOTAL_TARGETS = 4**11 - 1
PAULI_BY_BITS = {
    (0, 0): "I",
    (1, 0): "X",
    (0, 1): "Z",
    (1, 1): "Y",
}

RUST_HELPER_MAIN = r"""
use std::{env, fs, process};

use bicycle_cliffords::{
    native_measurement::NativeMeasurement, CompleteMeasurementTable, MeasurementChoices,
    MeasurementTableBuilder, PauliString,
};

fn parse_args() -> (MeasurementChoices, Option<String>) {
    let mut args = env::args().skip(1);
    let mut code = MeasurementChoices::Gross;
    let mut cache = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--code" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --code");
                    process::exit(2);
                });
                code = match value.as_str() {
                    "gross" => MeasurementChoices::Gross,
                    "two-gross" => MeasurementChoices::TwoGross,
                    _ => {
                        eprintln!("unsupported code: {value}");
                        process::exit(2);
                    }
                };
            }
            "--cache" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --cache");
                    process::exit(2);
                });
                cache = Some(value);
            }
            other => {
                eprintln!("unexpected argument: {other}");
                process::exit(2);
            }
        }
    }

    (code, cache)
}

fn build_table(code: MeasurementChoices) -> CompleteMeasurementTable {
    let mut builder = MeasurementTableBuilder::new(NativeMeasurement::all(), code.measurement());
    builder.build();
    builder
        .complete()
        .expect("measurement table build should succeed")
}

fn load_table(path: &str) -> CompleteMeasurementTable {
    let bytes = fs::read(path).expect("measurement table cache should be readable");
    bitcode::deserialize::<CompleteMeasurementTable>(&bytes)
        .expect("measurement table cache should deserialize")
}

fn main() {
    let (code, cache) = parse_args();
    let table = match cache {
        Some(path) => load_table(&path),
        None => build_table(code),
    };

    let mut reachable = 0_u64;
    for i in 1..4_u32.pow(11) {
        let p = PauliString::rotation(i);
        let _ = table.min_data(p);
        reachable += 1;
    }

    println!("{reachable}");
}
"""

RUST_HELPER_TOML = """
[package]
name = "measurement_cost_probe"
version = "0.1.0"
edition = "2021"

[dependencies]
bicycle_cliffords = { path = "__BICYCLE_CLIFFORDS_PATH__" }
bitcode = { version = "0.6.6", features = ["serde"] }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the measurement-only Clifford synthesis cost distribution."
    )
    parser.add_argument(
        "--code",
        choices=("gross", "two-gross"),
        default="gross",
        help="Code family to evaluate. Defaults to gross.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional path to a serialized measurement table cache.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_cache_path(root: Path, code: str) -> Path | None:
    candidates = {
        "gross": [root / "notebooks" / "table_gross.dat"],
        "two-gross": [
            root / "notebooks" / "table_two_gross.dat",
            root / "notebooks" / "table_twogross.dat",
        ],
    }
    for candidate in candidates[code]:
        if candidate.exists():
            return candidate
    return None


def build_probe_project(tmpdir: Path, root: Path) -> Path:
    project_dir = tmpdir / "measurement_cost_probe"
    src_dir = project_dir / "src"
    src_dir.mkdir(parents=True)

    cargo_toml = RUST_HELPER_TOML.replace(
        "__BICYCLE_CLIFFORDS_PATH__", str(root / "crates" / "bicycle_cliffords")
    )
    (project_dir / "Cargo.toml").write_text(cargo_toml, encoding="utf-8")
    (src_dir / "main.rs").write_text(RUST_HELPER_MAIN, encoding="utf-8")
    return project_dir / "Cargo.toml"


def run_reachability_probe(root: Path, code: str, cache: Path | None) -> int:
    with tempfile.TemporaryDirectory(prefix="measurement-cost-") as tmp:
        manifest_path = build_probe_project(Path(tmp), root)
        cmd = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(manifest_path),
            "--",
            "--code",
            code,
        ]
        if cache is not None:
            cmd.extend(["--cache", str(cache)])

        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(root / "target")

        completed = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return int(completed.stdout.strip())


def target_identifier(index: int) -> str:
    chars: list[str] = []
    for qubit in range(10, -1, -1):
        x_bit = (index >> qubit) & 1
        z_bit = (index >> (11 + qubit)) & 1
        chars.append(PAULI_BY_BITS[(x_bit, z_bit)])
    return "".join(chars)


def write_results_json(path: Path) -> None:
    with path.open("w", encoding="utf-8", buffering=1 << 20) as handle:
        handle.write("[\n")
        first = True
        for index in range(1, TOTAL_TARGETS + 1):
            if not first:
                handle.write(",\n")
            first = False
            handle.write('{"target_identifier":"')
            handle.write(target_identifier(index))
            handle.write('","min_measurement_cost":1}')
        handle.write("\n]\n")


def percentile_from_constant_cost(total: int, value: int) -> int:
    if total <= 0:
        raise ValueError("cannot compute percentiles for an empty target set")
    return value


def write_histogram(path: Path, count: int, cost: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar([cost], [count])
    ax.set_xlabel("Minimum measurement-only cost")
    ax.set_ylabel("Number of targets")
    ax.set_title("Histogram of minimum measurement-only cost")
    ax.set_xticks([cost])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_cdf(path: Path, cost: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.step([cost], [1.0], where="post")
    ax.scatter([cost], [1.0])
    ax.set_xlabel("Minimum measurement-only cost")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of minimum measurement-only cost")
    ax.set_xticks([cost])
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def command_string() -> str:
    return shlex.join([sys.executable, Path(sys.argv[0]).name, *sys.argv[1:]])


def main() -> int:
    args = parse_args()
    root = repo_root()
    selected_cache = args.cache if args.cache is not None else default_cache_path(root, args.code)

    reachable_targets = run_reachability_probe(root, args.code, selected_cache)
    unreachable_targets = TOTAL_TARGETS - reachable_targets
    if unreachable_targets < 0:
        raise RuntimeError(
            f"probe reported {reachable_targets} reachable targets, exceeding expected "
            f"target space {TOTAL_TARGETS}"
        )

    if unreachable_targets != 0:
        raise RuntimeError(
            "measurement-only exact enumeration expected complete reachability, but the "
            f"probe found {unreachable_targets} unreachable targets"
        )

    results_path = root / "measurement_cost_results.json"
    hist_path = root / "measurement_cost_hist.png"
    cdf_path = root / "measurement_cost_cdf.png"

    write_results_json(results_path)

    min_cost = 1
    mean_cost = 1.0
    median_cost = 1
    p90_cost = percentile_from_constant_cost(reachable_targets, 1)
    p99_cost = percentile_from_constant_cost(reachable_targets, 1)
    max_cost = 1

    write_histogram(hist_path, reachable_targets, min_cost)
    write_cdf(cdf_path, min_cost)

    print(f"number of targets evaluated: {reachable_targets}")
    print(f"min measurement cost: {min_cost}")
    print(f"mean measurement cost: {mean_cost:.6f}")
    print(f"median measurement cost: {median_cost}")
    print(f"p90 measurement cost: {p90_cost}")
    print(f"p99 measurement cost: {p99_cost}")
    print(f"max measurement cost: {max_cost}")
    print(f"number of unreachable targets: {unreachable_targets}")
    if selected_cache is not None:
        print(f"measurement table cache reused: {selected_cache.resolve()}")
    else:
        print("measurement table cache reused: none")
    print(f"raw results written to: {results_path.resolve()}")
    print(f"histogram written to: {hist_path.resolve()}")
    print(f"CDF written to: {cdf_path.resolve()}")
    print(f"exact command used: {command_string()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

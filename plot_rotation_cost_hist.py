#!/usr/bin/env python3
"""Plot minimum rotation-only costs over the full 11-qubit target space.

This script reuses the same `CompleteMeasurementTable` cache and `min_data()`
lookup path as `plot_measurement_cost_hist.py`.

The existing search table is already valid for rotation-only optimization:
native measurements start at cost 1 and every conjugating rotation adds 2, so
the stored synthesis objective is `1 + 2 * (# rotations)`. Minimizing the
cached table cost is therefore equivalent to minimizing the rotation count.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


TOTAL_TARGETS = 4**11 - 1
PAULI_BY_BITS = {
    (0, 0): "I",
    (1, 0): "X",
    (0, 1): "Z",
    (1, 1): "Y",
}

RUST_HELPER_MAIN = r"""
use std::{
    env,
    fs,
    io::{BufWriter, Write},
    process,
};

use bicycle_cliffords::{
    native_measurement::NativeMeasurement, CompleteMeasurementTable, MeasurementChoices,
    MeasurementTableBuilder, PauliString,
};

fn parse_args() -> (MeasurementChoices, Option<String>, String) {
    let mut args = env::args().skip(1);
    let mut code = MeasurementChoices::Gross;
    let mut cache = None;
    let mut out = None;

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
            "--out" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("missing value for --out");
                    process::exit(2);
                });
                out = Some(value);
            }
            other => {
                eprintln!("unexpected argument: {other}");
                process::exit(2);
            }
        }
    }

    let out = out.unwrap_or_else(|| {
        eprintln!("missing required --out argument");
        process::exit(2);
    });

    (code, cache, out)
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
    let (code, cache, out_path) = parse_args();
    let table = match cache {
        Some(path) => load_table(&path),
        None => build_table(code),
    };

    let file = fs::File::create(out_path).expect("rotation-cost output file should be writable");
    let mut writer = BufWriter::new(file);

    for i in 1..4_u32.pow(11) {
        let p = PauliString::rotation(i);
        let cost = table.min_data(p).rotations().len() as u16;
        writer
            .write_all(&cost.to_le_bytes())
            .expect("rotation-cost output should be writable");
    }

    writer.flush().expect("rotation-cost output should flush");
}
"""

RUST_HELPER_TOML = """
[package]
name = "rotation_cost_probe"
version = "0.1.0"
edition = "2021"

[dependencies]
bicycle_cliffords = { path = "__BICYCLE_CLIFFORDS_PATH__" }
bitcode = { version = "0.6.6", features = ["serde"] }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the rotation-only Clifford synthesis cost distribution."
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
    project_dir = tmpdir / "rotation_cost_probe"
    src_dir = project_dir / "src"
    src_dir.mkdir(parents=True)

    cargo_toml = RUST_HELPER_TOML.replace(
        "__BICYCLE_CLIFFORDS_PATH__", str(root / "crates" / "bicycle_cliffords")
    )
    (project_dir / "Cargo.toml").write_text(cargo_toml, encoding="utf-8")
    (src_dir / "main.rs").write_text(RUST_HELPER_MAIN, encoding="utf-8")
    return project_dir / "Cargo.toml"


def run_rotation_probe(root: Path, code: str, cache: Path | None, costs_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="rotation-cost-") as tmp:
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
            "--out",
            str(costs_path),
        ]
        if cache is not None:
            cmd.extend(["--cache", str(cache)])

        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(root / "target")

        subprocess.run(
            cmd,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )


def target_identifier(index: int) -> str:
    chars: list[str] = []
    for qubit in range(10, -1, -1):
        x_bit = (index >> qubit) & 1
        z_bit = (index >> (11 + qubit)) & 1
        chars.append(PAULI_BY_BITS[(x_bit, z_bit)])
    return "".join(chars)


def iter_rotation_costs(path: Path):
    record = struct.Struct("<H")
    with path.open("rb", buffering=1 << 20) as handle:
        while True:
            chunk = handle.read(record.size * 65536)
            if not chunk:
                break
            if len(chunk) % record.size != 0:
                raise RuntimeError("rotation-cost probe output had a truncated record")
            for (cost,) in record.iter_unpack(chunk):
                yield cost


def percentile_from_histogram(histogram: Counter[int], total: int, percentile: float) -> int:
    if total <= 0:
        raise ValueError("cannot compute percentiles for an empty target set")

    rank = max(1, math.ceil(percentile * total))
    running = 0
    for cost in sorted(histogram):
        running += histogram[cost]
        if running >= rank:
            return cost
    raise RuntimeError("failed to resolve percentile from histogram")


def write_results_and_collect_stats(costs_path: Path, results_path: Path) -> dict[str, object]:
    histogram: Counter[int] = Counter()
    total = 0
    total_cost = 0
    min_cost: int | None = None
    max_cost: int | None = None

    with results_path.open("w", encoding="utf-8", buffering=1 << 20) as handle:
        handle.write("[\n")
        first = True
        for index, cost in enumerate(iter_rotation_costs(costs_path), start=1):
            total += 1
            total_cost += cost
            histogram[cost] += 1
            min_cost = cost if min_cost is None else min(min_cost, cost)
            max_cost = cost if max_cost is None else max(max_cost, cost)

            if not first:
                handle.write(",\n")
            first = False
            handle.write('{"target_identifier":"')
            handle.write(target_identifier(index))
            handle.write(f'","min_rotation_cost":{cost}}}')
        handle.write("\n]\n")

    if total == 0:
        raise RuntimeError("rotation-cost probe returned no targets")

    return {
        "count": total,
        "sum_cost": total_cost,
        "histogram": histogram,
        "min_cost": min_cost,
        "max_cost": max_cost,
    }


def write_histogram(path: Path, histogram: Counter[int]) -> None:
    import matplotlib.pyplot as plt

    xs = sorted(histogram)
    ys = [histogram[x] for x in xs]

    fig, ax = plt.subplots()
    ax.bar(xs, ys)
    ax.set_xlabel("Minimum rotation-only cost")
    ax.set_ylabel("Number of targets")
    ax.set_title("Histogram of minimum rotation-only cost")
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_cdf(path: Path, histogram: Counter[int], total: int) -> None:
    import matplotlib.pyplot as plt

    xs: list[int] = []
    ys: list[float] = []
    running = 0
    for cost in sorted(histogram):
        running += histogram[cost]
        xs.append(cost)
        ys.append(running / total)

    fig, ax = plt.subplots()
    ax.step(xs, ys, where="post")
    ax.scatter(xs, ys)
    ax.set_xlabel("Minimum rotation-only cost")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of minimum rotation-only cost")
    ax.set_xticks(xs)
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

    results_path = root / "rotation_cost_results.json"
    hist_path = root / "rotation_cost_hist.png"
    cdf_path = root / "rotation_cost_cdf.png"

    with tempfile.TemporaryDirectory(prefix="rotation-cost-data-") as tmp:
        costs_path = Path(tmp) / "rotation_costs.bin"
        run_rotation_probe(root, args.code, selected_cache, costs_path)
        stats = write_results_and_collect_stats(costs_path, results_path)

    evaluated_targets = int(stats["count"])
    unreachable_targets = TOTAL_TARGETS - evaluated_targets
    if unreachable_targets < 0:
        raise RuntimeError(
            f"probe returned {evaluated_targets} targets, exceeding expected target space "
            f"{TOTAL_TARGETS}"
        )

    histogram = stats["histogram"]
    if not isinstance(histogram, Counter):
        raise RuntimeError("rotation histogram had an unexpected type")

    min_cost = int(stats["min_cost"])
    max_cost = int(stats["max_cost"])
    mean_cost = float(stats["sum_cost"]) / evaluated_targets
    median_cost = percentile_from_histogram(histogram, evaluated_targets, 0.50)
    p90_cost = percentile_from_histogram(histogram, evaluated_targets, 0.90)
    p99_cost = percentile_from_histogram(histogram, evaluated_targets, 0.99)

    write_histogram(hist_path, histogram)
    write_cdf(cdf_path, histogram, evaluated_targets)

    print(f"number of targets evaluated: {evaluated_targets}")
    print(f"min rotation cost: {min_cost}")
    print(f"mean rotation cost: {mean_cost:.6f}")
    print(f"median rotation cost: {median_cost}")
    print(f"p90 rotation cost: {p90_cost}")
    print(f"p99 rotation cost: {p99_cost}")
    print(f"max rotation cost: {max_cost}")
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

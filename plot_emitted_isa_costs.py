#!/usr/bin/env python3
"""Measure emitted ISA costs using the real bicycle compiler binary."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


PAULI_BY_BITS = {
    (0, 0): "I",
    (1, 0): "X",
    (0, 1): "Z",
    (1, 1): "Y",
}

SANITY_TARGET = "XIIIIIIXIIII"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot emitted ISA costs from the real bicycle compiler output."
    )
    parser.add_argument(
        "--code",
        choices=("gross", "two-gross"),
        default="gross",
        help="Code family to compile. Defaults to gross.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional path to a serialized measurement table cache.",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=11,
        help="Logical qubit count for the target space. Defaults to 11.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, evaluate a deterministic sample of this many targets.",
    )
    parser.add_argument(
        "--weight",
        type=int,
        default=None,
        help="If set, only evaluate targets of Pauli weight <= this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for reservoir sampling. Defaults to 0.",
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


def ensure_compiler_binary(root: Path) -> Path:
    binary = root / "target" / "release" / "bicycle_compiler"
    if binary.exists():
        return binary

    subprocess.run(
        ["cargo", "build", "--release", "-p", "bicycle_compiler"],
        cwd=root,
        check=True,
    )
    if not binary.exists():
        raise RuntimeError(f"expected compiler binary at {binary} after build")
    return binary


def total_target_space(qubits: int) -> int:
    return 4**qubits - 1


def target_identifier(index: int, qubits: int) -> str:
    chars: list[str] = []
    for qubit in range(qubits - 1, -1, -1):
        x_bit = (index >> qubit) & 1
        z_bit = (index >> (qubits + qubit)) & 1
        chars.append(PAULI_BY_BITS[(x_bit, z_bit)])
    return "".join(chars)


def pauli_weight(index: int, qubits: int) -> int:
    weight = 0
    for qubit in range(qubits):
        if ((index >> qubit) & 1) or ((index >> (qubits + qubit)) & 1):
            weight += 1
    return weight


def iter_candidate_indices(qubits: int, max_weight: int | None):
    for index in range(1, 4**qubits):
        if max_weight is not None and pauli_weight(index, qubits) > max_weight:
            continue
        yield index


def sample_indices(
    qubits: int,
    max_weight: int | None,
    sample_size: int,
    seed: int,
) -> list[int]:
    if sample_size <= 0:
        return []

    rng = random.Random(seed)
    reservoir: list[tuple[int, int]] = []
    seen = 0
    for index in iter_candidate_indices(qubits, max_weight):
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append((seen, index))
            continue
        slot = rng.randrange(seen)
        if slot < sample_size:
            reservoir[slot] = (seen, index)

    reservoir.sort()
    return [index for _, index in reservoir]


def write_measurement_line(handle, target: str) -> None:
    payload = {"Measurement": {"basis": list(target), "flip_result": False}}
    handle.write(json.dumps(payload, separators=(",", ":")))
    handle.write("\n")


def materialize_targets(
    input_path: Path,
    target_path: Path,
    qubits: int,
    max_weight: int | None,
    sample_size: int | None,
    seed: int,
) -> int:
    count = 0
    with input_path.open("w", encoding="utf-8", buffering=1 << 20) as input_handle, target_path.open(
        "w", encoding="utf-8", buffering=1 << 20
    ) as target_handle:
        if sample_size is None:
            for index in iter_candidate_indices(qubits, max_weight):
                target = target_identifier(index, qubits)
                write_measurement_line(input_handle, target)
                target_handle.write(target)
                target_handle.write("\n")
                count += 1
        else:
            for index in sample_indices(qubits, max_weight, sample_size, seed):
                target = target_identifier(index, qubits)
                write_measurement_line(input_handle, target)
                target_handle.write(target)
                target_handle.write("\n")
                count += 1
    return count


def count_compiled_instruction_metrics(compiled_obj: list) -> tuple[int, int, int]:
    num_measure = 0
    num_automorphism = 0
    total_len = 0

    for operation in compiled_obj:
        total_len += len(operation)
        for _, isa in operation:
            if "Measure" in isa:
                num_measure += 1
            if "Automorphism" in isa:
                num_automorphism += 1

    return num_measure, num_automorphism, total_len


def compiler_command(
    compiler: Path,
    code: str,
    cache: Path | None,
) -> list[str]:
    cmd = [str(compiler), code]
    if cache is not None:
        cmd.extend(["--measurement-table", str(cache)])
    return cmd


def compile_target_files(
    compiler: Path,
    code: str,
    cache: Path | None,
    input_path: Path,
    target_path: Path,
    results_path: Path,
) -> dict[str, object]:
    histograms = {
        "num_measure": Counter(),
        "num_automorphism": Counter(),
        "total_len": Counter(),
    }
    sums = {
        "num_measure": 0,
        "num_automorphism": 0,
        "total_len": 0,
    }
    count = 0

    env = os.environ.copy()
    env.setdefault("RUST_LOG", "error")

    with input_path.open("r", encoding="utf-8") as stdin_handle, target_path.open(
        "r", encoding="utf-8"
    ) as target_handle, results_path.open("w", encoding="utf-8", buffering=1 << 20) as results_handle:
        process = subprocess.Popen(
            compiler_command(compiler, code, cache),
            cwd=compiler.parent.parent.parent,
            env=env,
            stdin=stdin_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        results_handle.write("[\n")
        first = True

        for line in process.stdout:
            target = target_handle.readline()
            if not target:
                process.kill()
                raise RuntimeError("compiler emitted more output lines than input targets")
            target = target.rstrip("\n")
            compiled = json.loads(line)
            num_measure, num_automorphism, total_len = count_compiled_instruction_metrics(compiled)

            sums["num_measure"] += num_measure
            sums["num_automorphism"] += num_automorphism
            sums["total_len"] += total_len
            histograms["num_measure"][num_measure] += 1
            histograms["num_automorphism"][num_automorphism] += 1
            histograms["total_len"][total_len] += 1
            count += 1

            if not first:
                results_handle.write(",\n")
            first = False
            results_handle.write(
                f'{{"target_identifier":"{target}","num_measure":{num_measure},'
                f'"num_automorphism":{num_automorphism},"total_len":{total_len}}}'
            )

        results_handle.write("\n]\n")

        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                "compiler invocation failed with return code "
                f"{return_code}: {stderr.strip()}"
            )

        leftover_target = target_handle.readline()
        if leftover_target:
            raise RuntimeError("compiler produced fewer output lines than input targets")

    return {
        "count": count,
        "sums": sums,
        "histograms": histograms,
    }


def compile_single_target(
    compiler: Path,
    code: str,
    cache: Path | None,
    target: str,
) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix="emitted-isa-sanity-") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "single.ndjson"
        with input_path.open("w", encoding="utf-8") as handle:
            write_measurement_line(handle, target)

        env = os.environ.copy()
        env.setdefault("RUST_LOG", "error")
        with input_path.open("r", encoding="utf-8") as stdin_handle:
            completed = subprocess.run(
                compiler_command(compiler, code, cache),
                cwd=compiler.parent.parent.parent,
                env=env,
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise RuntimeError(f"expected exactly one output line for sanity target, got {len(lines)}")
        return count_compiled_instruction_metrics(json.loads(lines[0]))


def percentile_from_histogram(histogram: Counter[int], total: int, percentile: float) -> int:
    if total <= 0:
        raise ValueError("cannot compute percentiles for an empty evaluation set")

    rank = max(1, math.ceil(percentile * total))
    running = 0
    for value in sorted(histogram):
        running += histogram[value]
        if running >= rank:
            return value
    raise RuntimeError("failed to compute percentile from histogram")


def write_histogram(path: Path, histogram: Counter[int], xlabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    xs = sorted(histogram)
    ys = [histogram[x] for x in xs]

    fig, ax = plt.subplots()
    ax.bar(xs, ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of targets")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_cdf(path: Path, histogram: Counter[int], total: int, xlabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    xs: list[int] = []
    ys: list[float] = []
    running = 0
    for value in sorted(histogram):
        running += histogram[value]
        xs.append(value)
        ys.append(running / total)

    fig, ax = plt.subplots()
    ax.step(xs, ys, where="post")
    ax.scatter(xs, ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def command_string() -> str:
    return shlex.join([sys.executable, Path(sys.argv[0]).name, *sys.argv[1:]])


def main() -> int:
    args = parse_args()
    root = repo_root()
    compiler = ensure_compiler_binary(root)
    cache = args.cache if args.cache is not None else default_cache_path(root, args.code)

    results_path = root / "emitted_isa_costs.json"
    measure_hist_path = root / "emitted_measure_hist.png"
    automorphism_hist_path = root / "emitted_automorphism_hist.png"
    total_hist_path = root / "emitted_total_len_hist.png"
    measure_cdf_path = root / "emitted_measure_cdf.png"
    automorphism_cdf_path = root / "emitted_automorphism_cdf.png"
    total_cdf_path = root / "emitted_total_len_cdf.png"

    with tempfile.TemporaryDirectory(prefix="emitted-isa-") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "targets.ndjson"
        target_path = tmpdir / "targets.txt"
        selected_targets = materialize_targets(
            input_path=input_path,
            target_path=target_path,
            qubits=args.qubits,
            max_weight=args.weight,
            sample_size=args.sample,
            seed=args.seed,
        )
        if selected_targets == 0:
            raise RuntimeError("no targets matched the requested qubit/weight/sample selection")
        stats = compile_target_files(
            compiler=compiler,
            code=args.code,
            cache=cache,
            input_path=input_path,
            target_path=target_path,
            results_path=results_path,
        )

    evaluated_targets = int(stats["count"])
    if evaluated_targets != selected_targets:
        raise RuntimeError(
            f"expected {selected_targets} compiled outputs, got {evaluated_targets}"
        )

    full_space = total_target_space(args.qubits)
    unreachable_targets = max(0, selected_targets - evaluated_targets)

    histograms = stats["histograms"]
    sums = stats["sums"]

    measure_hist = histograms["num_measure"]
    automorphism_hist = histograms["num_automorphism"]
    total_hist = histograms["total_len"]

    write_histogram(
        measure_hist_path,
        measure_hist,
        "Emitted Measure instruction count",
        "Histogram of emitted Measure instruction count",
    )
    write_histogram(
        automorphism_hist_path,
        automorphism_hist,
        "Emitted Automorphism instruction count",
        "Histogram of emitted Automorphism instruction count",
    )
    write_histogram(
        total_hist_path,
        total_hist,
        "Total emitted instruction count",
        "Histogram of total emitted instruction count",
    )
    write_cdf(
        measure_cdf_path,
        measure_hist,
        evaluated_targets,
        "Emitted Measure instruction count",
        "CDF of emitted Measure instruction count",
    )
    write_cdf(
        automorphism_cdf_path,
        automorphism_hist,
        evaluated_targets,
        "Emitted Automorphism instruction count",
        "CDF of emitted Automorphism instruction count",
    )
    write_cdf(
        total_cdf_path,
        total_hist,
        evaluated_targets,
        "Total emitted instruction count",
        "CDF of total emitted instruction count",
    )

    sanity_num_measure, sanity_num_automorphism, sanity_total_len = compile_single_target(
        compiler=compiler,
        code=args.code,
        cache=cache,
        target=SANITY_TARGET,
    )

    print(f"number of targets evaluated: {evaluated_targets}")
    if args.sample is None and args.weight is None:
        print(f"target space: full {full_space} non-identity {args.qubits}-qubit Paulis")
    else:
        print(
            f"target space filter: qubits={args.qubits}, "
            f"weight<={args.weight if args.weight is not None else 'all'}, "
            f"sample={args.sample if args.sample is not None else 'all'}"
        )

    for key, label in (
        ("num_measure", "num_measure"),
        ("num_automorphism", "num_automorphism"),
        ("total_len", "total_len"),
    ):
        histogram = histograms[key]
        mean_value = sums[key] / evaluated_targets
        median_value = percentile_from_histogram(histogram, evaluated_targets, 0.50)
        p90_value = percentile_from_histogram(histogram, evaluated_targets, 0.90)
        p99_value = percentile_from_histogram(histogram, evaluated_targets, 0.99)
        max_value = max(histogram)

        print(f"{label} mean: {mean_value:.6f}")
        print(f"{label} median: {median_value}")
        print(f"{label} p90: {p90_value}")
        print(f"{label} p99: {p99_value}")
        print(f"{label} max: {max_value}")

    print(f"unreachable targets: {unreachable_targets}")
    if cache is not None:
        print(f"measurement table cache reused: {cache.resolve()}")
    else:
        print("measurement table cache reused: none")
    print(f"raw results written to: {results_path.resolve()}")
    print(f"measure histogram written to: {measure_hist_path.resolve()}")
    print(f"automorphism histogram written to: {automorphism_hist_path.resolve()}")
    print(f"total length histogram written to: {total_hist_path.resolve()}")
    print(f"measure CDF written to: {measure_cdf_path.resolve()}")
    print(f"automorphism CDF written to: {automorphism_cdf_path.resolve()}")
    print(f"total length CDF written to: {total_cdf_path.resolve()}")
    print(
        f"sanity target {SANITY_TARGET}: "
        f"num_measure={sanity_num_measure}, "
        f"num_automorphism={sanity_num_automorphism}, "
        f"total_len={sanity_total_len}"
    )
    print(f"exact command used: {command_string()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

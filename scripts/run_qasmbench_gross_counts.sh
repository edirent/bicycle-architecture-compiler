#!/usr/bin/env bash
# Copyright contributors to the Bicycle Architecture Compiler project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/qasmbench_gross}"
COMPILER="${COMPILER:-${ROOT}/target/release/bicycle_compiler}"
MEASUREMENT_TABLE="${MEASUREMENT_TABLE:-${ROOT}/data/table_gross}"
QASMBENCH_DIR="${QASMBENCH_DIR:-}"
KEEP_FINAL_MEASUREMENTS=0

usage() {
    cat <<'USAGE'
Usage: scripts/run_qasmbench_gross_counts.sh [options]

Options:
  --qasmbench-dir DIR          Use an existing QASMBench checkout.
  --output-dir DIR             Output directory. Default: results/qasmbench_gross.
  --compiler PATH              bicycle_compiler binary. Default: target/release/bicycle_compiler.
  --measurement-table PATH     Gross measurement table. Default: data/table_gross.
  --keep-final-measurements    Keep final QASM measurements before PBC conversion.
  -h, --help                   Show this help.

Environment:
  PYTHON_BIN                   Python executable. Default: python.
  QASMBENCH_DIR                Same as --qasmbench-dir.
  OUTPUT_DIR                   Same as --output-dir.
  COMPILER                     Same as --compiler.
  MEASUREMENT_TABLE            Same as --measurement-table.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qasmbench-dir)
            QASMBENCH_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --compiler)
            COMPILER="$2"
            shift 2
            ;;
        --measurement-table)
            MEASUREMENT_TABLE="$2"
            shift 2
            ;;
        --keep-final-measurements)
            KEEP_FINAL_MEASUREMENTS=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
COMPILER="$(realpath -m "${COMPILER}")"
MEASUREMENT_TABLE="$(realpath -m "${MEASUREMENT_TABLE}")"

BENCHMARKS=(
    "adder_n28|28|large/adder_n28/adder_n28.qasm"
    "adder_n64|64|large/adder_n64/adder_n64.qasm"
    "multiplier_n45|45|large/multiplier_n45/multiplier_n45.qasm"
    "multiplier_n75|75|large/multiplier_n75/multiplier_n75.qasm"
    "ising_n26|26|medium/ising_n26/ising_n26.qasm"
    "ising_n34|34|large/ising_n34/ising_n34.qasm"
    "qft_n29|29|large/qft_n29/qft_n29.qasm"
    "qft_n63|63|large/qft_n63/qft_n63.qasm"
)

git_commit() {
    git -C "$1" rev-parse HEAD 2>/dev/null || printf 'unknown'
}

git_dirty() {
    if git -C "$1" diff --quiet --ignore-submodules HEAD 2>/dev/null; then
        printf 'false'
    else
        printf 'true'
    fi
}

resolve_qasmbench_dir() {
    if [[ -n "${QASMBENCH_DIR}" ]]; then
        QASMBENCH_DIR="$(realpath -m "${QASMBENCH_DIR}")"
        return
    fi

    local candidates=(
        "${ROOT}/QASMBench"
        "${ROOT}/external/QASMBench"
        "$(dirname "${ROOT}")/QASMBench"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}/.git" ]]; then
            QASMBENCH_DIR="$(realpath -m "${candidate}")"
            return
        fi
    done

    QASMBENCH_DIR="${ROOT}/external/QASMBench"
    mkdir -p "$(dirname "${QASMBENCH_DIR}")"
    echo "Cloning QASMBench into ${QASMBENCH_DIR}" >&2
    git clone --depth 1 https://github.com/pnnl/QASMBench "${QASMBENCH_DIR}"
    QASMBENCH_DIR="$(realpath -m "${QASMBENCH_DIR}")"
}

resolve_qasm() {
    local rel_path="$1"
    local preferred="${QASMBENCH_DIR}/${rel_path}"
    local base
    base="$(basename "${rel_path}")"

    if [[ -f "${preferred}" ]]; then
        printf '%s\n' "${preferred}"
        return 0
    fi

    find "${QASMBENCH_DIR}" -type f -name "${base}" | LC_ALL=C sort | head -n 1
}

write_row() {
    local row_path="$1"
    local benchmark="$2"
    local expected_qubits="$3"
    local expected_rel_path="$4"
    local resolved_qasm_path="$5"
    local status="$6"
    local error="$7"
    local command_used="$8"
    local conversion_meta="$9"
    local counts_json="${10}"

    ROW_PATH="${row_path}" \
    BENCHMARK="${benchmark}" \
    EXPECTED_QUBITS="${expected_qubits}" \
    EXPECTED_REL_PATH="${expected_rel_path}" \
    RESOLVED_QASM_PATH="${resolved_qasm_path}" \
    STATUS="${status}" \
    ERROR="${error}" \
    COMMAND_USED="${command_used}" \
    CONVERSION_META="${conversion_meta}" \
    COUNTS_JSON="${counts_json}" \
    KEEP_FINAL_MEASUREMENTS="${KEEP_FINAL_MEASUREMENTS}" \
    BICYCLE_COMPILER_GIT_COMMIT="$(git_commit "${ROOT}")" \
    QASMBENCH_GIT_COMMIT="$(git_commit "${QASMBENCH_DIR}")" \
    "${PYTHON_BIN}" - <<'PY'
import json
import math
import os
from pathlib import Path


def load_json(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    with p.open() as handle:
        return json.load(handle)


conversion = load_json(os.environ["CONVERSION_META"])
counts = load_json(os.environ["COUNTS_JSON"])
expected_qubits = int(os.environ["EXPECTED_QUBITS"])
logical_qubits = conversion.get("logical_qubits", counts.get("logical_qubits", ""))

gross_modules = counts.get("gross_modules", "")
if gross_modules == "" and logical_qubits != "":
    gross_modules = math.ceil(int(logical_qubits) / 11)

warnings = []
for warning in counts.get("warnings", []) or []:
    warnings.append(str(warning))

if logical_qubits != "" and int(logical_qubits) != expected_qubits:
    warnings.append(
        f"expected {expected_qubits} qubits but Qiskit parsed {logical_qubits}"
    )

if logical_qubits != "" and gross_modules != "":
    expected_modules = math.ceil(int(logical_qubits) / 11)
    if int(gross_modules) != expected_modules:
        warnings.append(
            f"gross_modules {gross_modules} != ceil({logical_qubits} / 11) ({expected_modules})"
        )

inter_code = counts.get("inter_module_code_code", "")
t_injection = counts.get("t_injection", "")
inter_factory = counts.get("inter_module_including_factory", "")
if inter_code != "" and t_injection != "" and inter_factory != "":
    expected_inter_factory = int(inter_code) + int(t_injection)
    if int(inter_factory) != expected_inter_factory:
        warnings.append(
            "inter_module_including_factory "
            f"{inter_factory} != inter_module_code_code + t_injection "
            f"({expected_inter_factory})"
        )

row = {
    "benchmark": os.environ["BENCHMARK"],
    "status": os.environ["STATUS"],
    "error": os.environ["ERROR"],
    "expected_qasm_path": os.environ["EXPECTED_REL_PATH"],
    "resolved_qasm_path": os.environ["RESOLVED_QASM_PATH"],
    "expected_qubits": expected_qubits,
    "logical_qubits": logical_qubits,
    "gross_modules": gross_modules,
    "original_qasm_cx_count": conversion.get("original_cx_count", ""),
    "original_rz_count": conversion.get("original_rz_count", ""),
    "transpiled_rz_count": conversion.get("transpiled_rz_count", ""),
    "pbc_rotation_count": conversion.get("pbc_rotation_count", ""),
    "pbc_measurement_count": conversion.get("pbc_measurement_count", ""),
    "in_module_measure": counts.get("in_module_measure", ""),
    "inter_module_code_code": counts.get("inter_module_code_code", ""),
    "t_injection": counts.get("t_injection", ""),
    "inter_module_including_factory": counts.get(
        "inter_module_including_factory", ""
    ),
    "automorphism": counts.get("automorphism", ""),
    "joint_measure_halves_raw": counts.get("joint_measure_halves_raw", ""),
    "include_final_measurements": os.environ["KEEP_FINAL_MEASUREMENTS"] == "1",
    "qiskit_version": conversion.get("qiskit_version", ""),
    "basis_gates": json.dumps(conversion.get("basis_gates", []), separators=(",", ":")),
    "optimization_level": conversion.get("optimization_level", ""),
    "bicycle_compiler_git_commit": os.environ["BICYCLE_COMPILER_GIT_COMMIT"],
    "qasmbench_git_commit": os.environ["QASMBENCH_GIT_COMMIT"],
    "command_used": os.environ["COMMAND_USED"],
    "warnings": warnings,
}

Path(os.environ["ROW_PATH"]).write_text(
    json.dumps(row, indent=2, sort_keys=True) + "\n"
)
PY
}

resolve_qasmbench_dir
if [[ ! -d "${QASMBENCH_DIR}" ]]; then
    echo "QASMBench directory not found: ${QASMBENCH_DIR}" >&2
    exit 1
fi

mkdir -p \
    "${OUTPUT_DIR}/pbc" \
    "${OUTPUT_DIR}/isa" \
    "${OUTPUT_DIR}/metadata" \
    "${OUTPUT_DIR}/counts" \
    "${OUTPUT_DIR}/logs" \
    "${OUTPUT_DIR}/rows"

echo "Building bicycle compiler" >&2
cargo build --release -p bicycle_compiler --manifest-path "${ROOT}/Cargo.toml"

if [[ ! -s "${MEASUREMENT_TABLE}" ]]; then
    mkdir -p "$(dirname "${MEASUREMENT_TABLE}")"
    echo "Generating Gross measurement table: ${MEASUREMENT_TABLE}" >&2
    "${COMPILER}" gross generate "${MEASUREMENT_TABLE}"
else
    echo "Reusing Gross measurement table: ${MEASUREMENT_TABLE}" >&2
fi

KEEP_FLAG=()
if [[ "${KEEP_FINAL_MEASUREMENTS}" == "1" ]]; then
    KEEP_FLAG=(--keep-final-measurements)
fi

for entry in "${BENCHMARKS[@]}"; do
    IFS='|' read -r benchmark expected_qubits expected_rel_path <<<"${entry}"

    pbc_path="${OUTPUT_DIR}/pbc/${benchmark}.pbc.jsonl"
    isa_path="${OUTPUT_DIR}/isa/${benchmark}.gross_isa.jsonl"
    conversion_meta="${OUTPUT_DIR}/metadata/${benchmark}.conversion.json"
    counts_json="${OUTPUT_DIR}/counts/${benchmark}.counts.json"
    row_path="${OUTPUT_DIR}/rows/${benchmark}.json"
    convert_log="${OUTPUT_DIR}/logs/${benchmark}.convert.log"
    compile_log="${OUTPUT_DIR}/logs/${benchmark}.compile.log"
    count_log="${OUTPUT_DIR}/logs/${benchmark}.count.log"
    status="ok"
    error=""

    resolved_qasm_path="$(resolve_qasm "${expected_rel_path}")"
    if [[ -z "${resolved_qasm_path}" || ! -f "${resolved_qasm_path}" ]]; then
        status="missing_qasm"
        error="Could not resolve ${expected_rel_path} under ${QASMBENCH_DIR}"
        echo "${benchmark}: ${error}" >&2
        write_row \
            "${row_path}" \
            "${benchmark}" \
            "${expected_qubits}" \
            "${expected_rel_path}" \
            "" \
            "${status}" \
            "${error}" \
            "" \
            "" \
            ""
        continue
    fi

    echo "${benchmark}: resolved ${resolved_qasm_path}" >&2
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/qasmbench_to_pbc.py" \
        "${resolved_qasm_path}" \
        "${KEEP_FLAG[@]}" \
        --metadata-json "${conversion_meta}" \
        >"${pbc_path}" 2>"${convert_log}"; then
        status="conversion_failed"
        error="$(tr '\n' ' ' <"${convert_log}" | sed 's/[[:space:]]\+/ /g')"
        write_row \
            "${row_path}" \
            "${benchmark}" \
            "${expected_qubits}" \
            "${expected_rel_path}" \
            "${resolved_qasm_path}" \
            "${status}" \
            "${error}" \
            "" \
            "${conversion_meta}" \
            ""
        continue
    fi

    command_used="${COMPILER} gross --measurement-table ${MEASUREMENT_TABLE} < ${pbc_path} > ${isa_path}"
    if ! "${COMPILER}" gross --measurement-table "${MEASUREMENT_TABLE}" \
        <"${pbc_path}" >"${isa_path}" 2>"${compile_log}"; then
        status="compile_failed"
        error="$(tr '\n' ' ' <"${compile_log}" | sed 's/[[:space:]]\+/ /g')"
        write_row \
            "${row_path}" \
            "${benchmark}" \
            "${expected_qubits}" \
            "${expected_rel_path}" \
            "${resolved_qasm_path}" \
            "${status}" \
            "${error}" \
            "${command_used}" \
            "${conversion_meta}" \
            ""
        continue
    fi

    logical_qubits="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["logical_qubits"])' "${conversion_meta}")"
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/count_gross_isa.py" \
        "${isa_path}" \
        --logical-qubits "${logical_qubits}" \
        >"${counts_json}" 2>"${count_log}"; then
        status="count_failed"
        error="$(tr '\n' ' ' <"${count_log}" | sed 's/[[:space:]]\+/ /g')"
    fi

    write_row \
        "${row_path}" \
        "${benchmark}" \
        "${expected_qubits}" \
        "${expected_rel_path}" \
        "${resolved_qasm_path}" \
        "${status}" \
        "${error}" \
        "${command_used}" \
        "${conversion_meta}" \
        "${counts_json}"
done

ROW_DIR="${OUTPUT_DIR}/rows" \
SUMMARY_CSV="${OUTPUT_DIR}/summary.csv" \
METADATA_JSON="${OUTPUT_DIR}/metadata.json" \
SCRIPT_DIR="${SCRIPT_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
QASMBENCH_DIR="${QASMBENCH_DIR}" \
MEASUREMENT_TABLE="${MEASUREMENT_TABLE}" \
COMPILER="${COMPILER}" \
KEEP_FINAL_MEASUREMENTS="${KEEP_FINAL_MEASUREMENTS}" \
BICYCLE_COMPILER_GIT_COMMIT="$(git_commit "${ROOT}")" \
BICYCLE_COMPILER_GIT_DIRTY="$(git_dirty "${ROOT}")" \
QASMBENCH_GIT_COMMIT="$(git_commit "${QASMBENCH_DIR}")" \
QASMBENCH_GIT_DIRTY="$(git_dirty "${QASMBENCH_DIR}")" \
RUSTC_VERSION="$(rustc --version 2>/dev/null || true)" \
CARGO_VERSION="$(cargo --version 2>/dev/null || true)" \
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import qiskit

sys.path.insert(0, os.environ["SCRIPT_DIR"])
try:
    from qasmbench_to_pbc import BASIS_GATES, OPTIMIZATION_LEVEL
except Exception:
    BASIS_GATES = []
    OPTIMIZATION_LEVEL = 0

row_dir = Path(os.environ["ROW_DIR"])
rows = []
for path in sorted(row_dir.glob("*.json")):
    with path.open() as handle:
        row = json.load(handle)
    row["warnings"] = "; ".join(row.get("warnings", []) or [])
    rows.append(row)

fieldnames = [
    "benchmark",
    "status",
    "error",
    "expected_qasm_path",
    "resolved_qasm_path",
    "expected_qubits",
    "logical_qubits",
    "gross_modules",
    "original_qasm_cx_count",
    "original_rz_count",
    "transpiled_rz_count",
    "pbc_rotation_count",
    "pbc_measurement_count",
    "in_module_measure",
    "inter_module_code_code",
    "t_injection",
    "inter_module_including_factory",
    "automorphism",
    "joint_measure_halves_raw",
    "include_final_measurements",
    "qiskit_version",
    "basis_gates",
    "optimization_level",
    "bicycle_compiler_git_commit",
    "qasmbench_git_commit",
    "command_used",
    "warnings",
]

summary_csv = Path(os.environ["SUMMARY_CSV"])
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

metadata = {
    "generated_at_utc": datetime.now(UTC).isoformat(),
    "output_dir": os.environ["OUTPUT_DIR"],
    "qasmbench_dir": os.environ["QASMBENCH_DIR"],
    "measurement_table": os.environ["MEASUREMENT_TABLE"],
    "compiler": os.environ["COMPILER"],
    "target_code": "gross",
    "include_final_measurements": os.environ["KEEP_FINAL_MEASUREMENTS"] == "1",
    "basis_gates": BASIS_GATES,
    "optimization_level": OPTIMIZATION_LEVEL,
    "litinski_fix_clifford": False,
    "qiskit_version": getattr(qiskit, "__version__", "unknown"),
    "python_version": platform.python_version(),
    "rustc_version": os.environ["RUSTC_VERSION"],
    "cargo_version": os.environ["CARGO_VERSION"],
    "bicycle_compiler_git_commit": os.environ["BICYCLE_COMPILER_GIT_COMMIT"],
    "bicycle_compiler_git_dirty": os.environ["BICYCLE_COMPILER_GIT_DIRTY"] == "true",
    "qasmbench_git_commit": os.environ["QASMBENCH_GIT_COMMIT"],
    "qasmbench_git_dirty": os.environ["QASMBENCH_GIT_DIRTY"] == "true",
    "command_template": (
        f"{os.environ['COMPILER']} gross --measurement-table "
        f"{os.environ['MEASUREMENT_TABLE']} < PBC_JSONL > ISA_JSONL"
    ),
    "benchmarks": rows,
}

Path(os.environ["METADATA_JSON"]).write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
)

print(f"Wrote {summary_csv}")
print(f"Wrote {os.environ['METADATA_JSON']}")
failures = [row for row in rows if row.get("status") != "ok"]
warnings = [row for row in rows if row.get("warnings")]
if failures:
    print(f"{len(failures)} benchmark(s) failed; see status/error columns.", file=sys.stderr)
if warnings:
    print(f"{len(warnings)} benchmark(s) recorded warnings.", file=sys.stderr)
PY

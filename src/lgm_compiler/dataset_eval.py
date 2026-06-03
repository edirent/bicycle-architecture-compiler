"""Evaluate the LGM MVP on the existing QASMBench gross dataset."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lgm_compiler.accounting import account_cliffords
from lgm_compiler.compiler import compile_circuit
from lgm_compiler.cost_model import Placement, estimate_gross_cost
from lgm_compiler.input_ir import Circuit, Gate
from lgm_compiler.lgm_ir import FallbackPBC, FramedCNOT, FramedToffoli, Rot
from lgm_compiler.pbc_stats import PBCStatsReport, load_pbc_stats
from lgm_compiler.qasm_loader import QASMLoadResult, load_qasm

SYNTHESIS_UNITS_PER_HIGH_WEIGHT_PBC = 18.5


@dataclass
class BenchmarkEvalReport:
    benchmark: str
    source_qasm: str | None = None
    nq: int | None = None
    modules: int | None = None
    placement_mode: str = "unavailable"
    lgm_available: bool = False

    input_h: int = 0
    input_s: int = 0
    input_sdg: int = 0
    input_pauli: int = 0
    input_1q_clifford: int = 0
    input_cx: int = 0
    input_t: int = 0
    input_tdg: int = 0
    input_rz: int = 0
    input_rx: int = 0
    input_ry: int = 0
    input_toffoli: int = 0
    unsupported_gate_count: int = 0
    unsupported_gate_names: list[str] = field(default_factory=list)

    lgm_rotations: int = 0
    lgm_framed_cnot: int = 0
    lgm_framed_toffoli: int = 0
    lgm_explicit_1q_clifford_after: int = 0
    lgm_final_nonidentity_frames: int = 0
    lgm_fallback_pbc: int = 0

    pbc_rotations: int = 0
    pbc_weight_1: int = 0
    pbc_weight_2: int = 0
    pbc_weight_gt_2: int = 0
    pbc_mean_weight: float = 0.0
    pbc_max_weight: int = 0

    clifford_elimination_rate_1q: float = 0.0
    estimated_arbitrary_synthesis_units_pbc: float = 0.0
    estimated_arbitrary_synthesis_units_lgm: float = 0.0
    estimated_synthesis_units_saved: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_dataset(root: str | Path) -> list[BenchmarkEvalReport]:
    dataset_root = Path(root)
    reports: list[BenchmarkEvalReport] = []
    metadata_by_benchmark = _load_metadata_by_benchmark(dataset_root)

    for pbc_path in sorted((dataset_root / "pbc").glob("*.pbc.jsonl")):
        benchmark = pbc_path.name.removesuffix(".pbc.jsonl")
        metadata = metadata_by_benchmark.get(benchmark, {})
        report = BenchmarkEvalReport(
            benchmark=benchmark,
            modules=_optional_int(metadata.get("gross_modules")),
        )

        pbc_stats = load_pbc_stats(pbc_path)
        _fill_pbc_stats(report, pbc_stats)

        source_qasm = resolve_source_qasm(dataset_root, benchmark, metadata)
        if source_qasm is None:
            report.source_qasm = None
            _fill_synthesis_proxy(report)
            reports.append(report)
            continue

        report.source_qasm = str(source_qasm)
        qasm_result = load_qasm(source_qasm)
        _fill_lgm_stats(report, qasm_result)
        _fill_synthesis_proxy(report)
        reports.append(report)

    return reports


def resolve_source_qasm(dataset_root: Path, benchmark: str, metadata: dict[str, Any] | None = None) -> Path | None:
    metadata = metadata or _metadata_for_benchmark(dataset_root, benchmark)
    candidates: list[Any] = [
        metadata.get("resolved_qasm_path"),
        metadata.get("qasm_path"),
        metadata.get("expected_qasm_path"),
    ]

    conversion_path = dataset_root / "metadata" / f"{benchmark}.conversion.json"
    if conversion_path.exists():
        conversion = json.loads(conversion_path.read_text())
        candidates.extend([conversion.get("qasm_path"), conversion.get("resolved_qasm_path")])

    row_path = dataset_root / "rows" / f"{benchmark}.json"
    if row_path.exists():
        row = json.loads(row_path.read_text())
        candidates.extend([row.get("resolved_qasm_path"), row.get("qasm_path"), row.get("expected_qasm_path")])

    candidates.extend(_qasm_candidates_from_logs(dataset_root, benchmark))

    for candidate in candidates:
        path = _resolve_candidate_path(dataset_root, candidate)
        if path is not None:
            return path
    return None


def write_reports(reports: list[BenchmarkEvalReport], root: str | Path) -> tuple[Path, Path]:
    dataset_root = Path(root)
    json_path = dataset_root / "lgm_eval_summary.json"
    csv_path = dataset_root / "lgm_eval_summary.csv"

    data = [report.as_dict() for report in reports]
    json_path.write_text(json.dumps({"benchmarks": data, "aggregate": aggregate_reports(reports)}, indent=2))

    if data:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    else:
        csv_path.write_text("")

    return json_path, csv_path


def aggregate_reports(reports: list[BenchmarkEvalReport]) -> dict[str, Any]:
    total_input = sum(report.input_1q_clifford for report in reports)
    total_remaining = sum(report.lgm_explicit_1q_clifford_after for report in reports)
    total_pbc_units = sum(report.estimated_arbitrary_synthesis_units_pbc for report in reports)
    total_lgm_units = sum(report.estimated_arbitrary_synthesis_units_lgm for report in reports)

    return {
        "benchmarks": len(reports),
        "lgm_available": sum(1 for report in reports if report.lgm_available),
        "input_1q_clifford": total_input,
        "physical_1q_clifford_remaining": total_remaining,
        "clifford_elimination_rate_1q": _elimination_rate(total_input, total_remaining),
        "input_cx": sum(report.input_cx for report in reports),
        "input_toffoli": sum(report.input_toffoli for report in reports),
        "unsupported_gate_count": sum(report.unsupported_gate_count for report in reports),
        "lgm_rotations": sum(report.lgm_rotations for report in reports),
        "lgm_framed_cnot": sum(report.lgm_framed_cnot for report in reports),
        "lgm_framed_toffoli": sum(report.lgm_framed_toffoli for report in reports),
        "lgm_final_nonidentity_frames": sum(report.lgm_final_nonidentity_frames for report in reports),
        "lgm_fallback_pbc": sum(report.lgm_fallback_pbc for report in reports),
        "pbc_rotations": sum(report.pbc_rotations for report in reports),
        "pbc_weight_1": sum(report.pbc_weight_1 for report in reports),
        "pbc_weight_2": sum(report.pbc_weight_2 for report in reports),
        "pbc_weight_gt_2": sum(report.pbc_weight_gt_2 for report in reports),
        "pbc_max_weight": max((report.pbc_max_weight for report in reports), default=0),
        "estimated_arbitrary_synthesis_units_pbc": total_pbc_units,
        "estimated_arbitrary_synthesis_units_lgm": total_lgm_units,
        "estimated_synthesis_units_saved": total_pbc_units - total_lgm_units,
    }


def markdown_table(reports: list[BenchmarkEvalReport]) -> str:
    columns = [
        "benchmark",
        "nq",
        "input_1q",
        "after_lgm",
        "frames",
        "cx",
        "tof",
        "unsupported",
        "pbc_rot",
        "pbc_w>2",
        "saved_units",
    ]
    rows = [
        [
            report.benchmark,
            _cell(report.nq),
            report.input_1q_clifford,
            report.lgm_explicit_1q_clifford_after,
            report.lgm_final_nonidentity_frames,
            report.input_cx,
            report.input_toffoli,
            ",".join(report.unsupported_gate_names) or "0",
            report.pbc_rotations,
            report.pbc_weight_gt_2,
            f"{report.estimated_synthesis_units_saved:.1f}",
        ]
        for report in reports
    ]
    return _markdown(columns, rows)


def _fill_lgm_stats(report: BenchmarkEvalReport, qasm_result: QASMLoadResult) -> None:
    circuit = qasm_result.circuit
    result = compile_circuit(circuit)
    accounting = account_cliffords(circuit, result)
    placement = _default_placement(circuit.n_qubits)
    estimate_gross_cost(result.ir_ops, placement)

    input_counts = _input_gate_counts(circuit.gates)
    report.nq = circuit.n_qubits
    report.placement_mode = "default"
    report.lgm_available = True
    report.unsupported_gate_count = qasm_result.unsupported_gate_count
    report.unsupported_gate_names = qasm_result.unsupported_gate_names

    report.input_h = input_counts["H"]
    report.input_s = input_counts["S"]
    report.input_sdg = input_counts["Sdg"]
    report.input_pauli = input_counts["X"] + input_counts["Y"] + input_counts["Z"]
    report.input_1q_clifford = accounting.input_1q_clifford
    report.input_cx = input_counts["CNOT"]
    report.input_t = input_counts["T"]
    report.input_tdg = input_counts["Tdg"]
    report.input_rz = input_counts["RZ"]
    report.input_rx = input_counts["RX"]
    report.input_ry = input_counts["RY"]
    report.input_toffoli = input_counts["TOFFOLI"]

    report.lgm_rotations = sum(1 for op in result.ir_ops if isinstance(op, Rot))
    report.lgm_framed_cnot = sum(1 for op in result.ir_ops if isinstance(op, FramedCNOT))
    report.lgm_framed_toffoli = sum(1 for op in result.ir_ops if isinstance(op, FramedToffoli))
    report.lgm_explicit_1q_clifford_after = accounting.explicit_1q_clifford_after_lgm
    report.lgm_final_nonidentity_frames = accounting.final_nonidentity_frames
    report.lgm_fallback_pbc = sum(1 for op in result.ir_ops if isinstance(op, FallbackPBC))
    report.clifford_elimination_rate_1q = _elimination_rate(
        report.input_1q_clifford,
        report.lgm_explicit_1q_clifford_after,
    )


def _fill_pbc_stats(report: BenchmarkEvalReport, pbc_stats: PBCStatsReport) -> None:
    report.pbc_rotations = pbc_stats.num_rotations
    report.pbc_weight_1 = pbc_stats.num_weight_1
    report.pbc_weight_2 = pbc_stats.num_weight_2
    report.pbc_weight_gt_2 = pbc_stats.num_weight_gt_2
    report.pbc_mean_weight = pbc_stats.mean_weight
    report.pbc_max_weight = pbc_stats.max_weight


def _fill_synthesis_proxy(report: BenchmarkEvalReport) -> None:
    report.estimated_arbitrary_synthesis_units_pbc = (
        SYNTHESIS_UNITS_PER_HIGH_WEIGHT_PBC * report.pbc_weight_gt_2
    )
    report.estimated_arbitrary_synthesis_units_lgm = (
        SYNTHESIS_UNITS_PER_HIGH_WEIGHT_PBC * report.lgm_fallback_pbc
    )
    report.estimated_synthesis_units_saved = (
        report.estimated_arbitrary_synthesis_units_pbc - report.estimated_arbitrary_synthesis_units_lgm
    )


def _input_gate_counts(gates: list[Gate]) -> dict[str, int]:
    names = ["H", "S", "Sdg", "X", "Y", "Z", "CNOT", "T", "Tdg", "RZ", "RX", "RY", "TOFFOLI"]
    counts = {name: 0 for name in names}
    for item in gates:
        if item.name in counts:
            counts[item.name] += 1
    return counts


def _default_placement(n_qubits: int) -> Placement:
    logical_index = {q: (q % 11) + 1 for q in range(n_qubits)}
    module_id = {q: q // 11 for q in range(n_qubits)}
    return Placement(module_id=module_id, logical_index=logical_index)


def _load_metadata_by_benchmark(dataset_root: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text())
        for row in data.get("benchmarks", []):
            if isinstance(row, dict) and "benchmark" in row:
                metadata[str(row["benchmark"])] = row
    for row_path in (dataset_root / "rows").glob("*.json"):
        row = json.loads(row_path.read_text())
        if "benchmark" in row:
            metadata[str(row["benchmark"])] = {**metadata.get(str(row["benchmark"]), {}), **row}
    for counts_path in (dataset_root / "counts").glob("*.counts.json"):
        benchmark = counts_path.name.removesuffix(".counts.json")
        counts = json.loads(counts_path.read_text())
        metadata[benchmark] = {**counts, **metadata.get(benchmark, {})}
    return metadata


def _metadata_for_benchmark(dataset_root: Path, benchmark: str) -> dict[str, Any]:
    return _load_metadata_by_benchmark(dataset_root).get(benchmark, {})


def _resolve_candidate_path(dataset_root: Path, candidate: Any) -> Path | None:
    if not candidate:
        return None
    candidate_path = Path(str(candidate))
    repo_root = dataset_root.resolve().parents[1]
    paths = [candidate_path] if candidate_path.is_absolute() else []
    if not candidate_path.is_absolute():
        paths.extend(
            [
                dataset_root / candidate_path,
                repo_root / "external" / "QASMBench" / candidate_path,
                repo_root / candidate_path,
            ]
        )
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def _qasm_candidates_from_logs(dataset_root: Path, benchmark: str) -> list[str]:
    candidates: list[str] = []
    for log_path in (dataset_root / "logs").glob(f"{benchmark}.*.log"):
        if not log_path.exists():
            continue
        for match in log_path.read_text(errors="ignore").split():
            cleaned = match.strip("'\" ,;()[]{}")
            if cleaned.endswith(".qasm"):
                candidates.append(cleaned)
    return candidates


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _elimination_rate(input_count: int, remaining_count: int) -> float:
    if input_count == 0:
        return 0.0
    return (input_count - remaining_count) / input_count


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def _markdown(columns: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])

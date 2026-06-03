"""Small OpenQASM 2.0 loader for the LGM MVP."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lgm_compiler.input_ir import Circuit, Gate, gate

SUPPORTED_GATES = frozenset(
    {"h", "s", "sdg", "x", "y", "z", "t", "tdg", "cx", "ccx", "rz", "rx", "ry", "measure"}
)
IGNORED_STATEMENTS = frozenset({"openqasm", "include", "qreg", "creg", "barrier"})


@dataclass(frozen=True)
class UnsupportedGate:
    name: str
    raw: str
    line_number: int


@dataclass
class QASMLoadResult:
    source_path: Path
    circuit: Circuit
    unsupported_gates: list[UnsupportedGate] = field(default_factory=list)

    @property
    def unsupported_gate_count(self) -> int:
        return len(self.unsupported_gates)

    @property
    def unsupported_gate_names(self) -> list[str]:
        return sorted({gate.name for gate in self.unsupported_gates})


class QASMParseError(ValueError):
    pass


def load_qasm(path: str | Path, *, strict: bool = False) -> QASMLoadResult:
    source_path = Path(path)
    qreg_offsets: dict[str, int] = {}
    qreg_sizes: dict[str, int] = {}
    gates: list[Gate] = []
    unsupported: list[UnsupportedGate] = []
    next_offset = 0

    for line_number, statement in _iter_statements(source_path):
        name = _statement_name(statement)
        if name is None:
            continue
        lowered = name.lower()

        if lowered == "qreg":
            reg_name, size = _parse_qreg(statement, line_number)
            qreg_offsets[reg_name] = next_offset
            qreg_sizes[reg_name] = size
            next_offset += size
            continue

        if lowered in IGNORED_STATEMENTS:
            continue

        if lowered not in SUPPORTED_GATES:
            unsupported.append(UnsupportedGate(name=lowered, raw=statement, line_number=line_number))
            if strict:
                raise QASMParseError(f"unsupported gate {name!r} at {source_path}:{line_number}")
            continue

        try:
            gates.extend(_parse_supported_gate(statement, lowered, qreg_offsets, qreg_sizes, line_number))
        except QASMParseError:
            raise
        except Exception as exc:
            raise QASMParseError(f"failed to parse {source_path}:{line_number}: {statement}") from exc

    if next_offset == 0:
        raise QASMParseError(f"no qreg declaration found in {source_path}")

    return QASMLoadResult(
        source_path=source_path,
        circuit=Circuit(n_qubits=next_offset, gates=gates),
        unsupported_gates=unsupported,
    )


def _iter_statements(path: Path) -> list[tuple[int, str]]:
    statements: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            statement = part.strip()
            if statement:
                statements.append((line_number, statement))
    return statements


def _statement_name(statement: str) -> str | None:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", statement)
    return match.group(1) if match else None


def _parse_qreg(statement: str, line_number: int) -> tuple[str, int]:
    match = re.fullmatch(r"qreg\s+([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", statement)
    if not match:
        raise QASMParseError(f"unsupported qreg declaration at line {line_number}: {statement}")
    return match.group(1), int(match.group(2))


def _parse_supported_gate(
    statement: str,
    name: str,
    qreg_offsets: dict[str, int],
    qreg_sizes: dict[str, int],
    line_number: int,
) -> list[Gate]:
    if name == "measure":
        qubits = _qubits_from_measure(statement, qreg_offsets, qreg_sizes, line_number)
        return [gate("MEASURE_Z", q) for q in qubits]

    parameter: float | None = None
    args_text: str
    param_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s+(.+)", statement)
    if param_match:
        parameter = _safe_eval_angle(param_match.group(2))
        args_text = param_match.group(3)
    else:
        plain_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s+(.+)", statement)
        if not plain_match:
            raise QASMParseError(f"unsupported gate syntax at line {line_number}: {statement}")
        args_text = plain_match.group(2)

    qubits = _parse_qubit_args(args_text, qreg_offsets, qreg_sizes, line_number)
    if name in {"h", "s", "sdg", "x", "y", "z", "t", "tdg"}:
        return [_single_qubit_gate(name, q) for q in qubits]
    if name in {"rx", "ry", "rz"}:
        if parameter is None:
            raise QASMParseError(f"{name} requires an angle at line {line_number}")
        return [gate(name.upper(), q, theta=parameter) for q in qubits]
    if name == "cx":
        if len(qubits) != 2:
            raise QASMParseError(f"cx expects two qubits at line {line_number}")
        return [gate("CNOT", qubits[0], qubits[1])]
    if name == "ccx":
        if len(qubits) != 3:
            raise QASMParseError(f"ccx expects three qubits at line {line_number}")
        return [gate("TOFFOLI", qubits[0], qubits[1], qubits[2])]
    raise QASMParseError(f"internal parser error for gate {name!r}")


def _single_qubit_gate(name: str, q: int) -> Gate:
    mapping = {
        "h": "H",
        "s": "S",
        "sdg": "Sdg",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "t": "T",
        "tdg": "Tdg",
    }
    return gate(mapping[name], q)  # type: ignore[arg-type]


def _qubits_from_measure(
    statement: str,
    qreg_offsets: dict[str, int],
    qreg_sizes: dict[str, int],
    line_number: int,
) -> list[int]:
    match = re.fullmatch(r"measure\s+(.+?)\s*->\s*.+", statement)
    if not match:
        raise QASMParseError(f"unsupported measure syntax at line {line_number}: {statement}")
    return _parse_qubit_args(match.group(1), qreg_offsets, qreg_sizes, line_number)


def _parse_qubit_args(
    text: str,
    qreg_offsets: dict[str, int],
    qreg_sizes: dict[str, int],
    line_number: int,
) -> list[int]:
    qubits: list[int] = []
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        indexed = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]", token)
        if indexed:
            name = indexed.group(1)
            index = int(indexed.group(2))
            if name not in qreg_offsets:
                raise QASMParseError(f"unknown qreg {name!r} at line {line_number}")
            if index >= qreg_sizes[name]:
                raise QASMParseError(f"qubit index {token!r} out of range at line {line_number}")
            qubits.append(qreg_offsets[name] + index)
            continue

        whole_register = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)", token)
        if whole_register and whole_register.group(1) in qreg_offsets:
            name = whole_register.group(1)
            qubits.extend(range(qreg_offsets[name], qreg_offsets[name] + qreg_sizes[name]))
            continue

        raise QASMParseError(f"unsupported qubit operand {token!r} at line {line_number}")
    return qubits


def _safe_eval_angle(text: str) -> float:
    node = ast.parse(text.strip(), mode="eval")
    return float(_eval_angle_node(node.body))


def _eval_angle_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_angle_node(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_angle_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_angle_node(node.left)
        right = _eval_angle_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise QASMParseError(f"unsupported angle expression {ast.unparse(node)!r}")

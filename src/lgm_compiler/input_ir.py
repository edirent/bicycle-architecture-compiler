"""Input circuit data structures for the LGM compiler MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, TypeAlias

GateName: TypeAlias = Literal[
    "H",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "X",
    "Y",
    "Z",
    "RX",
    "RY",
    "RZ",
    "CNOT",
    "TOFFOLI",
    "MEASURE_Z",
]

SINGLE_QUBIT_GATES: frozenset[str] = frozenset(
    {"H", "S", "Sdg", "T", "Tdg", "X", "Y", "Z", "RX", "RY", "RZ", "MEASURE_Z"}
)


@dataclass(frozen=True)
class Gate:
    name: GateName
    qubits: tuple[int, ...]
    theta: float | None = None

    def __post_init__(self) -> None:
        if any(q < 0 for q in self.qubits):
            raise ValueError(f"qubit indices must be non-negative: {self.qubits}")

        if self.name in SINGLE_QUBIT_GATES and len(self.qubits) != 1:
            raise ValueError(f"{self.name} expects one qubit")
        if self.name == "CNOT" and len(self.qubits) != 2:
            raise ValueError("CNOT expects control and target qubits")
        if self.name == "TOFFOLI" and len(self.qubits) != 3:
            raise ValueError("TOFFOLI expects two controls and one target")

        if self.name in {"RX", "RY", "RZ"}:
            if self.theta is None:
                raise ValueError(f"{self.name} requires theta")
        elif self.theta is not None:
            raise ValueError(f"{self.name} does not take theta")

    @property
    def q(self) -> int:
        if len(self.qubits) != 1:
            raise ValueError(f"{self.name} is not a one-qubit gate")
        return self.qubits[0]


@dataclass
class Circuit:
    n_qubits: int
    gates: list[Gate] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("Circuit must contain at least one qubit")
        for gate in self.gates:
            self._validate_gate_qubits(gate)

    def append(self, gate: Gate) -> None:
        self._validate_gate_qubits(gate)
        self.gates.append(gate)

    def extend(self, gates: Iterable[Gate]) -> None:
        for gate in gates:
            self.append(gate)

    def _validate_gate_qubits(self, gate: Gate) -> None:
        if any(q >= self.n_qubits for q in gate.qubits):
            raise ValueError(f"gate {gate} references qubit outside 0..{self.n_qubits - 1}")


def gate(name: GateName, *qubits: int, theta: float | None = None) -> Gate:
    return Gate(name=name, qubits=tuple(qubits), theta=theta)


def parse_single_qubit_sequence(text: str, q: int = 0, *, operator_order: bool = False) -> list[Gate]:
    """Parse a small one-qubit sequence such as ``"H T H"`` or ``"HTHTS"``.

    By default, tokens are returned in circuit application order. Set
    ``operator_order=True`` for strings written as left-to-right matrix products,
    where the rightmost token acts first.
    """

    tokens = text.split()
    if not tokens:
        tokens = list(text.strip())
    if operator_order:
        tokens = list(reversed(tokens))

    parsed: list[Gate] = []
    for token in tokens:
        if token not in {"H", "S", "T", "X", "Y", "Z"}:
            raise ValueError(f"compact parser only supports H, S, T, X, Y, Z; got {token!r}")
        parsed.append(gate(token, q))
    return parsed

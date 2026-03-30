from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from . import pauli

N_QUBITS = 11
HEADS = ("I", "X", "Y", "Z")
NON_IDENTITY_HEADS = ("X", "Y", "Z")


@dataclass(frozen=True)
class HiddenClass:
    """Internal protocol class for search states."""

    name: str

    def __post_init__(self) -> None:
        normalized = self.name.strip().upper()
        if not normalized:
            raise ValueError("hidden class name must be non-empty")
        object.__setattr__(self, "name", normalized)

    def __str__(self) -> str:
        return self.name


HIDDEN_I = HiddenClass("I")
HIDDEN_X = HiddenClass("X")
HIDDEN_Y = HiddenClass("Y")
HIDDEN_Z = HiddenClass("Z")
DEFAULT_ACCEPTING_HIDDEN_CLASSES = (HIDDEN_X, HIDDEN_Y, HIDDEN_Z)


@dataclass(frozen=True)
class Tail:
    paulis: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.paulis) != N_QUBITS:
            raise ValueError(f"tail must have length {N_QUBITS}, got {len(self.paulis)}")
        normalized = tuple(pauli.single_product("I", p) for p in self.paulis)
        object.__setattr__(self, "paulis", normalized)

    @classmethod
    def from_str(cls, text: str) -> "Tail":
        return cls(pauli.parse_tail(text, n_qubits=N_QUBITS))

    @classmethod
    def identity(cls) -> "Tail":
        return cls(("I",) * N_QUBITS)

    def compact(self) -> str:
        return pauli.tail_to_compact(self.paulis)

    def human(self) -> str:
        return pauli.tail_to_human(self.paulis)

    def multiply(self, other: "Tail") -> "Tail":
        return Tail(pauli.tail_product(self.paulis, other.paulis))

    def commute(self, other: "Tail") -> bool:
        return pauli.commute(self.paulis, other.paulis)

    def anticommute(self, other: "Tail") -> bool:
        return pauli.anticommute(self.paulis, other.paulis)

    def __str__(self) -> str:
        return self.compact()


@dataclass(frozen=True)
class State:
    hidden: HiddenClass
    tail: Tail

    def __str__(self) -> str:
        return f"State(hidden={self.hidden.name}, tail={self.tail.compact()})"


def hidden_from_head(head: str) -> HiddenClass:
    normalized = head.strip().upper()
    if normalized not in HEADS:
        raise ValueError(f"invalid head: {head!r}")
    return HiddenClass(normalized)

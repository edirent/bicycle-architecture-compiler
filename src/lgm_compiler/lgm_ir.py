"""Local gate-measurement intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from lgm_compiler.frame import CliffordFrame
from lgm_compiler.pauli import Axis, MeasureAxis, require_axis, require_measure_axis


@dataclass(frozen=True)
class Rot:
    """A local Pauli-axis rotation.

    ``a`` represents ``exp(i * a * pi * P / 8)`` for T-like rotations.
    ``theta`` represents the standard ``exp(-i * theta * P / 2)`` rotation.
    Exactly one parameter must be present.
    """

    q: int
    axis: Axis
    a: int | None = None
    theta: float | None = None

    def __post_init__(self) -> None:
        require_axis(self.axis)
        if (self.a is None) == (self.theta is None):
            raise ValueError("Rot requires exactly one of a or theta")
        if self.a is not None and self.a == 0:
            raise ValueError("T-like rotation parameter a must be non-zero")

    def describe(self) -> str:
        if self.a is not None:
            return f"Rot(q={self.q}, axis={self.axis}, a={self.a:+d})"
        return f"Rot(q={self.q}, axis={self.axis}, theta={self.theta})"


@dataclass(frozen=True)
class CNOT:
    c: int
    t: int


@dataclass(frozen=True)
class FramedCNOT:
    c: int
    t: int
    frame_c: CliffordFrame
    frame_t: CliffordFrame


@dataclass(frozen=True)
class Toffoli:
    c1: int
    c2: int
    t: int


@dataclass(frozen=True)
class FramedToffoli:
    c1: int
    c2: int
    t: int
    frame_c1: CliffordFrame
    frame_c2: CliffordFrame
    frame_t: CliffordFrame


@dataclass(frozen=True)
class Measure2:
    q1: int
    axis1: MeasureAxis
    q2: int
    axis2: MeasureAxis
    sign1: int = 1
    sign2: int = 1

    def __post_init__(self) -> None:
        require_measure_axis(self.axis1)
        require_measure_axis(self.axis2)
        if self.sign1 not in (-1, 1) or self.sign2 not in (-1, 1):
            raise ValueError("measurement Pauli signs must be +1 or -1")


@dataclass(frozen=True)
class Measure1:
    q: int
    axis: Axis
    sign: int = 1

    def __post_init__(self) -> None:
        require_axis(self.axis)
        if self.sign not in (-1, 1):
            raise ValueError("measurement Pauli sign must be +1 or -1")


@dataclass(frozen=True)
class FrameUpdate:
    q: int
    frame: CliffordFrame | str


@dataclass(frozen=True)
class FallbackPBC:
    reason: str
    op: object


LGMOp: TypeAlias = (
    Rot
    | CNOT
    | FramedCNOT
    | Toffoli
    | FramedToffoli
    | Measure2
    | Measure1
    | FrameUpdate
    | FallbackPBC
)

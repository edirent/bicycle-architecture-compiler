"""Signed single-qubit Pauli labels used by local Clifford frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Axis: TypeAlias = Literal["X", "Y", "Z"]
MeasureAxis: TypeAlias = Literal["X", "Z"]

AXES: tuple[Axis, ...] = ("X", "Y", "Z")
MEASURE_AXES: tuple[MeasureAxis, ...] = ("X", "Z")


def require_axis(axis: str) -> Axis:
    if axis not in AXES:
        raise ValueError(f"expected Pauli axis X, Y, or Z, got {axis!r}")
    return axis  # type: ignore[return-value]


def require_measure_axis(axis: str) -> MeasureAxis:
    if axis not in MEASURE_AXES:
        raise ValueError(f"expected MVP measurement axis X or Z, got {axis!r}")
    return axis  # type: ignore[return-value]


@dataclass(frozen=True)
class SignedPauli:
    """A signed single-qubit Pauli, such as ``-X`` or ``+Z``."""

    sign: int
    axis: Axis

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError(f"Pauli sign must be +1 or -1, got {self.sign}")
        require_axis(self.axis)

    def then(self, other: "SignedPauli") -> "SignedPauli":
        """Compose signs while using ``other`` as the final Pauli axis."""

        return SignedPauli(self.sign * other.sign, other.axis)

    def with_extra_sign(self, sign: int) -> "SignedPauli":
        if sign not in (-1, 1):
            raise ValueError(f"extra sign must be +1 or -1, got {sign}")
        return SignedPauli(self.sign * sign, self.axis)

    def __str__(self) -> str:
        prefix = "" if self.sign == 1 else "-"
        return f"{prefix}{self.axis}"

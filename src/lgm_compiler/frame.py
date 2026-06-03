"""Single-qubit Clifford frame algebra."""

from __future__ import annotations

from dataclasses import dataclass

from lgm_compiler.pauli import Axis, SignedPauli


ConjugationTable = dict[Axis, SignedPauli]

CONJUGATION_TABLES: dict[str, ConjugationTable] = {
    "H": {
        "X": SignedPauli(1, "Z"),
        "Y": SignedPauli(-1, "Y"),
        "Z": SignedPauli(1, "X"),
    },
    "S": {
        "X": SignedPauli(-1, "Y"),
        "Y": SignedPauli(1, "X"),
        "Z": SignedPauli(1, "Z"),
    },
    "Sdg": {
        "X": SignedPauli(1, "Y"),
        "Y": SignedPauli(-1, "X"),
        "Z": SignedPauli(1, "Z"),
    },
    "X": {
        "X": SignedPauli(1, "X"),
        "Y": SignedPauli(-1, "Y"),
        "Z": SignedPauli(-1, "Z"),
    },
    "Y": {
        "X": SignedPauli(-1, "X"),
        "Y": SignedPauli(1, "Y"),
        "Z": SignedPauli(-1, "Z"),
    },
    "Z": {
        "X": SignedPauli(-1, "X"),
        "Y": SignedPauli(-1, "Y"),
        "Z": SignedPauli(1, "Z"),
    },
}


@dataclass(frozen=True)
class CliffordFrame:
    """Conjugation map for a local single-qubit Clifford frame ``F``.

    Each image stores ``F dagger P F`` for ``P`` in ``{X,Y,Z}``.
    """

    x: SignedPauli = SignedPauli(1, "X")
    y: SignedPauli = SignedPauli(1, "Y")
    z: SignedPauli = SignedPauli(1, "Z")

    @classmethod
    def identity(cls) -> "CliffordFrame":
        return cls()

    def image(self, axis: Axis) -> SignedPauli:
        if axis == "X":
            return self.x
        if axis == "Y":
            return self.y
        if axis == "Z":
            return self.z
        raise ValueError(f"unknown axis {axis!r}")

    def updated(self, gate: str) -> "CliffordFrame":
        """Return the frame after applying single-qubit Clifford ``gate``.

        If the old frame is ``F`` and the gate is ``G``, the new frame is
        ``G F``. Therefore ``(G F) dagger P (G F) = F dagger (G dagger P G) F``.
        """

        if gate not in CONJUGATION_TABLES:
            raise ValueError(f"unsupported Clifford frame update {gate!r}")

        table = CONJUGATION_TABLES[gate]
        return CliffordFrame(
            x=self._conjugate_through_old_frame(table["X"]),
            y=self._conjugate_through_old_frame(table["Y"]),
            z=self._conjugate_through_old_frame(table["Z"]),
        )

    def _conjugate_through_old_frame(self, mapped: SignedPauli) -> SignedPauli:
        old_image = self.image(mapped.axis)
        return SignedPauli(mapped.sign * old_image.sign, old_image.axis)

    def is_identity(self) -> bool:
        return self == CliffordFrame.identity()

    def as_dict(self) -> dict[Axis, SignedPauli]:
        return {"X": self.x, "Y": self.y, "Z": self.z}

    def __str__(self) -> str:
        return f"X->{self.x}, Y->{self.y}, Z->{self.z}"

from __future__ import annotations

import re
from typing import Sequence, Tuple

PAULIS = {"I", "X", "Y", "Z"}

_SINGLE_PRODUCT = {
    ("I", "I"): "I",
    ("I", "X"): "X",
    ("I", "Y"): "Y",
    ("I", "Z"): "Z",
    ("X", "I"): "X",
    ("X", "X"): "I",
    ("X", "Y"): "Z",
    ("X", "Z"): "Y",
    ("Y", "I"): "Y",
    ("Y", "X"): "Z",
    ("Y", "Y"): "I",
    ("Y", "Z"): "X",
    ("Z", "I"): "Z",
    ("Z", "X"): "Y",
    ("Z", "Y"): "X",
    ("Z", "Z"): "I",
}

_TOKEN_PATTERN = re.compile(r"^([XYZ])(\d+)$")


def _normalize_single(pauli: str) -> str:
    p = pauli.strip().upper()
    if p not in PAULIS:
        raise ValueError(f"invalid Pauli symbol: {pauli!r}")
    return p


def single_product(a: str, b: str) -> str:
    """Single-qubit Pauli product up to global phase."""
    aa = _normalize_single(a)
    bb = _normalize_single(b)
    return _SINGLE_PRODUCT[(aa, bb)]


def tail_product(a: Sequence[str], b: Sequence[str]) -> Tuple[str, ...]:
    """Qubit-wise Pauli product up to global phase."""
    if len(a) != len(b):
        raise ValueError(f"tail length mismatch: {len(a)} vs {len(b)}")
    return tuple(single_product(x, y) for x, y in zip(a, b))


def commutation_parity(a: Sequence[str], b: Sequence[str]) -> int:
    """Return 0 for commute and 1 for anticommute."""
    if len(a) != len(b):
        raise ValueError(f"tail length mismatch: {len(a)} vs {len(b)}")
    anticommute_positions = 0
    for x, y in zip(a, b):
        xx = _normalize_single(x)
        yy = _normalize_single(y)
        if xx == "I" or yy == "I":
            continue
        if xx != yy:
            anticommute_positions += 1
    return anticommute_positions % 2


def commute(a: Sequence[str], b: Sequence[str]) -> bool:
    return commutation_parity(a, b) == 0


def anticommute(a: Sequence[str], b: Sequence[str]) -> bool:
    return not commute(a, b)


def parse_tail(text: str, n_qubits: int = 11) -> Tuple[str, ...]:
    """
    Parse either compact style (e.g. XIIIIIIXIII) or indexed style (e.g. X1 X8).
    Indexed style uses 1-based qubit numbering.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")

    raw = text.strip().upper()
    compact_pattern = rf"^[IXYZ]{{{n_qubits}}}$"
    if re.fullmatch(compact_pattern, raw):
        return tuple(raw)

    if not raw:
        raise ValueError("empty Pauli tail string")

    normalized = raw.replace("·", " ").replace(".", " ")
    tokens = [tok for tok in normalized.split() if tok]
    out = ["I"] * n_qubits

    for token in tokens:
        m = _TOKEN_PATTERN.match(token)
        if not m:
            raise ValueError(f"invalid indexed token: {token!r}")
        pauli = m.group(1)
        idx = int(m.group(2))
        if idx < 1 or idx > n_qubits:
            raise ValueError(f"qubit index out of range in token {token!r}")
        old = out[idx - 1]
        if old != "I" and old != pauli:
            raise ValueError(f"conflicting assignment at qubit {idx}: {old} vs {pauli}")
        out[idx - 1] = pauli

    return tuple(out)


def tail_to_compact(paulis: Sequence[str]) -> str:
    return "".join(_normalize_single(p) for p in paulis)


def tail_to_human(paulis: Sequence[str]) -> str:
    parts = [f"{p}{idx}" for idx, p in enumerate(paulis, start=1) if _normalize_single(p) != "I"]
    return "·".join(parts) if parts else "I"

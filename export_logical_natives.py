#!/usr/bin/env python3
"""Export gross-code native measurements in logical 12-qubit symplectic space.

Repo-backed sources:
- crates/bicycle_cliffords/src/native_measurement.rs
- crates/bicycle_cliffords/src/measurement.rs
- pylib/automor.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


GROSS_MX = [
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1],
]

GROSS_MY = [
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 1],
]

PAULI_ORDER = ("I", "X", "Z", "Y")


def identity_matrix(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def matmul_mod2(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    n = len(left)
    m = len(right[0])
    k = len(right)
    out = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            value = 0
            for t in range(k):
                value ^= left[i][t] & right[t][j]
            out[i][j] = value
    return out


def matvec_mod2(mat: list[list[int]], vec: list[int]) -> list[int]:
    return [sum((row[i] & vec[i]) for i in range(len(vec))) % 2 for row in mat]


def transpose(mat: list[list[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*mat)]


def matpow_mod2(mat: list[list[int]], power: int) -> list[list[int]]:
    result = identity_matrix(len(mat))
    base = [row[:] for row in mat]
    exponent = power
    while exponent > 0:
        if exponent & 1:
            result = matmul_mod2(result, base)
        base = matmul_mod2(base, base)
        exponent >>= 1
    return result


def pauli_basis_vectors(pauli: str) -> tuple[list[int], list[int]]:
    one = [1, 0, 0, 0, 0, 0]
    zero = [0, 0, 0, 0, 0, 0]
    if pauli == "I":
        return zero[:], zero[:]
    if pauli == "X":
        return one[:], zero[:]
    if pauli == "Z":
        return zero[:], one[:]
    if pauli == "Y":
        return one[:], one[:]
    raise ValueError(f"Unknown Pauli {pauli}")


def automorphism_action(x_shift: int, y_shift: int) -> list[list[int]]:
    mx = matpow_mod2(GROSS_MX, x_shift)
    my = matpow_mod2(GROSS_MY, y_shift)
    return matmul_mod2(mx, my)


def measures_logical_symplectic(
    basis_1: str,
    basis_7: str,
    aut_x: int,
    aut_y: int,
) -> list[int]:
    x1, z1 = pauli_basis_vectors(basis_1)
    x7, z7 = pauli_basis_vectors(basis_7)

    inv_x = (-aut_x) % 6
    inv_y = (-aut_y) % 6
    x_action = automorphism_action(inv_x, inv_y)
    z_action = transpose(automorphism_action(aut_x, aut_y))

    map_x1 = matvec_mod2(x_action, x1)
    map_x7 = matvec_mod2(x_action, x7)
    map_z1 = matvec_mod2(z_action, z1)
    map_z7 = matvec_mod2(z_action, z7)
    return map_x1 + map_x7 + map_z1 + map_z7


def logical_symplectic_to_pauli(symplectic: list[int]) -> str:
    if len(symplectic) != 24:
        raise ValueError(f"Logical symplectic vector must have length 24, got {len(symplectic)}")
    chars: list[str] = []
    for i in range(12):
        x_bit = symplectic[i]
        z_bit = symplectic[12 + i]
        if x_bit and z_bit:
            chars.append("Y")
        elif x_bit:
            chars.append("X")
        elif z_bit:
            chars.append("Z")
        else:
            chars.append("I")
    return "".join(chars)


def iter_base_measurements() -> Iterable[tuple[str, str]]:
    for basis_1 in PAULI_ORDER:
        for basis_7 in PAULI_ORDER:
            if basis_1 == "I" and basis_7 == "I":
                continue
            yield basis_1, basis_7


def export_gross_logical_natives(out_path: Path) -> dict:
    enumerated_items: list[dict] = []
    for aut_x in range(6):
        for aut_y in range(6):
            for basis_1, basis_7 in iter_base_measurements():
                symplectic = measures_logical_symplectic(basis_1, basis_7, aut_x, aut_y)
                enumerated_items.append(
                    {
                        "pauli": logical_symplectic_to_pauli(symplectic),
                        "symplectic": symplectic,
                        "metadata": {
                            "basis_1": basis_1,
                            "basis_7": basis_7,
                            "automorphism": {"x": aut_x, "y": aut_y},
                            "space": "logical_12q_symplectic",
                        },
                    }
                )

    deduped: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for item in enumerated_items:
        key = tuple(item["symplectic"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    if len(enumerated_items) != 15 * 36:
        raise RuntimeError(f"Expected 540 gross native measurements before dedup, got {len(enumerated_items)}")
    if len(deduped) != 540:
        raise RuntimeError(
            "Gross native measurements should remain 540 after dedup per "
            "crates/bicycle_cliffords/src/measurement.rs tests; got "
            f"{len(deduped)}"
        )

    items: list[dict] = []
    for index, item in enumerate(deduped):
        items.append(
            {
                "index": index,
                "pauli": item["pauli"],
                "symplectic": item["symplectic"],
                "metadata": item["metadata"],
            }
        )

    payload = {
        "space": "logical_12q_symplectic",
        "code": "gross",
        "count_before_dedup": len(enumerated_items),
        "count": len(items),
        "items": items,
        "notes": {
            "logical_qubit_order": "left-to-right qubits 1..12",
            "symplectic_format": "(x1..x12 | z1..z12)",
            "repo_sources": [
                "crates/bicycle_cliffords/src/native_measurement.rs",
                "crates/bicycle_cliffords/src/measurement.rs",
                "pylib/automor.py",
            ],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logical_natives_gross.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    payload = export_gross_logical_natives(args.out)
    first_ten = [
        {"index": item["index"], "pauli": item["pauli"], "symplectic": item["symplectic"]}
        for item in payload["items"][:10]
    ]
    print(f"count before dedup: {payload['count_before_dedup']}")
    print(f"count after dedup: {payload['count']}")
    print("example first 10 entries:")
    print(json.dumps(first_ten, indent=2))


if __name__ == "__main__":
    main()

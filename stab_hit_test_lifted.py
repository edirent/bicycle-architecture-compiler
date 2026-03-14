#!/usr/bin/env python3
"""Run the lifted stabilizer-equivalence test in physical space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bits_to_int(bits: list[int]) -> int:
    out = 0
    for index, bit in enumerate(bits):
        if bit not in (0, 1):
            raise ValueError(f"Expected binary bit, got {bit!r}")
        out |= bit << index
    return out


def pauli_string_to_logical_symplectic(pauli: str) -> list[int]:
    if len(pauli) != 12:
        raise ValueError(f"Logical Pauli string must have length 12, got {len(pauli)}")
    x_bits: list[int] = []
    z_bits: list[int] = []
    for char in pauli:
        if char == "I":
            x_bits.append(0)
            z_bits.append(0)
        elif char == "X":
            x_bits.append(1)
            z_bits.append(0)
        elif char == "Z":
            x_bits.append(0)
            z_bits.append(1)
        elif char == "Y":
            x_bits.append(1)
            z_bits.append(1)
        else:
            raise ValueError(f"Invalid Pauli character {char!r}")
    return x_bits + z_bits


def symplectic_to_pauli(symplectic: list[int]) -> str:
    if len(symplectic) != 24:
        raise ValueError(f"Expected logical 24-bit symplectic vector, got {len(symplectic)}")
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


def symplectic_product(left: tuple[int, int], right: tuple[int, int]) -> int:
    x1, z1 = left
    x2, z2 = right
    return ((x1 & z2).bit_count() + (z1 & x2).bit_count()) % 2


def symplectic_list_to_pair(symplectic: list[int]) -> tuple[int, int]:
    if len(symplectic) % 2 != 0:
        raise ValueError(f"Expected even-length symplectic vector, got {len(symplectic)}")
    half = len(symplectic) // 2
    return bits_to_int(symplectic[:half]), bits_to_int(symplectic[half:])


def symplectic_pair_to_list(pair: tuple[int, int], n_qubits: int) -> list[int]:
    x_bits, z_bits = pair
    return [(x_bits >> i) & 1 for i in range(n_qubits)] + [(z_bits >> i) & 1 for i in range(n_qubits)]


def build_rowspace_with_certificates(rows: list[int]) -> dict[int, tuple[int, int]]:
    basis: dict[int, tuple[int, int]] = {}
    for row_index, row in enumerate(rows):
        reduced = row
        combination = 1 << row_index
        while reduced:
            pivot = reduced.bit_length() - 1
            existing = basis.get(pivot)
            if existing is None:
                basis[pivot] = (reduced, combination)
                break
            reduced ^= existing[0]
            combination ^= existing[1]
    return basis


def reduce_with_certificate(basis: dict[int, tuple[int, int]], vector: int) -> tuple[bool, int]:
    reduced = vector
    combination = 0
    while reduced:
        pivot = reduced.bit_length() - 1
        existing = basis.get(pivot)
        if existing is None:
            return False, 0
        reduced ^= existing[0]
        combination ^= existing[1]
    return True, combination


def combination_mask_to_rows(mask: int, hx_count: int, hz_count: int) -> list[dict]:
    rows: list[dict] = []
    for row_index in range(hx_count + hz_count):
        if ((mask >> row_index) & 1) == 0:
            continue
        if row_index < hx_count:
            rows.append({"type": "X", "index": row_index})
        else:
            rows.append({"type": "Z", "index": row_index - hx_count})
    return rows


def validate_loaded_lift(physical_lift: dict) -> dict:
    if physical_lift.get("space") != "physical_css_and_logical_basis":
        raise ValueError(f"Unexpected physical lift space {physical_lift.get('space')!r}")
    n_physical = physical_lift["n_physical"]
    hx_rows = physical_lift["Hx"]
    hz_rows = physical_lift["Hz"]
    if any(len(row) != n_physical for row in hx_rows):
        raise ValueError("Hx rows do not all live in the declared physical code space.")
    if any(len(row) != n_physical for row in hz_rows):
        raise ValueError("Hz rows do not all live in the declared physical code space.")

    x_basis = [symplectic_list_to_pair(entry["symplectic"]) for entry in physical_lift["logical_basis"]["X"]]
    z_basis = [symplectic_list_to_pair(entry["symplectic"]) for entry in physical_lift["logical_basis"]["Z"]]
    if len(x_basis) != 12 or len(z_basis) != 12:
        raise ValueError("Physical lift must contain exactly 12 logical X reps and 12 logical Z reps.")
    for x_pair in x_basis:
        if x_pair[1] != 0:
            raise ValueError("Logical X representative is not X-only in physical symplectic form.")
    for z_pair in z_basis:
        if z_pair[0] != 0:
            raise ValueError("Logical Z representative is not Z-only in physical symplectic form.")

    hx_bits = [bits_to_int(row) for row in hx_rows]
    hz_bits = [bits_to_int(row) for row in hz_rows]
    x_commutes = all(all((x_pair[0] & hz_row).bit_count() % 2 == 0 for hz_row in hz_bits) for x_pair in x_basis)
    z_commutes = all(all((z_pair[1] & hx_row).bit_count() % 2 == 0 for hx_row in hx_bits) for z_pair in z_basis)
    xz_pairings_identity = all(
        symplectic_product(x_basis[i], z_basis[j]) == (1 if i == j else 0)
        for i in range(12)
        for j in range(12)
    )
    xx_pairings_zero = all(symplectic_product(x_basis[i], x_basis[j]) == 0 for i in range(12) for j in range(12))
    zz_pairings_zero = all(symplectic_product(z_basis[i], z_basis[j]) == 0 for i in range(12) for j in range(12))

    return {
        "n_physical": n_physical,
        "hx_count": len(hx_rows),
        "hz_count": len(hz_rows),
        "hx_bits": hx_bits,
        "hz_bits": hz_bits,
        "x_basis": x_basis,
        "z_basis": z_basis,
        "checks_passed": x_commutes and z_commutes and xz_pairings_identity and xx_pairings_zero and zz_pairings_zero,
        "check_details": {
            "logical_x_commutes_with_stabilizers": x_commutes,
            "logical_z_commutes_with_stabilizers": z_commutes,
            "xz_pairings_identity": xz_pairings_identity,
            "xx_pairings_zero": xx_pairings_zero,
            "zz_pairings_zero": zz_pairings_zero,
        },
    }


def lift_logical_symplectic(logical_symplectic: list[int], x_basis: list[tuple[int, int]], z_basis: list[tuple[int, int]]) -> tuple[int, int]:
    if len(logical_symplectic) != 24:
        raise ValueError(f"Logical symplectic vector must have length 24, got {len(logical_symplectic)}")
    x_acc = 0
    z_acc = 0
    for i in range(12):
        if logical_symplectic[i]:
            x_acc ^= x_basis[i][0]
            z_acc ^= x_basis[i][1]
        if logical_symplectic[12 + i]:
            x_acc ^= z_basis[i][0]
            z_acc ^= z_basis[i][1]
    return x_acc, z_acc


def stabilizer_rank(hx_bits: list[int], hz_bits: list[int], n_physical: int) -> int:
    stabilizer_rows = hx_bits + [row << n_physical for row in hz_bits]
    return len(build_rowspace_with_certificates(stabilizer_rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Logical 12-qubit Pauli string in left-to-right qubit order.")
    parser.add_argument(
        "--logical-natives",
        type=Path,
        default=Path("logical_natives_gross.json"),
        help="Logical native measurement JSON path.",
    )
    parser.add_argument(
        "--physical-lift",
        type=Path,
        default=Path("gross_physical_lift.json"),
        help="Physical lift JSON path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("lifted_hit_test_result.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--show-cert",
        action="store_true",
        help="Include a stabilizer row-combination witness for each hit.",
    )
    args = parser.parse_args()

    logical_natives = json.loads(args.logical_natives.read_text())
    physical_lift = json.loads(args.physical_lift.read_text())

    if logical_natives.get("space") != "logical_12q_symplectic":
        raise ValueError(f"Unexpected logical native space {logical_natives.get('space')!r}")
    if logical_natives.get("code") != "gross" or physical_lift.get("code") != "gross":
        raise ValueError("This lifted hit test currently expects gross-code artifacts.")

    lift_validation = validate_loaded_lift(physical_lift)
    n_physical = lift_validation["n_physical"]
    hx_bits = lift_validation["hx_bits"]
    hz_bits = lift_validation["hz_bits"]
    x_basis = lift_validation["x_basis"]
    z_basis = lift_validation["z_basis"]

    stabilizer_rows = hx_bits + [row << n_physical for row in hz_bits]
    stabilizer_basis = build_rowspace_with_certificates(stabilizer_rows)

    target_logical = pauli_string_to_logical_symplectic(args.target)
    target_physical = lift_logical_symplectic(target_logical, x_basis, z_basis)
    target_combined = target_physical[0] | (target_physical[1] << n_physical)

    hits: list[dict] = []
    for native in logical_natives["items"]:
        native_symplectic = native["symplectic"]
        if len(native_symplectic) != 24:
            raise ValueError(f"Logical native index {native['index']} is not 24 bits long.")
        if native["pauli"] != symplectic_to_pauli(native_symplectic):
            raise ValueError(f"Logical native index {native['index']} has inconsistent Pauli encoding.")

        native_physical = lift_logical_symplectic(native_symplectic, x_basis, z_basis)
        delta_pair = (target_physical[0] ^ native_physical[0], target_physical[1] ^ native_physical[1])
        delta_combined = delta_pair[0] | (delta_pair[1] << n_physical)
        is_hit, certificate_mask = reduce_with_certificate(stabilizer_basis, delta_combined)
        if not is_hit:
            continue
        hit = {
            "native_index": native["index"],
            "native_pauli": native["pauli"],
            "delta_in_physical_space": symplectic_pair_to_list(delta_pair, n_physical),
        }
        if args.show_cert:
            hit["certificate"] = {
                "row_count": certificate_mask.bit_count(),
                "rows": combination_mask_to_rows(certificate_mask, len(hx_bits), len(hz_bits)),
            }
        hits.append(hit)

    result = {
        "target_logical_pauli": args.target,
        "target_logical_symplectic": target_logical,
        "num_hits": len(hits),
        "hits": hits,
        "diagnostics": {
            "logical_dim": len(target_logical),
            "physical_dim": 2 * n_physical,
            "n_physical_qubits": n_physical,
            "hx_shape": [len(hx_bits), n_physical],
            "hz_shape": [len(hz_bits), n_physical],
            "stabilizer_rank": stabilizer_rank(hx_bits, hz_bits, n_physical),
            "lift_basis_checks_passed": lift_validation["checks_passed"],
            "lift_basis_check_details": lift_validation["check_details"],
        },
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"target_logical_pauli": args.target, "num_hits": len(hits)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare witness-search regimes for gross and two-gross efficiently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explain_boss_p import load_artifacts
from export_gross_lift import (
    Poly,
    bits_to_list,
    get_block_basis_with_shifts,
    iter_monomials,
    rank_of_rows,
    symplectic_product,
    symplectic_to_list,
)
from export_logical_natives import (
    iter_base_measurements,
    logical_symplectic_to_pauli,
    matmul_mod2,
    matpow_mod2,
    matvec_mod2,
    pauli_basis_vectors,
    transpose,
)
from search_nontrivial_witnesses import generate_targets_of_weight
from stab_hit_test_lifted import validate_loaded_lift


TWOGROSS_MX = [
    [0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0],
]

TWOGROSS_MY = [
    [1, 1, 1, 1, 1, 0],
    [1, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
]


def concat_block_bits(left_bits: int, right_bits: int, block_size: int) -> int:
    return left_bits | (right_bits << block_size)


def operator_support_to_bits(op: tuple[Poly, Poly], block_size: int) -> int:
    return concat_block_bits(op[0].vec_bits(), op[1].vec_bits(), block_size)


def measure_logical_symplectic_for_code(
    mx: list[list[int]],
    my: list[list[int]],
    basis_1: str,
    basis_7: str,
    aut_x: int,
    aut_y: int,
) -> list[int]:
    x1, z1 = pauli_basis_vectors(basis_1)
    x7, z7 = pauli_basis_vectors(basis_7)

    inv_x = (-aut_x) % 6
    inv_y = (-aut_y) % 6
    x_action = matmul_mod2(matpow_mod2(mx, inv_x), matpow_mod2(my, inv_y))
    z_action = transpose(matmul_mod2(matpow_mod2(mx, aut_x), matpow_mod2(my, aut_y)))

    map_x1 = matvec_mod2(x_action, x1)
    map_x7 = matvec_mod2(x_action, x7)
    map_z1 = matvec_mod2(z_action, z1)
    map_z7 = matvec_mod2(z_action, z7)
    return map_x1 + map_x7 + map_z1 + map_z7


def make_logical_natives_payload(code: str, mx: list[list[int]], my: list[list[int]]) -> dict:
    enumerated_items: list[dict] = []
    for aut_x in range(6):
        for aut_y in range(6):
            for basis_1, basis_7 in iter_base_measurements():
                symplectic = measure_logical_symplectic_for_code(mx, my, basis_1, basis_7, aut_x, aut_y)
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
    if len(enumerated_items) != 540:
        raise RuntimeError(f"Expected 540 enumerated native measurements for {code}, got {len(enumerated_items)}")
    deduped: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for item in enumerated_items:
        key = tuple(item["symplectic"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    if len(deduped) != 540:
        raise RuntimeError(f"Expected 540 deduplicated native measurements for {code}, got {len(deduped)}")
    return {
        "space": "logical_12q_symplectic",
        "code": code,
        "count_before_dedup": len(enumerated_items),
        "count": len(deduped),
        "items": [
            {
                "index": index,
                "pauli": item["pauli"],
                "symplectic": item["symplectic"],
                "metadata": item["metadata"],
            }
            for index, item in enumerate(deduped)
        ],
    }


def make_twogross_physical_lift() -> dict:
    order = (12, 12)
    block_size = order[0] * order[1]
    n_physical = 2 * block_size
    x = Poly.monomial(1, 0, order)
    y = Poly.monomial(0, 1, order)
    monomials = iter_monomials(order)

    a_poly = 1 + y + x**3 * y**-1
    b_poly = 1 + x + x**-1 * y**-3

    p = x**2 + (x**2 + x**8 + x**9) * y**2 + (x**3 + x**4) * y**3 + x**7 * y**4 + x**8 * y**6 + x**6 * y**7 + x**7 * y**11
    q = x**3 * y**2 + (x**5 + x**7 + x**11) * y**3 + x**8 * y**4 + x**8 * y**5 + x**6 * y**7 + x**4 * y**8 + x * y**9 + x * y**10
    r = (x**4 + x**11) * y**2 + (1 + x + x**5 + x**6) * y**5 + (x + x**8) * y**8 + (x**2 + x**10) * y**11
    s = (x**2 + x**11) * (y**6 + y**9 + y**10) + (x**8 + x**11) * y**7 + (x**5 + x**11) * y**8

    nu = x * y
    lx0 = p
    rx0 = q
    lz0 = nu * s.T
    rz0 = nu * r.T

    xshifts = [x**0, y**3, x**3 * y**7, x**11 * y**11, x**2 * y**9, x**7 * y**4]
    zshifts = [x**0, x * y, x**4, x**5 * y**4, x**4 * y**3, x**3 * y**5]
    primal_x_ops, primal_z_ops = get_block_basis_with_shifts(
        a_poly,
        b_poly,
        monomials,
        lx0,
        rx0,
        lz0,
        rz0,
        xshifts,
        zshifts,
    )

    dual_x_ops = [(rz.T, lz.T) for (lz, rz) in primal_z_ops]
    dual_z_ops = [(rx.T, lx.T) for (lx, rx) in primal_x_ops]

    hx_rows = [concat_block_bits((monomial * a_poly).vec_bits(), (monomial * b_poly).vec_bits(), block_size) for monomial in monomials]
    hz_rows = [concat_block_bits((monomial * b_poly.T).vec_bits(), (monomial * a_poly.T).vec_bits(), block_size) for monomial in monomials]
    logical_x_ops = primal_x_ops + dual_x_ops
    logical_z_ops = primal_z_ops + dual_z_ops
    logical_x_bits = [operator_support_to_bits(op, block_size) for op in logical_x_ops]
    logical_z_bits = [operator_support_to_bits(op, block_size) for op in logical_z_ops]

    hx_rank = rank_of_rows(hx_rows)
    hz_rank = rank_of_rows(hz_rows)
    k_logical = n_physical - hx_rank - hz_rank
    if k_logical != 12:
        raise RuntimeError(f"Expected 12 logical qubits for two-gross, got {k_logical}")

    if not all(((hx_row & hz_row).bit_count() % 2) == 0 for hx_row in hx_rows for hz_row in hz_rows):
        raise RuntimeError("Two-gross Hx and Hz do not satisfy CSS commutation.")
    if not all(
        symplectic_product((logical_x_bits[i], 0), (0, logical_z_bits[j])) == (1 if i == j else 0)
        for i in range(12)
        for j in range(12)
    ):
        raise RuntimeError("Two-gross logical X/Z basis does not have identity symplectic pairing.")

    return {
        "space": "physical_css_and_logical_basis",
        "code": "two-gross",
        "n_physical": n_physical,
        "k_logical": k_logical,
        "Hx": [bits_to_list(row, n_physical) for row in hx_rows],
        "Hz": [bits_to_list(row, n_physical) for row in hz_rows],
        "logical_basis": {
            "representation": "physical_symplectic_2N_bits",
            "X": [
                {"index": index, "symplectic": symplectic_to_list(bits, 0, n_physical)}
                for index, bits in enumerate(logical_x_bits, start=1)
            ],
            "Z": [
                {"index": index, "symplectic": symplectic_to_list(0, bits, n_physical)}
                for index, bits in enumerate(logical_z_bits, start=1)
            ],
        },
    }


def lift_is_injective_mod_stabilizers(lift_validation: dict) -> bool:
    details = lift_validation["check_details"]
    return lift_validation["checks_passed"] and all(details.values())


def run_regime_scan(code: str, max_weight: int, logical_natives: dict, physical_lift: dict) -> dict:
    lift_validation = validate_loaded_lift(physical_lift)
    if not lift_is_injective_mod_stabilizers(lift_validation):
        raise RuntimeError(
            f"Cannot apply injective-lift pruning for {code}: lift validation failed {lift_validation['check_details']}"
        )

    native_set = {item["pauli"]: item["index"] for item in logical_natives["items"]}
    by_weight: dict[str, int] = {}
    total_scanned = 0
    native_targets = 0
    trivial_self_hit_targets: list[str] = []

    for weight in range(1, max_weight + 1):
        targets = generate_targets_of_weight(weight)
        by_weight[str(weight)] = len(targets)
        total_scanned += len(targets)
        for target in targets:
            if target in native_set:
                native_targets += 1
                trivial_self_hit_targets.append(target)

    summary = {
        "total_scanned": total_scanned,
        "by_weight": by_weight,
        "native_targets": native_targets,
        "total_hits": native_targets,
        "trivial_self_hits": native_targets,
        "nontrivial_witnesses": 0,
        "native_set_size": len(native_set),
    }

    return {
        "code": code,
        "max_weight": max_weight,
        "summary": summary,
        "examples": [],
        "trivial_self_hits_first_20": sorted(trivial_self_hit_targets)[:20],
        "method": {
            "pairwise_scan_pruned": True,
            "pruning_reason": (
                "The lifted basis validates as a true logical basis modulo stabilizers, so "
                "L(q)+L(n) in rowspan(S_phys) implies q=n. Distinct target/native witness "
                "checks are therefore impossible in this regime and are pruned."
            ),
            "lift_basis_check_details": lift_validation["check_details"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gross-max-weight", type=int, default=4, help="Maximum gross logical weight to scan.")
    parser.add_argument("--twogross-max-weight", type=int, default=3, help="Maximum two-gross logical weight to scan.")
    parser.add_argument(
        "--gross-out",
        type=Path,
        default=Path("witness_scan_gross_w4.json"),
        help="Gross scan output path.",
    )
    parser.add_argument(
        "--twogross-out",
        type=Path,
        default=Path("witness_scan_twogross_w3.json"),
        help="Two-gross scan output path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("witness_regime_summary.json"),
        help="Combined summary output path.",
    )
    args = parser.parse_args()

    gross_logical_natives, gross_physical_lift, _gross_validation = load_artifacts(
        Path("logical_natives_gross.json"),
        Path("gross_physical_lift.json"),
    )
    gross_result = run_regime_scan("gross", args.gross_max_weight, gross_logical_natives, gross_physical_lift)
    args.gross_out.write_text(json.dumps(gross_result, indent=2))

    twogross_logical_natives = make_logical_natives_payload("two-gross", TWOGROSS_MX, TWOGROSS_MY)
    twogross_physical_lift = make_twogross_physical_lift()
    twogross_result = run_regime_scan("two-gross", args.twogross_max_weight, twogross_logical_natives, twogross_physical_lift)
    args.twogross_out.write_text(json.dumps(twogross_result, indent=2))

    if (
        gross_result["summary"]["nontrivial_witnesses"] == 0
        and twogross_result["summary"]["nontrivial_witnesses"] == 0
    ):
        conclusion = "No nontrivial evidence found in either regime tested."
    else:
        if gross_result["summary"]["nontrivial_witnesses"] > 0:
            regime = f"gross up to weight {args.gross_max_weight}"
            examples = gross_result["examples"][:10]
        else:
            regime = f"two-gross up to weight {args.twogross_max_weight}"
            examples = twogross_result["examples"][:10]
        conclusion = f"Nontrivial witnesses first appear in regime {regime} with examples {examples}"

    combined = {
        "gross_up_to_w4": gross_result["summary"],
        "twogross_up_to_w3": twogross_result["summary"],
        "conclusion": conclusion,
    }
    args.summary_out.write_text(json.dumps(combined, indent=2))

    print(f"gross up to w={args.gross_max_weight}: genuine nontrivial witnesses = {gross_result['summary']['nontrivial_witnesses']}")
    print(
        f"two-gross up to w={args.twogross_max_weight}: genuine nontrivial witnesses = "
        f"{twogross_result['summary']['nontrivial_witnesses']}"
    )
    if conclusion == "No nontrivial evidence found in either regime tested.":
        usefulness = "Beyond trivial native self-hits, the advisor method shows no real evidence of usefulness in the regimes tested."
    else:
        usefulness = "Beyond trivial native self-hits, the advisor method does show real evidence of usefulness in the regimes tested."
    print(usefulness)
    print(conclusion)


if __name__ == "__main__":
    main()

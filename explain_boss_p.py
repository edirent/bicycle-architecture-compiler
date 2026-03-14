#!/usr/bin/env python3
"""Explain stabilizer witnesses for the advisor-style boss-P question."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stab_hit_test_lifted import (
    build_rowspace_with_certificates,
    combination_mask_to_rows,
    lift_logical_symplectic,
    pauli_string_to_logical_symplectic,
    reduce_with_certificate,
    symplectic_pair_to_list,
    symplectic_to_pauli,
    validate_loaded_lift,
)


X1X7 = "XIIIIIXIIIII"
X1X8 = "XIIIIIIXIIII"


def split_certificate_rows(rows: list[dict]) -> tuple[list[int], list[int], list[str]]:
    hx = [row["index"] for row in rows if row["type"] == "X"]
    hz = [row["index"] for row in rows if row["type"] == "Z"]
    labels = [f"Hx[{index}]" for index in hx] + [f"Hz[{index}]" for index in hz]
    return hx, hz, labels


def load_artifacts(logical_natives_path: Path, physical_lift_path: Path) -> tuple[dict, dict, dict]:
    logical_natives = json.loads(logical_natives_path.read_text())
    physical_lift = json.loads(physical_lift_path.read_text())

    if logical_natives.get("space") != "logical_12q_symplectic":
        raise ValueError(f"Unexpected logical native space {logical_natives.get('space')!r}")
    if logical_natives.get("code") != "gross" or physical_lift.get("code") != "gross":
        raise ValueError("This script expects gross-code lifted artifacts.")

    lift_validation = validate_loaded_lift(physical_lift)
    if not lift_validation["checks_passed"]:
        raise ValueError(f"Physical lift validation failed: {lift_validation['check_details']}")
    return logical_natives, physical_lift, lift_validation


def analyze_target(target: str, logical_natives: dict, lift_validation: dict) -> dict:
    n_physical = lift_validation["n_physical"]
    hx_bits = lift_validation["hx_bits"]
    hz_bits = lift_validation["hz_bits"]
    x_basis = lift_validation["x_basis"]
    z_basis = lift_validation["z_basis"]

    stabilizer_rows = hx_bits + [row << n_physical for row in hz_bits]
    stabilizer_basis = build_rowspace_with_certificates(stabilizer_rows)

    target_logical = pauli_string_to_logical_symplectic(target)
    target_physical = lift_logical_symplectic(target_logical, x_basis, z_basis)

    hits: list[dict] = []
    for native in logical_natives["items"]:
        native_symplectic = native["symplectic"]
        if native["pauli"] != symplectic_to_pauli(native_symplectic):
            raise ValueError(f"Logical native index {native['index']} has inconsistent Pauli encoding.")
        native_physical = lift_logical_symplectic(native_symplectic, x_basis, z_basis)
        delta_pair = (target_physical[0] ^ native_physical[0], target_physical[1] ^ native_physical[1])
        delta_combined = delta_pair[0] | (delta_pair[1] << n_physical)
        is_hit, certificate_mask = reduce_with_certificate(stabilizer_basis, delta_combined)
        if not is_hit:
            continue
        certificate_rows = combination_mask_to_rows(certificate_mask, len(hx_bits), len(hz_bits))
        hx_row_indices, hz_row_indices, certificate_labels = split_certificate_rows(certificate_rows)
        delta_zero = delta_combined == 0
        if delta_zero:
            interpretation = "identity stabilizer"
        else:
            interpretation = "nontrivial stabilizer witness"
        hits.append(
            {
                "native_index": native["index"],
                "native_pauli": native["pauli"],
                "delta_zero": delta_zero,
                "physical_delta": symplectic_pair_to_list(delta_pair, n_physical),
                "certificate_row_indices": certificate_labels,
                "hx_row_indices": hx_row_indices,
                "hz_row_indices": hz_row_indices,
                "interpretation": interpretation,
            }
        )

    return {
        "target": target,
        "target_logical_symplectic": target_logical,
        "num_hits": len(hits),
        "hits": hits,
        "checked_native_count": len(logical_natives["items"]),
    }


def summarize_x1x7(report: dict) -> str:
    nontrivial_hits = [hit for hit in report["hits"] if not hit["delta_zero"]]
    if report["num_hits"] == 0:
        return "No native hit was found for X1X7."
    if nontrivial_hits:
        return (
            "At least one nontrivial stabilizer witness exists: a different native is reached by "
            "multiplying X1X7 by a physical stabilizer."
        )
    return (
        "The only native hit is X1X7 itself with zero physical delta, so the witness is P = I. "
        "No nontrivial stabilizer witness to a different native exists in the 540-native set."
    )


def summarize_x1x8(report: dict) -> str:
    if report["num_hits"] != 0:
        return "Unexpectedly found a native hit for X1X8."
    return (
        "No native n has L(target)+L(n) in the physical stabilizer row space, so no stabilizer "
        "witness P exists within the native set."
    )


def print_hit_report(label: str, report: dict) -> None:
    print(f"{label}:")
    print(f"  logical target Pauli: {report['target']}")
    print(f"  native hits found: {report['num_hits']} / {report['checked_native_count']}")
    if report["num_hits"] == 0:
        print("  matched logical native Pauli: none")
        print("  physical delta: no stabilizer-space match for any native")
        print("  stabilizer certificate / row combination: none")
        return
    for hit in report["hits"]:
        print(f"  matched logical native Pauli: {hit['native_pauli']} (index {hit['native_index']})")
        if hit["delta_zero"]:
            print("  physical delta: zero vector")
            print("  stabilizer certificate / row combination: []")
        else:
            weight = sum(hit["physical_delta"])
            print(f"  physical delta: nonzero 288-bit symplectic vector, Hamming weight {weight}")
            print(f"  stabilizer certificate / row combination: {hit['certificate_row_indices']}")
        print(f"  interpretation: {hit['interpretation']}")


def main() -> None:
    parser = argparse.ArgumentParser()
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
        default=Path("boss_p_report.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    logical_natives, _physical_lift, lift_validation = load_artifacts(args.logical_natives, args.physical_lift)

    x1x7_report = analyze_target(X1X7, logical_natives, lift_validation)
    x1x7_report["summary"] = summarize_x1x7(x1x7_report)

    x1x8_report = analyze_target(X1X8, logical_natives, lift_validation)
    x1x8_report["summary"] = summarize_x1x8(x1x8_report)

    output = {
        "x1x7": x1x7_report,
        "x1x8": x1x8_report,
    }
    args.out.write_text(json.dumps(output, indent=2))

    print_hit_report("X1X7", x1x7_report)
    print()
    print_hit_report("X1X8", x1x8_report)
    print()
    if any(not hit["delta_zero"] for hit in x1x7_report["hits"]):
        x1x7_short = "X1X7: a nontrivial stabilizer witness exists to another native."
    elif x1x7_report["num_hits"] > 0:
        x1x7_short = "X1X7: only the trivial witness P = I exists in the native set."
    else:
        x1x7_short = "X1X7: no native witness was found."
    x1x8_short = "X1X8: every native gives a physical delta outside rowspan(S_phys), so no witness exists."
    print(x1x7_short)
    print(x1x8_short)


if __name__ == "__main__":
    main()

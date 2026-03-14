#!/usr/bin/env python3
"""Search low-weight logical Paulis for genuine nontrivial stabilizer witnesses."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from explain_boss_p import load_artifacts, split_certificate_rows
from stab_hit_test_lifted import (
    build_rowspace_with_certificates,
    combination_mask_to_rows,
    lift_logical_symplectic,
    pauli_string_to_logical_symplectic,
    reduce_with_certificate,
    symplectic_to_pauli,
)


NON_IDENTITY_PAULIS = ("X", "Y", "Z")


def generate_targets_of_weight(weight: int) -> list[str]:
    targets: list[str] = []
    for positions in itertools.combinations(range(12), weight):
        for occupied_paulis in itertools.product(NON_IDENTITY_PAULIS, repeat=weight):
            chars = ["I"] * 12
            for position, pauli in zip(positions, occupied_paulis):
                chars[position] = pauli
            targets.append("".join(chars))
    return targets


def prepare_native_data(logical_natives: dict, lift_validation: dict) -> tuple[dict[str, int], list[dict], dict[int, tuple[int, int]]]:
    x_basis = lift_validation["x_basis"]
    z_basis = lift_validation["z_basis"]
    n_physical = lift_validation["n_physical"]
    hx_bits = lift_validation["hx_bits"]
    hz_bits = lift_validation["hz_bits"]
    stabilizer_rows = hx_bits + [row << n_physical for row in hz_bits]
    stabilizer_basis = build_rowspace_with_certificates(stabilizer_rows)

    native_index_by_pauli: dict[str, int] = {}
    prepared_natives: list[dict] = []
    for native in logical_natives["items"]:
        native_pauli = native["pauli"]
        native_symplectic = native["symplectic"]
        if native_pauli != symplectic_to_pauli(native_symplectic):
            raise ValueError(f"Logical native index {native['index']} has inconsistent Pauli encoding.")
        native_index_by_pauli[native_pauli] = native["index"]
        prepared_natives.append(
            {
                "index": native["index"],
                "pauli": native_pauli,
                "physical_pair": lift_logical_symplectic(native_symplectic, x_basis, z_basis),
            }
        )
    return native_index_by_pauli, prepared_natives, stabilizer_basis


def target_hit_details(
    target_pauli: str,
    prepared_natives: list[dict],
    stabilizer_basis: dict[int, tuple[int, int]],
    lift_validation: dict,
) -> tuple[list[dict], tuple[int, int]]:
    n_physical = lift_validation["n_physical"]
    hx_count = lift_validation["hx_count"]
    hz_count = lift_validation["hz_count"]
    x_basis = lift_validation["x_basis"]
    z_basis = lift_validation["z_basis"]

    target_logical = pauli_string_to_logical_symplectic(target_pauli)
    target_physical = lift_logical_symplectic(target_logical, x_basis, z_basis)

    hits: list[dict] = []
    for native in prepared_natives:
        native_physical = native["physical_pair"]
        delta_pair = (
            target_physical[0] ^ native_physical[0],
            target_physical[1] ^ native_physical[1],
        )
        delta_combined = delta_pair[0] | (delta_pair[1] << n_physical)
        is_hit, certificate_mask = reduce_with_certificate(stabilizer_basis, delta_combined)
        if not is_hit:
            continue
        certificate_rows = combination_mask_to_rows(certificate_mask, hx_count, hz_count)
        hx_row_indices, hz_row_indices, certificate_labels = split_certificate_rows(certificate_rows)
        hits.append(
            {
                "native_index": native["index"],
                "native_pauli": native["pauli"],
                "delta_zero": delta_combined == 0,
                "certificate_empty": certificate_mask == 0,
                "certificate_size": len(certificate_rows),
                "certificate_row_indices": certificate_labels,
                "hx_row_indices": hx_row_indices,
                "hz_row_indices": hz_row_indices,
                "delta_weight_physical": delta_pair[0].bit_count() + delta_pair[1].bit_count(),
            }
        )
    return hits, target_physical


def is_native_only_trivial_self_hit(target_pauli: str, hits: list[dict]) -> bool:
    if len(hits) != 1:
        return False
    hit = hits[0]
    return (
        hit["native_pauli"] == target_pauli
        and hit["delta_zero"]
        and hit["certificate_empty"]
    )


def sort_nontrivial_examples(examples: list[dict]) -> list[dict]:
    return sorted(
        examples,
        key=lambda example: (
            example["weight"],
            example["certificate_size"],
            example["delta_weight_physical"],
            example["target_pauli"],
            example["native_pauli"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-weight", type=int, default=3, help="Maximum logical Pauli weight to scan.")
    parser.add_argument("--limit", type=int, default=20, help="How many examples to print.")
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
        default=Path("nontrivial_witnesses_gross.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    if args.max_weight < 1:
        raise ValueError("--max-weight must be at least 1.")
    if args.max_weight > 12:
        raise ValueError("--max-weight cannot exceed 12 logical qubits.")

    logical_natives, _physical_lift, lift_validation = load_artifacts(args.logical_natives, args.physical_lift)
    native_index_by_pauli, prepared_natives, stabilizer_basis = prepare_native_data(logical_natives, lift_validation)

    by_weight: dict[str, int] = {}
    total_scanned = 0
    native_targets = 0
    total_hits = 0
    trivial_self_hits = 0
    trivial_self_hit_targets: list[str] = []
    nontrivial_examples: list[dict] = []

    for weight in range(1, args.max_weight + 1):
        targets = generate_targets_of_weight(weight)
        by_weight[str(weight)] = len(targets)
        total_scanned += len(targets)
        for target_pauli in targets:
            target_is_native = target_pauli in native_index_by_pauli
            if target_is_native:
                native_targets += 1
            hits, _target_physical = target_hit_details(target_pauli, prepared_natives, stabilizer_basis, lift_validation)
            total_hits += len(hits)
            if is_native_only_trivial_self_hit(target_pauli, hits):
                trivial_self_hits += 1
                trivial_self_hit_targets.append(target_pauli)
            for hit in hits:
                if target_is_native:
                    if hit["native_pauli"] == target_pauli:
                        continue
                    if hit["delta_zero"]:
                        continue
                    if hit["certificate_empty"]:
                        continue
                    continue
                if hit["native_pauli"] == target_pauli:
                    continue
                if hit["delta_zero"]:
                    continue
                if hit["certificate_empty"]:
                    continue
                nontrivial_examples.append(
                    {
                        "target_pauli": target_pauli,
                        "weight": weight,
                        "native_index": hit["native_index"],
                        "native_pauli": hit["native_pauli"],
                        "certificate_row_indices": hit["certificate_row_indices"],
                        "hx_row_indices": hit["hx_row_indices"],
                        "hz_row_indices": hit["hz_row_indices"],
                        "certificate_size": hit["certificate_size"],
                        "delta_weight_physical": hit["delta_weight_physical"],
                    }
                )

    nontrivial_examples = sort_nontrivial_examples(nontrivial_examples)
    trivial_self_hit_targets = sorted(trivial_self_hit_targets)

    output = {
        "code": "gross",
        "max_weight": args.max_weight,
        "summary": {
            "total_scanned": total_scanned,
            "by_weight": by_weight,
            "native_targets": native_targets,
            "total_hits": total_hits,
            "trivial_self_hits": trivial_self_hits,
            "nontrivial_witnesses": len(nontrivial_examples),
        },
        "examples": nontrivial_examples,
        "native_only_trivial_self_hits": trivial_self_hit_targets,
    }
    args.out.write_text(json.dumps(output, indent=2))

    print(f"code: gross")
    print(f"max weight scanned: {args.max_weight}")
    print(f"total targets scanned: {total_scanned}")
    print(f"scanned by weight: {json.dumps(by_weight, sort_keys=True)}")
    print(f"native targets: {native_targets}")
    print(f"total target/native hits: {total_hits}")
    print(f"trivial self-hits: {trivial_self_hits}")
    print(f"genuinely nontrivial witnesses: {len(nontrivial_examples)}")
    print()

    if nontrivial_examples:
        print(f"first {min(args.limit, len(nontrivial_examples))} nontrivial examples:")
        for example in nontrivial_examples[: args.limit]:
            print(
                f"  target={example['target_pauli']} weight={example['weight']} "
                f"native={example['native_pauli']}#{example['native_index']} "
                f"cert_size={example['certificate_size']} "
                f"delta_weight_physical={example['delta_weight_physical']} "
                f"certificate={example['certificate_row_indices']}"
            )
    else:
        print(f"no genuine nontrivial witness exists up to weight {args.max_weight}.")
    print()

    print(f"first {min(args.limit, len(trivial_self_hit_targets))} native-only trivial self-hits:")
    for target_pauli in trivial_self_hit_targets[: args.limit]:
        print(f"  {target_pauli}")


if __name__ == "__main__":
    main()

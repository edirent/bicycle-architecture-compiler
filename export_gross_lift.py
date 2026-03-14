#!/usr/bin/env python3
"""Export gross-code physical CSS checks and a physical logical lift basis.

Repo-backed sources:
- notebooks/gross_code_automorphisms.ipynb
- notebooks/polynomial.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


ORDER = (12, 6)
BLOCK_SIZE = ORDER[0] * ORDER[1]
N_PHYSICAL = 2 * BLOCK_SIZE


def monomial_index(term: tuple[int, int], order: tuple[int, int]) -> int:
    return term[0] * order[1] + term[1]


@dataclass(frozen=True)
class Poly:
    terms: frozenset[tuple[int, int]]
    order: tuple[int, int]

    @staticmethod
    def zero(order: tuple[int, int]) -> "Poly":
        return Poly(frozenset(), order)

    @staticmethod
    def one(order: tuple[int, int]) -> "Poly":
        return Poly(frozenset({(0, 0)}), order)

    @staticmethod
    def monomial(ax: int, ay: int, order: tuple[int, int]) -> "Poly":
        return Poly(frozenset({(ax % order[0], ay % order[1])}), order)

    def __post_init__(self) -> None:
        canonical = frozenset((ax % self.order[0], ay % self.order[1]) for ax, ay in self.terms)
        object.__setattr__(self, "terms", canonical)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __contains__(self, item: object) -> bool:
        if item == 1:
            return (0, 0) in self.terms
        if isinstance(item, tuple) and len(item) == 2:
            return ((item[0] % self.order[0]), (item[1] % self.order[1])) in self.terms
        return False

    def __add__(self, other: "Poly | int") -> "Poly":
        other_poly = self._coerce(other)
        return Poly(self.terms.symmetric_difference(other_poly.terms), self.order)

    def __radd__(self, other: "Poly | int") -> "Poly":
        return self.__add__(other)

    def __mul__(self, other: "Poly | int") -> "Poly":
        other_poly = self._coerce(other)
        terms = set()
        for ax, ay in self.terms:
            for bx, by in other_poly.terms:
                term = ((ax + bx) % self.order[0], (ay + by) % self.order[1])
                if term in terms:
                    terms.remove(term)
                else:
                    terms.add(term)
        return Poly(frozenset(terms), self.order)

    def __rmul__(self, other: "Poly | int") -> "Poly":
        return self.__mul__(other)

    def __pow__(self, exponent: int) -> "Poly":
        if exponent == 0:
            return Poly.one(self.order)
        if exponent < 0:
            if len(self.terms) != 1:
                raise ValueError("Negative powers are only supported for monomials.")
            (ax, ay) = next(iter(self.terms))
            return Poly.monomial(-ax * abs(exponent), -ay * abs(exponent), self.order)
        result = Poly.one(self.order)
        base = self
        power = exponent
        while power > 0:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result

    @property
    def T(self) -> "Poly":
        return Poly(
            frozenset(((-ax) % self.order[0], (-ay) % self.order[1]) for ax, ay in self.terms),
            self.order,
        )

    def vec_bits(self) -> int:
        out = 0
        for term in self.terms:
            out |= 1 << monomial_index(term, self.order)
        return out

    def terms_as_lists(self) -> list[list[int]]:
        ordered = sorted(self.terms, key=lambda term: monomial_index(term, self.order))
        return [[ax, ay] for ax, ay in ordered]

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        ordered = sorted(self.terms, key=lambda term: monomial_index(term, self.order))
        return " + ".join(term_to_string(term) for term in ordered)

    def _coerce(self, other: "Poly | int") -> "Poly":
        if isinstance(other, Poly):
            if other.order != self.order:
                raise ValueError(f"Order mismatch: {self.order} != {other.order}")
            return other
        if other == 0:
            return Poly.zero(self.order)
        if other == 1:
            return Poly.one(self.order)
        raise TypeError(f"Cannot coerce {other!r} to Poly")


def term_to_string(term: tuple[int, int]) -> str:
    ax, ay = term
    if (ax, ay) == (0, 0):
        return "1"
    factors: list[str] = []
    if ax != 0:
        factors.append("x" if ax == 1 else f"x**{ax}")
    if ay != 0:
        factors.append("y" if ay == 1 else f"y**{ay}")
    return "*".join(factors)


def iter_monomials(order: tuple[int, int]) -> list[Poly]:
    return [Poly.monomial(ax, ay, order) for ax in range(order[0]) for ay in range(order[1])]


def concat_block_bits(left_bits: int, right_bits: int) -> int:
    return left_bits | (right_bits << BLOCK_SIZE)


def bits_to_list(bits: int, width: int) -> list[int]:
    return [(bits >> i) & 1 for i in range(width)]


def symplectic_to_list(x_bits: int, z_bits: int, n_qubits: int) -> list[int]:
    return bits_to_list(x_bits, n_qubits) + bits_to_list(z_bits, n_qubits)


def symplectic_product(left: tuple[int, int], right: tuple[int, int]) -> int:
    x1, z1 = left
    x2, z2 = right
    return ((x1 & z2).bit_count() + (z1 & x2).bit_count()) % 2


def build_xor_basis(rows: list[int]) -> dict[int, int]:
    basis: dict[int, int] = {}
    for row in rows:
        reduced = row
        while reduced:
            pivot = reduced.bit_length() - 1
            existing = basis.get(pivot)
            if existing is None:
                basis[pivot] = reduced
                break
            reduced ^= existing
    return basis


def rank_of_rows(rows: list[int]) -> int:
    return len(build_xor_basis(rows))


def in_rowspace(rows: list[int], vector: int) -> bool:
    basis = build_xor_basis(rows)
    reduced = vector
    while reduced:
        pivot = reduced.bit_length() - 1
        existing = basis.get(pivot)
        if existing is None:
            return False
        reduced ^= existing
    return True


def anticommute(lx: Poly, rx: Poly, lz: Poly, rz: Poly) -> bool:
    return 1 in (lx * lz.T + rx * rz.T)


def check_logical_qubit(a_poly: Poly, b_poly: Poly, monomials: list[Poly], lx: Poly, rx: Poly, lz: Poly, rz: Poly) -> None:
    if not anticommute(lx, rx, lz, rz):
        raise RuntimeError("Provided logical X and Z operators do not anticommute.")
    for monomial in monomials:
        if anticommute(monomial * a_poly, monomial * b_poly, lz, rz):
            raise RuntimeError("Candidate logical operator does not commute with an X stabilizer.")
        if anticommute(lx, rx, monomial * b_poly.T, monomial * a_poly.T):
            raise RuntimeError("Candidate logical operator does not commute with a Z stabilizer.")


def get_block_basis_with_shifts(
    a_poly: Poly,
    b_poly: Poly,
    monomials: list[Poly],
    lx: Poly,
    rx: Poly,
    lz: Poly,
    rz: Poly,
    xshifts: list[Poly],
    zshifts: list[Poly],
) -> tuple[list[tuple[Poly, Poly]], list[tuple[Poly, Poly]]]:
    if len(xshifts) != len(zshifts):
        raise RuntimeError("Shift lists must have the same length.")
    check_logical_qubit(
        a_poly,
        b_poly,
        monomials,
        xshifts[0] * lx,
        xshifts[0] * rx,
        zshifts[0] * lz,
        zshifts[0] * rz,
    )
    x_ops = [(alpha * lx, alpha * rx) for alpha in xshifts]
    z_ops = [(beta * lz, beta * rz) for beta in zshifts]
    for i, alpha in enumerate(xshifts):
        for j, beta in enumerate(zshifts):
            if anticommute(alpha * lx, alpha * rx, beta * lz, beta * rz) != (i == j):
                raise RuntimeError("Shift-derived basis is not symplectic.")
    return x_ops, z_ops


def operator_support_to_x_bits(op: tuple[Poly, Poly]) -> int:
    left_bits, right_bits = op[0].vec_bits(), op[1].vec_bits()
    return concat_block_bits(left_bits, right_bits)


def operator_support_to_z_bits(op: tuple[Poly, Poly]) -> int:
    left_bits, right_bits = op[0].vec_bits(), op[1].vec_bits()
    return concat_block_bits(left_bits, right_bits)


def make_gross_system_3_data() -> dict:
    order = ORDER
    x = Poly.monomial(1, 0, order)
    y = Poly.monomial(0, 1, order)
    monomials = iter_monomials(order)

    a_poly = 1 + y + x**3 * y**-1
    b_poly = 1 + x + x**-1 * y**-3

    p = x**4 + x**5 + x**6 * y + x**4 * y**2 + x**5 * y**4 + x**6 * y**5
    q = x**3 + x**4 + x**3 * y + x**3 * y**2 + x**4 * y**2 + x**3 * y**5
    r = 1 + x**8 + x * y + x**9 * y + x**3 * y**4 + x**11 * y**4
    s = x + x**9 + x**4 * y**4 + x**8 * y**4 + y**5 + x**8 * y**5

    nu = x * y
    lx0 = p
    rx0 = q
    lz0 = nu * s.T
    rz0 = nu * r.T

    xshifts = [x**0, x**3 * y**5, x**11 * y**5, x**10 * y, x**5 * y**4, x**4 * y**2]
    zshifts = [x**0, x**2 * y**4, x * y**2, x**2 * y**5, x * y, x**3 * y]
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

    hx_rows = [concat_block_bits((monomial * a_poly).vec_bits(), (monomial * b_poly).vec_bits()) for monomial in monomials]
    hz_rows = [concat_block_bits((monomial * b_poly.T).vec_bits(), (monomial * a_poly.T).vec_bits()) for monomial in monomials]

    logical_x_ops = primal_x_ops + dual_x_ops
    logical_z_ops = primal_z_ops + dual_z_ops
    logical_x_bits = [operator_support_to_x_bits(op) for op in logical_x_ops]
    logical_z_bits = [operator_support_to_z_bits(op) for op in logical_z_ops]

    hx_rank = rank_of_rows(hx_rows)
    hz_rank = rank_of_rows(hz_rows)
    k_logical = N_PHYSICAL - hx_rank - hz_rank

    commutation_checks = all(((hx_row & hz_row).bit_count() % 2) == 0 for hx_row in hx_rows for hz_row in hz_rows)
    x_commutes_with_stabilizers = all(
        all((x_bits & hz_row).bit_count() % 2 == 0 for hz_row in hz_rows) for x_bits in logical_x_bits
    )
    z_commutes_with_stabilizers = all(
        all((z_bits & hx_row).bit_count() % 2 == 0 for hx_row in hx_rows) for z_bits in logical_z_bits
    )
    xx_pairings_zero = all(
        symplectic_product((logical_x_bits[i], 0), (logical_x_bits[j], 0)) == 0
        for i in range(12)
        for j in range(12)
    )
    zz_pairings_zero = all(
        symplectic_product((0, logical_z_bits[i]), (0, logical_z_bits[j])) == 0
        for i in range(12)
        for j in range(12)
    )
    xz_pairings_identity = all(
        symplectic_product((logical_x_bits[i], 0), (0, logical_z_bits[j])) == (1 if i == j else 0)
        for i in range(12)
        for j in range(12)
    )
    x_independent_mod_hx = rank_of_rows(hx_rows + logical_x_bits) == hx_rank + 12
    z_independent_mod_hz = rank_of_rows(hz_rows + logical_z_bits) == hz_rank + 12

    logical_basis = {"X": [], "Z": []}
    support_labels = ["primal"] * 6 + ["dual"] * 6
    support_ops_x = logical_x_ops
    support_ops_z = logical_z_ops
    for index, (block_label, op_bits, op_support) in enumerate(zip(support_labels, logical_x_bits, support_ops_x), start=1):
        logical_basis["X"].append(
            {
                "index": index,
                "block": block_label,
                "symplectic": symplectic_to_list(op_bits, 0, N_PHYSICAL),
                "support_terms": {
                    "L": op_support[0].terms_as_lists(),
                    "R": op_support[1].terms_as_lists(),
                },
                "support_polynomials": {
                    "L": str(op_support[0]),
                    "R": str(op_support[1]),
                },
            }
        )
    for index, (block_label, op_bits, op_support) in enumerate(zip(support_labels, logical_z_bits, support_ops_z), start=1):
        logical_basis["Z"].append(
            {
                "index": index,
                "block": block_label,
                "symplectic": symplectic_to_list(0, op_bits, N_PHYSICAL),
                "support_terms": {
                    "L": op_support[0].terms_as_lists(),
                    "R": op_support[1].terms_as_lists(),
                },
                "support_polynomials": {
                    "L": str(op_support[0]),
                    "R": str(op_support[1]),
                },
            }
        )

    return {
        "space": "physical_css_and_logical_basis",
        "code": "gross",
        "toric": True,
        "order": [ORDER[0], ORDER[1]],
        "block_size": BLOCK_SIZE,
        "n_physical": N_PHYSICAL,
        "k_logical": k_logical,
        "Hx": [bits_to_list(row, N_PHYSICAL) for row in hx_rows],
        "Hz": [bits_to_list(row, N_PHYSICAL) for row in hz_rows],
        "logical_basis": {
            "representation": "physical_symplectic_2N_bits",
            "logical_order": "qubits 1-6 primal block, qubits 7-12 dual block",
            **logical_basis,
        },
        "notes": {
            "how_basis_was_derived": (
                "Reproduced from notebooks/gross_code_automorphisms.ipynb "
                "get_automorphisms_for_system_3() with toric=True, using "
                "get_block_basis_with_shifts() for the primal block and the notebook's "
                "documented ZX-duality rule for the dual block."
            ),
            "source_functions": [
                "notebooks/gross_code_automorphisms.ipynb:get_block_basis_with_shifts",
                "notebooks/gross_code_automorphisms.ipynb:check_logical_qubit",
                "notebooks/polynomial.py:Polynomial",
            ],
            "internal_intermediate_representation": (
                "Bivariate polynomial supports over the L and R physical blocks. "
                "The exported basis vectors are converted to final physical symplectic form."
            ),
            "validation": {
                "hx_rank": hx_rank,
                "hz_rank": hz_rank,
                "css_commutation_checks_passed": commutation_checks,
                "logical_x_commutes_with_stabilizers": x_commutes_with_stabilizers,
                "logical_z_commutes_with_stabilizers": z_commutes_with_stabilizers,
                "xx_pairings_zero": xx_pairings_zero,
                "zz_pairings_zero": zz_pairings_zero,
                "xz_pairings_identity": xz_pairings_identity,
                "logical_x_independent_mod_hx": x_independent_mod_hx,
                "logical_z_independent_mod_hz": z_independent_mod_hz,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("gross_physical_lift.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    payload = make_gross_system_3_data()
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"physical qubit count: {payload['n_physical']}")
    print(f"Hx shape: ({len(payload['Hx'])}, {len(payload['Hx'][0])})")
    print(f"Hz shape: ({len(payload['Hz'])}, {len(payload['Hz'][0])})")
    print(f"logical X reps: {len(payload['logical_basis']['X'])}")
    print(f"logical Z reps: {len(payload['logical_basis']['Z'])}")
    print("validation:")
    print(json.dumps(payload["notes"]["validation"], indent=2))


if __name__ == "__main__":
    main()

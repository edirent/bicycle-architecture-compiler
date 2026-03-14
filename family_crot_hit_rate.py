import csv
import json
import itertools
import argparse
from collections import defaultdict

PAULI_SET = set("IXYZ")


# ----------------------------
# Pauli / symplectic utilities
# ----------------------------

def normalize_pauli_string(s: str) -> str:
    s = s.strip().upper()
    if not s:
        raise ValueError("empty Pauli string")
    bad = set(s) - PAULI_SET
    if bad:
        raise ValueError(f"invalid Pauli string {s}, bad chars = {bad}")
    return s


def pauli_to_bits(pauli: str) -> int:
    """
    Map an 11-qubit or 12-qubit Pauli string to a bit-packed symplectic vector:
      low n bits   = x
      high n bits  = z
    I -> (0,0), X -> (1,0), Z -> (0,1), Y -> (1,1)
    """
    n = len(pauli)
    x = 0
    z = 0
    for i, c in enumerate(pauli):
        if c == 'I':
            pass
        elif c == 'X':
            x |= (1 << i)
        elif c == 'Z':
            z |= (1 << i)
        elif c == 'Y':
            x |= (1 << i)
            z |= (1 << i)
        else:
            raise ValueError(f"bad Pauli char: {c}")
    return x | (z << n)


def bits_to_pauli(bits: int, n: int) -> str:
    x = bits & ((1 << n) - 1)
    z = bits >> n
    out = []
    for i in range(n):
        xb = (x >> i) & 1
        zb = (z >> i) & 1
        if xb == 0 and zb == 0:
            out.append('I')
        elif xb == 1 and zb == 0:
            out.append('X')
        elif xb == 0 and zb == 1:
            out.append('Z')
        else:
            out.append('Y')
    return ''.join(out)


def popcount(x: int) -> int:
    return x.bit_count()


def symplectic_inner(v: int, w: int, n: int) -> int:
    """
    <(x|z),(x'|z')> = x·z' + z·x' mod 2
    """
    mask = (1 << n) - 1
    x1 = v & mask
    z1 = v >> n
    x2 = w & mask
    z2 = w >> n
    return (popcount(x1 & z2) + popcount(z1 & x2)) & 1


def transvection(v: int, s: int, n: int) -> int:
    """
    One native π/2 Pauli rotation about axis s acts on v as:
      τ_s(v) = v + <s,v> s
    over F2.
    """
    if symplectic_inner(s, v, n) == 0:
        return v
    return v ^ s


# ----------------------------
# Load native.csv
# ----------------------------

def load_native_csv(csv_path="native.csv"):
    """
    native_map[full_pauli] = [meta1, meta2, ...]
    """
    native_map = defaultdict(list)
    total_rows = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(not x.strip() for x in row):
                continue
            if len(row) != 6:
                raise ValueError(f"bad row len != 6: {row}")

            dx, dy, p1, p7, block0, block1 = [x.strip() for x in row]
            p1 = normalize_pauli_string(p1)
            p7 = normalize_pauli_string(p7)
            block0 = normalize_pauli_string(block0)
            block1 = normalize_pauli_string(block1)

            full_pauli = normalize_pauli_string(block0 + block1)
            if len(full_pauli) != 12:
                raise ValueError(f"full pauli length != 12: {full_pauli}")

            native_map[full_pauli].append({
                "dx": dx,
                "dy": dy,
                "p1": p1,
                "p7": p7,
                "block0": block0,
                "block1": block1,
                "full_pauli": full_pauli,
            })
            total_rows += 1

    return native_map, total_rows


# ----------------------------
# Native buckets N_Q and native rotation axes K
# ----------------------------

def drop_pivot(full_pauli: str, pivot_index: int) -> str:
    return full_pauli[:pivot_index] + full_pauli[pivot_index + 1:]


def build_native_buckets(native_map, pivot_index=0):
    """
    For each Q in {X,Y,Z}, define
      N_Q = { tail r : Q ⊗ r ∈ N }
    with witnesses.
    Also define native rotation axis set
      K = union_Q N_Q
    according to Eq. (15).
    """
    NQ = {'X': {}, 'Y': {}, 'Z': {}}
    axis_witnesses = defaultdict(list)

    for full_pauli, metas in native_map.items():
        q = full_pauli[pivot_index]
        if q not in {'X', 'Y', 'Z'}:
            continue
        tail = drop_pivot(full_pauli, pivot_index)
        tail_bits = pauli_to_bits(tail)

        if tail_bits not in NQ[q]:
            NQ[q][tail_bits] = {
                "tail": tail,
                "full_paulis": [],
                "witnesses": []
            }

        NQ[q][tail_bits]["full_paulis"].append(full_pauli)
        NQ[q][tail_bits]["witnesses"].extend(metas)

        axis_witnesses[tail_bits].append({
            "basis": q,
            "full_pauli": full_pauli,
            "meta": metas[0],
        })

    # Remove identity axis from K; its rotation is trivial
    K = set(axis_witnesses.keys())
    zero11 = pauli_to_bits('I' * 11)
    if zero11 in K:
        K.remove(zero11)

    return NQ, K, axis_witnesses


# ----------------------------
# Crot1 closure
# ----------------------------

def build_crot1_closure(NQ, K, axis_witnesses):
    """
    Crot1_Q = { τ_s(r) : r in N_Q, s in K }
    where τ_s is the symplectic transvection.
    Store one witness per discovered target tail.
    """
    Crot1 = {'X': {}, 'Y': {}, 'Z': {}}
    n = 11

    for Q in ['X', 'Y', 'Z']:
        for r_bits, payload in NQ[Q].items():
            # include the direct point too (0-rotation)
            if r_bits not in Crot1[Q]:
                Crot1[Q][r_bits] = {
                    "tail": payload["tail"],
                    "source_tail": payload["tail"],
                    "rotation_axis_tail": None,
                    "source_basis": Q,
                    "source_full_pauli": payload["full_paulis"][0],
                    "rotation_axis_full_pauli": None,
                }

            for s_bits in K:
                t_bits = transvection(r_bits, s_bits, n)
                if t_bits not in Crot1[Q]:
                    Crot1[Q][t_bits] = {
                        "tail": bits_to_pauli(t_bits, n),
                        "source_tail": payload["tail"],
                        "rotation_axis_tail": bits_to_pauli(s_bits, n),
                        "source_basis": Q,
                        "source_full_pauli": payload["full_paulis"][0],
                        "rotation_axis_full_pauli": axis_witnesses[s_bits][0]["full_pauli"],
                    }

    return Crot1


# ----------------------------
# Families
# ----------------------------

def family_alphabet(family_name: str):
    family_name = family_name.lower()
    if family_name == "full-x":
        return ["I", "X"]
    if family_name == "full-y":
        return ["I", "Y"]
    if family_name == "full-z":
        return ["I", "Z"]
    if family_name == "full-xz":
        return ["I", "X", "Z"]
    if family_name == "full-xyz":
        return ["I", "X", "Y", "Z"]
    raise ValueError(f"unknown family: {family_name}")


def enumerate_family_tails(family_name: str):
    alphabet = family_alphabet(family_name)
    for bits in itertools.product(alphabet, repeat=11):
        yield "".join(bits)


def make_full_pauli_from_Q_and_tail(Q: str, tail11: str, pivot_index: int = 0) -> str:
    if len(tail11) != 11:
        raise ValueError(f"tail length must be 11, got {len(tail11)}")
    if pivot_index == 0:
        return Q + tail11
    return tail11[:pivot_index] + Q + tail11[pivot_index:]


# ----------------------------
# Main analysis
# ----------------------------

def analyze_family_crot(csv_path="native.csv", family_name="full-x", pivot_index=0, mode="crot1", sample_limit=20):
    """
    mode:
      - "direct": only Q⊗P_i in N
      - "crot1": one native measurement + at most one native rotation
      - "crot-inf": unlimited native rotations, one measurement only
                    (theorem mode; by Tour de gross this is 100%)
    """
    native_map, total_rows = load_native_csv(csv_path)
    native_set = set(native_map.keys())

    NQ, K, axis_witnesses = build_native_buckets(native_map, pivot_index=pivot_index)
    Crot1 = build_crot1_closure(NQ, K, axis_witnesses)

    total_targets = 0
    direct_hits = 0
    crot1_only_hits = 0
    combined_hits = 0
    misses = 0

    direct_by_basis = {"X": 0, "Y": 0, "Z": 0}
    crot1_by_basis = {"X": 0, "Y": 0, "Z": 0}

    sample_direct = []
    sample_crot1 = []
    sample_miss = []

    for tail11 in enumerate_family_tails(family_name):
        total_targets += 1
        tail_bits = pauli_to_bits(tail11)

        direct_bases = []
        direct_fulls = []
        crot1_bases = []
        crot1_payloads = []

        for Q in ["X", "Y", "Z"]:
            full_pauli = make_full_pauli_from_Q_and_tail(Q, tail11, pivot_index=pivot_index)

            if full_pauli in native_set:
                direct_bases.append(Q)
                direct_fulls.append(full_pauli)
            elif mode == "crot1" and tail_bits in Crot1[Q]:
                crot1_bases.append(Q)
                crot1_payloads.append((Q, Crot1[Q][tail_bits]))
            elif mode == "crot-inf":
                # By theorem: native rotations generate the Clifford group,
                # so any Q⊗P_i is reachable from some native Q⊗R by Eq. (17).
                crot1_bases.append(Q)
                crot1_payloads.append((Q, None))

        if direct_bases:
            direct_hits += 1
            combined_hits += 1
            for Q in direct_bases:
                direct_by_basis[Q] += 1
            if len(sample_direct) < sample_limit:
                sample_direct.append({
                    "tail11": tail11,
                    "hit_bases": direct_bases,
                    "full_paulis": direct_fulls,
                })
        elif crot1_bases:
            crot1_only_hits += 1
            combined_hits += 1
            for Q in crot1_bases:
                crot1_by_basis[Q] += 1
            if len(sample_crot1) < sample_limit:
                if mode == "crot1":
                    entry = {
                        "tail11": tail11,
                        "hit_bases": crot1_bases,
                        "witnesses": []
                    }
                    for Q, payload in crot1_payloads:
                        entry["witnesses"].append({
                            "basis": Q,
                            "target_full": make_full_pauli_from_Q_and_tail(Q, tail11, pivot_index=pivot_index),
                            "source_full_pauli": payload["source_full_pauli"],
                            "source_tail": payload["source_tail"],
                            "rotation_axis_tail": payload["rotation_axis_tail"],
                            "rotation_axis_full_pauli": payload["rotation_axis_full_pauli"],
                        })
                    sample_crot1.append(entry)
                else:
                    sample_crot1.append({
                        "tail11": tail11,
                        "hit_bases": crot1_bases,
                        "note": "reachable by Eq. (17) with unrestricted native rotations (theorem mode)"
                    })
        else:
            misses += 1
            if len(sample_miss) < sample_limit:
                sample_miss.append({
                    "tail11": tail11,
                    "candidates": {
                        Q: make_full_pauli_from_Q_and_tail(Q, tail11, pivot_index=pivot_index)
                        for Q in ["X", "Y", "Z"]
                    }
                })

    return {
        "raw_rows": total_rows,
        "unique_native_count": len(native_set),
        "pivot_index": pivot_index,
        "family": family_name,
        "alphabet": family_alphabet(family_name),
        "mode": mode,
        "native_tail_bucket_sizes": {Q: len(NQ[Q]) for Q in ["X", "Y", "Z"]},
        "native_rotation_axis_count": len(K),
        "total_targets": total_targets,
        "direct_hits": direct_hits,
        "crot1_only_hits": crot1_only_hits,
        "combined_hits": combined_hits,
        "misses": misses,
        "direct_hit_rate_percent": 100.0 * direct_hits / total_targets,
        "crot_only_hit_rate_percent": 100.0 * crot1_only_hits / total_targets,
        "combined_hit_rate_percent": 100.0 * combined_hits / total_targets,
        "direct_by_basis": direct_by_basis,
        "crot_by_basis": crot1_by_basis,
        "sample_direct_hits": sample_direct,
        "sample_crot_hits": sample_crot1,
        "sample_misses": sample_miss,
    }


def print_report(result):
    print(f"========== {result['family'].upper()} {result['mode'].upper()} HIT TEST ==========")
    print(f"raw csv rows              : {result['raw_rows']}")
    print(f"unique native count       : {result['unique_native_count']}")
    print(f"pivot index               : {result['pivot_index']}")
    print(f"alphabet                  : {result['alphabet']}")
    print(f"native tail bucket sizes  : {result['native_tail_bucket_sizes']}")
    print(f"native rotation axis cnt  : {result['native_rotation_axis_count']}")
    print(f"total targets             : {result['total_targets']}")
    print()
    print(f"direct hits               : {result['direct_hits']}")
    print(f"{result['mode']}-only hits         : {result['crot1_only_hits']}")
    print(f"combined hits             : {result['combined_hits']}")
    print(f"misses                    : {result['misses']}")
    print()
    print(f"direct hit rate           : {result['direct_hit_rate_percent']:.6f}%")
    print(f"{result['mode']}-only hit rate      : {result['crot_only_hit_rate_percent']:.6f}%")
    print(f"combined hit rate         : {result['combined_hit_rate_percent']:.6f}%")
    print()
    print(f"direct by basis           : {result['direct_by_basis']}")
    print(f"{result['mode']} by basis          : {result['crot_by_basis']}")
    print()

    print("========== SAMPLE DIRECT HITS ==========")
    for item in result["sample_direct_hits"]:
        print(f"tail11     = {item['tail11']}")
        print(f"hit_bases  = {item['hit_bases']}")
        print(f"fulls      = {item['full_paulis']}")
        print()

    print("========== SAMPLE CROT HITS ==========")
    for item in result["sample_crot_hits"]:
        print(f"tail11     = {item['tail11']}")
        print(f"hit_bases  = {item['hit_bases']}")
        if "witnesses" in item:
            for w in item["witnesses"]:
                print(f"  basis                  = {w['basis']}")
                print(f"  target_full            = {w['target_full']}")
                print(f"  source_full_pauli      = {w['source_full_pauli']}")
                print(f"  source_tail            = {w['source_tail']}")
                print(f"  rotation_axis_tail     = {w['rotation_axis_tail']}")
                print(f"  rotation_axis_full     = {w['rotation_axis_full_pauli']}")
        else:
            print(f"  {item['note']}")
        print()

    print("========== SAMPLE MISSES ==========")
    for item in result["sample_misses"]:
        print(f"tail11     = {item['tail11']}")
        print(f"X⊗P_i      = {item['candidates']['X']}")
        print(f"Y⊗P_i      = {item['candidates']['Y']}")
        print(f"Z⊗P_i      = {item['candidates']['Z']}")
        print()

    print("NOTE 1: direct hit matches native membership in N.")
    print("NOTE 2: crot1 uses one native measurement + at most one native rotation transvection.")
    print("NOTE 3: crot-inf is theorem mode: with unrestricted native rotations, Eq. (17) gives 100% reachability.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=["full-x", "full-y", "full-z", "full-xz", "full-xyz"],
        required=True
    )
    parser.add_argument("--csv", default="native.csv")
    parser.add_argument("--pivot-index", type=int, default=0)
    parser.add_argument("--mode", choices=["direct", "crot1", "crot-inf"], default="crot1")
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()

    result = analyze_family_crot(
        csv_path=args.csv,
        family_name=args.family,
        pivot_index=args.pivot_index,
        mode=args.mode,
        sample_limit=args.sample_limit
    )

    print_report(result)

    out_file = f"{args.family}_{args.mode}_hit_report.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved report to {out_file}")


if __name__ == "__main__":
    main()
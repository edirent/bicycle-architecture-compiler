import csv
import json
import itertools
import argparse
from collections import defaultdict, Counter, deque

PAULI_SET = set("IXYZ")

PAULI_MUL = {
    ('I', 'I'): 'I', ('I', 'X'): 'X', ('I', 'Y'): 'Y', ('I', 'Z'): 'Z',
    ('X', 'I'): 'X', ('X', 'X'): 'I', ('X', 'Y'): 'Z', ('X', 'Z'): 'Y',
    ('Y', 'I'): 'Y', ('Y', 'X'): 'Z', ('Y', 'Y'): 'I', ('Y', 'Z'): 'X',
    ('Z', 'I'): 'Z', ('Z', 'X'): 'Y', ('Z', 'Y'): 'X', ('Z', 'Z'): 'I',
}


# ============================================================
# Basic Pauli / symplectic utilities
# ============================================================

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
    Pack an n-qubit Pauli into a 2n-bit symplectic integer:
      low n bits  = x part
      high n bits = z part
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
    mask = (1 << n) - 1
    x = bits & mask
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


def symplectic_inner(v: int, w: int, n: int) -> int:
    """
    <(x|z),(x'|z')> = x·z' + z·x' mod 2
    """
    mask = (1 << n) - 1
    x1 = v & mask
    z1 = v >> n
    x2 = w & mask
    z2 = w >> n
    return ((x1 & z2).bit_count() + (z1 & x2).bit_count()) & 1


def transvection(v: int, s: int, n: int) -> int:
    """
    τ_s(v) = v + <s,v> s  over F2
    This is the Pauli conjugation action of a π/2 rotation about axis s.
    """
    if symplectic_inner(s, v, n) == 0:
        return v
    return v ^ s


def pauli_mul(p: str, q: str) -> str:
    return ''.join(PAULI_MUL[(a, b)] for a, b in zip(p, q))


def is_commuting(p: str, q: str) -> bool:
    anti = 0
    for a, b in zip(p, q):
        if a != 'I' and b != 'I' and a != b:
            anti += 1
    return anti % 2 == 0


# ============================================================
# Load native.csv
# ============================================================

def load_native_csv(csv_path="native.csv"):
    """
    Returns:
      native_map[full_pauli] = [meta1, meta2, ...]
      total_rows
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


# ============================================================
# Build N_Q buckets and native rotation axes K
# ============================================================

def drop_pivot(full_pauli: str, pivot_index: int) -> str:
    return full_pauli[:pivot_index] + full_pauli[pivot_index + 1:]


def make_full_pauli_from_Q_and_tail(Q: str, tail11: str, pivot_index: int = 0) -> str:
    if len(tail11) != 11:
        raise ValueError(f"tail length must be 11, got {len(tail11)}")
    if pivot_index == 0:
        return Q + tail11
    return tail11[:pivot_index] + Q + tail11[pivot_index:]


def build_native_buckets(native_map, pivot_index=0):
    """
    NQ[Q] = dict from 11-qubit tail bits -> one witness payload
    K = set of all non-identity native rotation axes (11-qubit tails)
        derived from Eq. (15): any Q⊗R in N gives axis R
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
                "full_pauli": full_pauli,
                "meta": metas[0],
            }

        axis_witnesses[tail_bits].append({
            "basis": q,
            "full_pauli": full_pauli,
            "meta": metas[0],
        })

    zero11 = pauli_to_bits('I' * 11)
    K = set(axis_witnesses.keys())
    if zero11 in K:
        K.remove(zero11)

    return NQ, K, axis_witnesses


# ============================================================
# Bounded multi-source BFS for 1-measurement + rotations
# ============================================================

def build_bounded_rotation_reachability(NQ, K, max_rot):
    """
    For each pivot basis Q, compute the set of tails reachable from NQ[Q]
    using at most max_rot transvections.

    Returns:
      reach[Q][tail_bits] = minimal rotation depth
    """
    n = 11
    reach = {'X': {}, 'Y': {}, 'Z': {}}

    for Q in ['X', 'Y', 'Z']:
        visited = {}
        frontier = []

        for r_bits in NQ[Q].keys():
            visited[r_bits] = 0
            frontier.append(r_bits)

        for depth in range(1, max_rot + 1):
            nxt = []
            for v in frontier:
                for s in K:
                    u = transvection(v, s, n)
                    if u not in visited:
                        visited[u] = depth
                        nxt.append(u)
            frontier = nxt
            if not frontier:
                break

        reach[Q] = visited

    return reach


def find_rotation_path_to_native(target_bits, native_bucket_bits, K, max_rot):
    """
    Single-target BFS from target to nearest native bucket in <= max_rot steps.
    Since transvections are involutions, backward BFS is the same as forward BFS.

    Returns:
      dict with:
        depth
        source_bits
        axis_sequence_bits
      or None
    """
    n = 11
    if target_bits in native_bucket_bits:
        return {
            "depth": 0,
            "source_bits": target_bits,
            "axis_sequence_bits": [],
        }

    q = deque([target_bits])
    parent = {target_bits: None}
    parent_axis = {}
    depth = {target_bits: 0}

    while q:
        v = q.popleft()
        d = depth[v]
        if d == max_rot:
            continue

        for s in K:
            u = transvection(v, s, n)
            if u in depth:
                continue
            depth[u] = d + 1
            parent[u] = v
            parent_axis[u] = s

            if u in native_bucket_bits:
                axes = []
                cur = u
                while parent[cur] is not None:
                    axes.append(parent_axis[cur])
                    cur = parent[cur]
                axes.reverse()
                return {
                    "depth": d + 1,
                    "source_bits": u,
                    "axis_sequence_bits": axes,
                }

            q.append(u)

    return None


# ============================================================
# Optional C2 fallback
# ============================================================

def build_c2(native_map):
    native_strings = sorted(native_map.keys())
    native_set = set(native_strings)

    c2_map = defaultdict(list)
    commute_pairs = 0
    anti_pairs = 0

    n = len(native_strings)
    for i in range(n):
        p = native_strings[i]
        for j in range(i + 1, n):
            q = native_strings[j]

            if is_commuting(p, q):
                commute_pairs += 1
                r = pauli_mul(p, q)
                if r == "I" * 12:
                    continue
                if r in native_set:
                    continue

                c2_map[r].append({
                    "left_pauli": p,
                    "right_pauli": q,
                    "left_witness": native_map[p][0],
                    "right_witness": native_map[q][0],
                })
            else:
                anti_pairs += 1

    return c2_map, commute_pairs, anti_pairs


# ============================================================
# Families
# ============================================================

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


# ============================================================
# Main analysis
# ============================================================

def analyze_family(
    csv_path="native.csv",
    family_name="full-x",
    pivot_index=0,
    max_rot=4,
    enable_c2_fallback=True,
    sample_limit=20
):
    native_map, total_rows = load_native_csv(csv_path)
    native_set = set(native_map.keys())

    NQ, K, axis_witnesses = build_native_buckets(native_map, pivot_index=pivot_index)
    reach = build_bounded_rotation_reachability(NQ, K, max_rot=max_rot)

    c2_map = None
    c2_set = set()
    commute_pairs = 0
    anti_pairs = 0
    if enable_c2_fallback:
        c2_map, commute_pairs, anti_pairs = build_c2(native_map)
        c2_set = set(c2_map.keys())

    total_targets = 0
    direct_hits = 0
    rot_only_hits = 0
    c2_fallback_hits = 0
    one_measure_hits = 0
    overall_hits = 0
    misses = 0

    direct_by_basis = {"X": 0, "Y": 0, "Z": 0}
    rot_by_basis = {"X": 0, "Y": 0, "Z": 0}
    c2_by_basis = {"X": 0, "Y": 0, "Z": 0}

    rot_depth_hist = Counter()

    sample_direct = []
    sample_rot = []
    sample_c2 = []
    sample_miss = []

    for tail11 in enumerate_family_tails(family_name):
        total_targets += 1
        tail_bits = pauli_to_bits(tail11)

        direct_options = []
        rot_options = []
        c2_options = []

        for Q in ["X", "Y", "Z"]:
            full = make_full_pauli_from_Q_and_tail(Q, tail11, pivot_index=pivot_index)

            if full in native_set:
                direct_options.append((Q, full))

            if tail_bits in reach[Q] and full not in native_set:
                rot_options.append((Q, full, reach[Q][tail_bits]))

            if enable_c2_fallback and full in c2_set and full not in native_set:
                c2_options.append((Q, full))

        # Cost model: measurement count first, then rotation depth
        if direct_options:
            direct_hits += 1
            one_measure_hits += 1
            overall_hits += 1

            for Q, _ in direct_options:
                direct_by_basis[Q] += 1

            if len(sample_direct) < sample_limit:
                sample_direct.append({
                    "tail11": tail11,
                    "options": [{"basis": Q, "full_pauli": full} for Q, full in direct_options]
                })

        elif rot_options:
            rot_only_hits += 1
            one_measure_hits += 1
            overall_hits += 1

            best_depth = min(d for _, _, d in rot_options)
            best_options = [(Q, full, d) for (Q, full, d) in rot_options if d == best_depth]

            for Q, _, _ in best_options:
                rot_by_basis[Q] += 1
            rot_depth_hist[best_depth] += 1

            if len(sample_rot) < sample_limit:
                sample_rot.append({
                    "tail11": tail11,
                    "best_depth": best_depth,
                    "options": [
                        {"basis": Q, "full_pauli": full, "rot_depth": d}
                        for Q, full, d in best_options
                    ]
                })

        elif c2_options:
            c2_fallback_hits += 1
            overall_hits += 1

            for Q, _ in c2_options:
                c2_by_basis[Q] += 1

            if len(sample_c2) < sample_limit:
                entry = {"tail11": tail11, "options": []}
                for Q, full in c2_options[:3]:
                    ws = c2_map[full][:2]
                    entry["options"].append({
                        "basis": Q,
                        "full_pauli": full,
                        "witnesses": ws,
                    })
                sample_c2.append(entry)

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
        "max_rot": max_rot,
        "enable_c2_fallback": enable_c2_fallback,
        "native_tail_bucket_sizes": {Q: len(NQ[Q]) for Q in ["X", "Y", "Z"]},
        "native_rotation_axis_count": len(K),
        "c2_candidate_count": len(c2_set),
        "commuting_pairs": commute_pairs,
        "anti_commuting_pairs": anti_pairs,
        "total_targets": total_targets,
        "direct_hits": direct_hits,
        "rot_only_hits": rot_only_hits,
        "c2_fallback_hits": c2_fallback_hits,
        "one_measure_hits": one_measure_hits,
        "overall_hits": overall_hits,
        "misses": misses,
        "direct_hit_rate_percent": 100.0 * direct_hits / total_targets,
        "rot_only_hit_rate_percent": 100.0 * rot_only_hits / total_targets,
        "one_measure_hit_rate_percent": 100.0 * one_measure_hits / total_targets,
        "c2_fallback_hit_rate_percent": 100.0 * c2_fallback_hits / total_targets if total_targets else 0.0,
        "overall_hit_rate_percent": 100.0 * overall_hits / total_targets,
        "direct_by_basis": direct_by_basis,
        "rot_by_basis": rot_by_basis,
        "c2_by_basis": c2_by_basis,
        "rot_depth_hist": dict(sorted(rot_depth_hist.items())),
        "sample_direct_hits": sample_direct,
        "sample_rot_hits": sample_rot,
        "sample_c2_hits": sample_c2,
        "sample_misses": sample_miss,
    }


# ============================================================
# Optional single-target explanation
# ============================================================

def explain_target(
    target_full: str,
    csv_path="native.csv",
    pivot_index=0,
    max_rot=4,
    enable_c2_fallback=True
):
    target_full = normalize_pauli_string(target_full)
    if len(target_full) != 12:
        raise ValueError("target_full must be length 12")

    tail11 = drop_pivot(target_full, pivot_index)
    tail_bits = pauli_to_bits(tail11)

    native_map, _ = load_native_csv(csv_path)
    native_set = set(native_map.keys())
    NQ, K, axis_witnesses = build_native_buckets(native_map, pivot_index=pivot_index)

    c2_map = None
    c2_set = set()
    if enable_c2_fallback:
        c2_map, _, _ = build_c2(native_map)
        c2_set = set(c2_map.keys())

    report = {
        "target_full": target_full,
        "target_tail": tail11,
        "max_rot": max_rot,
        "per_basis": {}
    }

    for Q in ["X", "Y", "Z"]:
        full = make_full_pauli_from_Q_and_tail(Q, tail11, pivot_index=pivot_index)
        basis_report = {
            "basis": Q,
            "candidate_full": full,
            "direct": full in native_set,
            "rot_path": None,
            "c2": enable_c2_fallback and (full in c2_set),
        }

        if full not in native_set:
            path = find_rotation_path_to_native(
                target_bits=tail_bits,
                native_bucket_bits=set(NQ[Q].keys()),
                K=K,
                max_rot=max_rot
            )
            if path is not None:
                basis_report["rot_path"] = {
                    "depth": path["depth"],
                    "source_tail": bits_to_pauli(path["source_bits"], 11),
                    "source_full": make_full_pauli_from_Q_and_tail(
                        Q, bits_to_pauli(path["source_bits"], 11), pivot_index=pivot_index
                    ),
                    "axis_sequence_tail": [bits_to_pauli(a, 11) for a in path["axis_sequence_bits"]],
                }

        if enable_c2_fallback and full in c2_set:
            basis_report["c2_witnesses"] = c2_map[full][:2]

        report["per_basis"][Q] = basis_report

    return report


# ============================================================
# Printing
# ============================================================

def print_report(r):
    print(f"========== {r['family'].upper()} ROTATION-PATH SEARCH ==========")
    print(f"raw csv rows               : {r['raw_rows']}")
    print(f"unique native count        : {r['unique_native_count']}")
    print(f"pivot index                : {r['pivot_index']}")
    print(f"alphabet                   : {r['alphabet']}")
    print(f"max rotation depth         : {r['max_rot']}")
    print(f"use C2 fallback            : {r['enable_c2_fallback']}")
    print(f"native tail bucket sizes   : {r['native_tail_bucket_sizes']}")
    print(f"native rotation axis count : {r['native_rotation_axis_count']}")
    if r["enable_c2_fallback"]:
        print(f"C2 candidate count         : {r['c2_candidate_count']}")
        print(f"commuting pairs            : {r['commuting_pairs']}")
        print(f"anti-commuting pairs       : {r['anti_commuting_pairs']}")
    print(f"total targets              : {r['total_targets']}")
    print()

    print(f"direct hits                : {r['direct_hits']}      (1M,0R)")
    print(f"rot-only hits              : {r['rot_only_hits']}      (1M,1..dR)")
    if r["enable_c2_fallback"]:
        print(f"C2 fallback hits           : {r['c2_fallback_hits']}      (2M,0R)")
    print(f"one-measure hits           : {r['one_measure_hits']}")
    print(f"overall hits               : {r['overall_hits']}")
    print(f"misses                     : {r['misses']}")
    print()

    print(f"direct hit rate            : {r['direct_hit_rate_percent']:.6f}%")
    print(f"rot-only hit rate          : {r['rot_only_hit_rate_percent']:.6f}%")
    print(f"one-measure hit rate       : {r['one_measure_hit_rate_percent']:.6f}%")
    if r["enable_c2_fallback"]:
        print(f"C2 fallback hit rate       : {r['c2_fallback_hit_rate_percent']:.6f}%")
    print(f"overall hit rate           : {r['overall_hit_rate_percent']:.6f}%")
    print()

    print(f"direct by basis            : {r['direct_by_basis']}")
    print(f"rot by basis               : {r['rot_by_basis']}")
    if r["enable_c2_fallback"]:
        print(f"C2 by basis                : {r['c2_by_basis']}")
    print(f"rotation depth hist        : {r['rot_depth_hist']}")
    print()

    print("========== SAMPLE ROT HITS ==========")
    for item in r["sample_rot_hits"]:
        print(f"tail11       = {item['tail11']}")
        print(f"best depth   = {item['best_depth']}")
        for op in item["options"]:
            print(f"  basis      = {op['basis']}")
            print(f"  full       = {op['full_pauli']}")
            print(f"  rot_depth  = {op['rot_depth']}")
        print()

    if r["enable_c2_fallback"]:
        print("========== SAMPLE C2 FALLBACK HITS ==========")
        for item in r["sample_c2_hits"]:
            print(f"tail11       = {item['tail11']}")
            for op in item["options"]:
                print(f"  basis      = {op['basis']}")
                print(f"  full       = {op['full_pauli']}")
            print()

    print("========== NOTE ==========")
    print("This script uses lexicographic cost:")
    print("  1) minimize measurement count")
    print("  2) then minimize bounded rotation depth")
    print("So 1-measurement + rotations is preferred over 2-measurement C2.")


def print_target_report(rep):
    print("========== TARGET EXPLANATION ==========")
    print(f"target full  : {rep['target_full']}")
    print(f"target tail  : {rep['target_tail']}")
    print(f"max_rot      : {rep['max_rot']}")
    print()

    for Q in ["X", "Y", "Z"]:
        b = rep["per_basis"][Q]
        print(f"[basis {Q}]")
        print(f"candidate full    : {b['candidate_full']}")
        print(f"direct native     : {b['direct']}")
        print(f"C2 fallback       : {b['c2']}")
        if b["rot_path"] is None:
            print("rotation path      : none within bound")
        else:
            print(f"rotation depth     : {b['rot_path']['depth']}")
            print(f"source tail        : {b['rot_path']['source_tail']}")
            print(f"source full        : {b['rot_path']['source_full']}")
            print(f"axis sequence tail : {b['rot_path']['axis_sequence_tail']}")
        print()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=["full-x", "full-y", "full-z", "full-xz", "full-xyz"],
        help="family to analyze"
    )
    parser.add_argument("--csv", default="native.csv")
    parser.add_argument("--pivot-index", type=int, default=0)
    parser.add_argument("--max-rot", type=int, default=4)
    parser.add_argument("--disable-c2-fallback", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument(
        "--target-full",
        type=str,
        default=None,
        help="optional 12-qubit target for explicit path explanation"
    )
    args = parser.parse_args()

    enable_c2_fallback = not args.disable_c2_fallback

    if args.family is None and args.target_full is None:
        raise SystemExit("Provide at least --family or --target-full")

    if args.family is not None:
        result = analyze_family(
            csv_path=args.csv,
            family_name=args.family,
            pivot_index=args.pivot_index,
            max_rot=args.max_rot,
            enable_c2_fallback=enable_c2_fallback,
            sample_limit=args.sample_limit
        )
        print_report(result)

        out_file = f"{args.family}_rotsearch_r{args.max_rot}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved report to {out_file}")
        print()

    if args.target_full is not None:
        rep = explain_target(
            target_full=args.target_full,
            csv_path=args.csv,
            pivot_index=args.pivot_index,
            max_rot=args.max_rot,
            enable_c2_fallback=enable_c2_fallback
        )
        print_target_report(rep)

        out_file = f"target_{normalize_pauli_string(args.target_full)}_rotpath_r{args.max_rot}.json"
        with open(out_file, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"Saved target explanation to {out_file}")


if __name__ == "__main__":
    main()
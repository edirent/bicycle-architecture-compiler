import csv
import json
import itertools
from collections import defaultdict

PAULI_SET = set("IXYZ")

PAULI_MUL = {
    ('I', 'I'): 'I', ('I', 'X'): 'X', ('I', 'Y'): 'Y', ('I', 'Z'): 'Z',
    ('X', 'I'): 'X', ('X', 'X'): 'I', ('X', 'Y'): 'Z', ('X', 'Z'): 'Y',
    ('Y', 'I'): 'Y', ('Y', 'X'): 'Z', ('Y', 'Y'): 'I', ('Y', 'Z'): 'X',
    ('Z', 'I'): 'Z', ('Z', 'X'): 'Y', ('Z', 'Y'): 'X', ('Z', 'Z'): 'I',
}

def normalize_pauli_string(s: str) -> str:
    s = s.strip().upper()
    if not s:
        raise ValueError("empty Pauli string")
    bad = set(s) - PAULI_SET
    if bad:
        raise ValueError(f"invalid Pauli string {s}, bad chars = {bad}")
    return s

def load_native_csv(csv_path="native.csv"):
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

def is_commuting(p: str, q: str) -> bool:
    anti = 0
    for a, b in zip(p, q):
        if a != 'I' and b != 'I' and a != b:
            anti += 1
    return anti % 2 == 0

def pauli_mul(p: str, q: str) -> str:
    return ''.join(PAULI_MUL[(a, b)] for a, b in zip(p, q))

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

def make_full_pauli_from_Q_and_tail(Q: str, tail11: str, pivot_index: int = 0) -> str:
    if len(tail11) != 11:
        raise ValueError(f"tail length must be 11, got {len(tail11)}")
    if pivot_index == 0:
        return Q + tail11
    return tail11[:pivot_index] + Q + tail11[pivot_index:]

def enumerate_full_x_space():
    for bits in itertools.product(["I", "X"], repeat=11):
        yield "".join(bits)

def analyze_full_x_with_c2(csv_path="native.csv", pivot_index=0):
    native_map, total_rows = load_native_csv(csv_path)
    native_set = set(native_map.keys())

    c2_map, commute_pairs, anti_pairs = build_c2(native_map)
    c2_set = set(c2_map.keys())

    total_targets = 0
    direct_hits = 0
    c2_only_hits = 0
    combined_hits = 0
    misses = 0

    direct_by_basis = {"X": 0, "Y": 0, "Z": 0}
    c2_by_basis = {"X": 0, "Y": 0, "Z": 0}

    sample_direct = []
    sample_c2 = []
    sample_miss = []

    for tail11 in enumerate_full_x_space():
        total_targets += 1

        direct_bases = []
        direct_fulls = []
        c2_bases = []
        c2_fulls = []

        for Q in ["X", "Y", "Z"]:
            full_pauli = make_full_pauli_from_Q_and_tail(Q, tail11, pivot_index=pivot_index)

            if full_pauli in native_set:
                direct_bases.append(Q)
                direct_fulls.append(full_pauli)

            elif full_pauli in c2_set:
                c2_bases.append(Q)
                c2_fulls.append(full_pauli)

        if direct_bases:
            direct_hits += 1
            combined_hits += 1
            for Q in direct_bases:
                direct_by_basis[Q] += 1
            if len(sample_direct) < 20:
                sample_direct.append({
                    "tail11": tail11,
                    "hit_bases": direct_bases,
                    "full_paulis": direct_fulls,
                })
        elif c2_bases:
            c2_only_hits += 1
            combined_hits += 1
            for Q in c2_bases:
                c2_by_basis[Q] += 1
            if len(sample_c2) < 20:
                sample_c2.append({
                    "tail11": tail11,
                    "hit_bases": c2_bases,
                    "full_paulis": c2_fulls,
                    "witnesses": {fp: c2_map[fp][:2] for fp in c2_fulls},
                })
        else:
            misses += 1
            if len(sample_miss) < 20:
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
        "c2_count": len(c2_set),
        "commuting_pairs": commute_pairs,
        "anti_commuting_pairs": anti_pairs,
        "pivot_index": pivot_index,
        "target_space": "{I,X}^11",
        "total_targets": total_targets,
        "direct_hits": direct_hits,
        "c2_only_hits": c2_only_hits,
        "combined_hits": combined_hits,
        "misses": misses,
        "direct_hit_rate_percent": 100.0 * direct_hits / total_targets,
        "c2_only_hit_rate_percent": 100.0 * c2_only_hits / total_targets,
        "combined_hit_rate_percent": 100.0 * combined_hits / total_targets,
        "direct_by_basis": direct_by_basis,
        "c2_by_basis": c2_by_basis,
        "sample_direct_hits": sample_direct,
        "sample_c2_hits": sample_c2,
        "sample_misses": sample_miss,
    }

def main():
    result = analyze_full_x_with_c2(csv_path="native.csv", pivot_index=0)

    print("========== FULL-X DIRECT + C2 HIT TEST ==========")
    print(f"raw csv rows            : {result['raw_rows']}")
    print(f"unique native count     : {result['unique_native_count']}")
    print(f"C2 candidate count      : {result['c2_count']}")
    print(f"commuting pairs         : {result['commuting_pairs']}")
    print(f"anti-commuting pairs    : {result['anti_commuting_pairs']}")
    print(f"pivot index             : {result['pivot_index']}")
    print(f"target space            : {result['target_space']}")
    print(f"total targets           : {result['total_targets']}")
    print()
    print(f"direct hits             : {result['direct_hits']}")
    print(f"C2-only hits            : {result['c2_only_hits']}")
    print(f"combined hits           : {result['combined_hits']}")
    print(f"misses                  : {result['misses']}")
    print()
    print(f"direct hit rate         : {result['direct_hit_rate_percent']:.6f}%")
    print(f"C2-only hit rate        : {result['c2_only_hit_rate_percent']:.6f}%")
    print(f"combined hit rate       : {result['combined_hit_rate_percent']:.6f}%")
    print()
    print(f"direct by basis         : {result['direct_by_basis']}")
    print(f"C2 by basis             : {result['c2_by_basis']}")
    print()

    print("========== SAMPLE DIRECT HITS ==========")
    for item in result["sample_direct_hits"]:
        print(f"tail11     = {item['tail11']}")
        print(f"hit_bases  = {item['hit_bases']}")
        print(f"fulls      = {item['full_paulis']}")
        print()

    print("========== SAMPLE C2-ONLY HITS ==========")
    for item in result["sample_c2_hits"]:
        print(f"tail11     = {item['tail11']}")
        print(f"hit_bases  = {item['hit_bases']}")
        print(f"fulls      = {item['full_paulis']}")
        for fp, ws in item["witnesses"].items():
            if ws:
                w = ws[0]
                print(f"  witness for {fp}:")
                print(f"    left  = {w['left_pauli']}")
                print(f"    right = {w['right_pauli']}")
        print()

    print("========== SAMPLE MISSES ==========")
    for item in result["sample_misses"]:
        print(f"tail11     = {item['tail11']}")
        print(f"X⊗P_i      = {item['candidates']['X']}")
        print(f"Y⊗P_i      = {item['candidates']['Y']}")
        print(f"Z⊗P_i      = {item['candidates']['Z']}")
        print()

    with open("full_x_c2_hit_report.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Saved report to full_x_c2_hit_report.json")
    print("NOTE: C2 hit is an auxiliary algebraic metric, not the paper's native-hit definition.")


if __name__ == "__main__":
    main()
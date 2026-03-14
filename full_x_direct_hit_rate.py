import csv
import json
import itertools
from collections import defaultdict

PAULI_SET = set("IXYZ")


def normalize_pauli_string(s: str) -> str:
    s = s.strip().upper()
    if not s:
        raise ValueError("empty Pauli string")
    bad = set(s) - PAULI_SET
    if bad:
        raise ValueError(f"invalid Pauli string {s}, bad chars = {bad}")
    return s


def load_native_csv(csv_path="native.csv"):
    """
    读取 native.csv，返回:
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


def make_full_pauli_from_Q_and_tail(Q: str, tail11: str, pivot_index: int = 0) -> str:
    """
    把 Q⊗P_i 具体化成 12 位字符串。
    默认 pivot 在第 0 位，所以 full_pauli = Q + tail11。
    如果以后 pivot 不在第 0 位，可以改这里。
    """
    if len(tail11) != 11:
        raise ValueError(f"tail length must be 11, got {len(tail11)}")

    if pivot_index == 0:
        return Q + tail11
    else:
        # 更一般地插入 pivot 位
        return tail11[:pivot_index] + Q + tail11[pivot_index:]


def enumerate_full_x_space():
    """
    枚举所有 P_i ∈ {I,X}^11
    """
    for bits in itertools.product(["I", "X"], repeat=11):
        yield "".join(bits)


def analyze_full_x_direct_hits(csv_path="native.csv", pivot_index=0):
    native_map, total_rows = load_native_csv(csv_path)
    native_set = set(native_map.keys())

    total_targets = 0
    direct_hits = 0
    misses = 0

    hit_by_basis = {"X": 0, "Y": 0, "Z": 0}
    exact_hit_records = []
    miss_records = []

    for tail11 in enumerate_full_x_space():
        total_targets += 1

        hit_bases = []
        hit_fulls = []

        for Q in ["X", "Y", "Z"]:
            full_pauli = make_full_pauli_from_Q_and_tail(Q, tail11, pivot_index=pivot_index)
            if full_pauli in native_set:
                hit_bases.append(Q)
                hit_fulls.append(full_pauli)

        if hit_bases:
            direct_hits += 1
            for Q in hit_bases:
                hit_by_basis[Q] += 1

            exact_hit_records.append({
                "tail11": tail11,
                "hit_bases": hit_bases,
                "full_paulis": hit_fulls,
                "witnesses": {fp: native_map[fp] for fp in hit_fulls},
            })
        else:
            misses += 1
            if len(miss_records) < 20:
                miss_records.append({
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
        "target_space": "{I,X}^11",
        "total_targets": total_targets,
        "direct_hits": direct_hits,
        "misses": misses,
        "direct_hit_rate_percent": 100.0 * direct_hits / total_targets,
        "hit_by_basis": hit_by_basis,
        "sample_hits": exact_hit_records[:20],
        "sample_misses": miss_records,
    }


def main():
    result = analyze_full_x_direct_hits(csv_path="native.csv", pivot_index=0)

    print("========== FULL-X DIRECT NATIVE HIT TEST ==========")
    print(f"raw csv rows            : {result['raw_rows']}")
    print(f"unique native count     : {result['unique_native_count']}")
    print(f"pivot index             : {result['pivot_index']}")
    print(f"target space            : {result['target_space']}")
    print(f"total targets           : {result['total_targets']}")
    print(f"direct native hits      : {result['direct_hits']}")
    print(f"misses                  : {result['misses']}")
    print(f"direct hit rate         : {result['direct_hit_rate_percent']:.6f}%")
    print(f"hit count by basis Q    : {result['hit_by_basis']}")
    print()

    print("========== SAMPLE DIRECT HITS ==========")
    for item in result["sample_hits"]:
        print(f"tail11     = {item['tail11']}")
        print(f"hit_bases  = {item['hit_bases']}")
        print(f"fulls      = {item['full_paulis']}")
        print()

    print("========== SAMPLE MISSES ==========")
    for item in result["sample_misses"]:
        print(f"tail11     = {item['tail11']}")
        print(f"X⊗P_i      = {item['candidates']['X']}")
        print(f"Y⊗P_i      = {item['candidates']['Y']}")
        print(f"Z⊗P_i      = {item['candidates']['Z']}")
        print()

    with open("full_x_direct_hit_report.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Saved report to full_x_direct_hit_report.json")


if __name__ == "__main__":
    main()
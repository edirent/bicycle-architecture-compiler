import csv
import json
import argparse
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


def load_native_csv(csv_path: str):
    """
    读取 native.csv
    返回:
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


def pauli_to_symplectic(pauli: str):
    """
    把 n-qubit Pauli 串转成 (x|z) in F_2^(2n)
    I -> (0,0), X -> (1,0), Z -> (0,1), Y -> (1,1)
    """
    x = []
    z = []
    for c in pauli:
        if c == "I":
            x.append(0)
            z.append(0)
        elif c == "X":
            x.append(1)
            z.append(0)
        elif c == "Z":
            x.append(0)
            z.append(1)
        elif c == "Y":
            x.append(1)
            z.append(1)
        else:
            raise ValueError(f"bad Pauli char: {c}")
    return x + z


def symplectic_inner(v1, v2):
    """
    辛内积:
      <(x|z),(x'|z')> = x·z' + z·x' mod 2
    """
    n2 = len(v1)
    if len(v2) != n2 or n2 % 2 != 0:
        raise ValueError("bad symplectic vector lengths")

    n = n2 // 2
    x1, z1 = v1[:n], v1[n:]
    x2, z2 = v2[:n], v2[n:]

    s = 0
    for a, b in zip(x1, z2):
        s ^= (a & b)
    for a, b in zip(z1, x2):
        s ^= (a & b)
    return s


def symplectic_commute(pauli1: str, pauli2: str) -> bool:
    """
    通过辛内积判定两个 Pauli 是否对易
    """
    v1 = pauli_to_symplectic(pauli1)
    v2 = pauli_to_symplectic(pauli2)
    return symplectic_inner(v1, v2) == 0


def drop_pivot(pauli: str, pivot_index: int) -> str:
    """
    删掉第 pivot_index 位，得到 11-qubit tail
    """
    return pauli[:pivot_index] + pauli[pivot_index + 1:]


def build_bucket(native_map, pivot_index: int, basis: str):
    """
    固定 pivot qubit 和 pivot basis Q，构造:
      N_Q = { native P : P[pivot_index] == basis }
    返回列表，每项包括:
      full_pauli, tail, witness_count, witnesses
    """
    if basis not in {"X", "Y", "Z"}:
        raise ValueError("basis must be one of X, Y, Z")
    if not (0 <= pivot_index < 12):
        raise ValueError("pivot_index must be in [0, 11]")

    bucket = []
    for full_pauli, witnesses in native_map.items():
        if full_pauli[pivot_index] == basis:
            tail = drop_pivot(full_pauli, pivot_index)
            bucket.append({
                "full_pauli": full_pauli,
                "tail": tail,
                "witness_count": len(witnesses),
                "witnesses": witnesses,
            })

    bucket.sort(key=lambda x: x["full_pauli"])
    return bucket


def build_commute_graph(bucket):
    """
    在固定 pivot basis 的 bucket 上，
    对 tails 两两做 symplectic commute check
    """
    n = len(bucket)
    commute_edges = []
    anti_edges = []

    adjacency = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            a = bucket[i]
            b = bucket[j]
            if symplectic_commute(a["tail"], b["tail"]):
                commute_edges.append((i, j))
                adjacency[i].append(j)
                adjacency[j].append(i)
            else:
                anti_edges.append((i, j))

    return commute_edges, anti_edges, adjacency


def summarize_bucket(bucket):
    """
    统计 bucket 中 tail 的首字符分布等
    """
    tail_first_bucket = {"I": 0, "X": 0, "Y": 0, "Z": 0}
    for item in bucket:
        tail_first_bucket[item["tail"][0]] += 1

    return {
        "bucket_size": len(bucket),
        "tail_first_char_bucket": tail_first_bucket,
    }


def sample_pairs(bucket, edges, k=5):
    out = []
    for idx, (i, j) in enumerate(edges[:k]):
        out.append({
            "left_full": bucket[i]["full_pauli"],
            "left_tail": bucket[i]["tail"],
            "right_full": bucket[j]["full_pauli"],
            "right_tail": bucket[j]["tail"],
        })
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build same-pivot-basis commute graph from native.csv."
    )
    parser.add_argument("--csv", default="native.csv", help="path to native.csv")
    parser.add_argument("--pivot-index", type=int, default=0, help="pivot qubit index in 12-char string")
    parser.add_argument("--basis", choices=["X", "Y", "Z"], required=True, help="fixed pivot basis")
    parser.add_argument("--save-json", default=None, help="optional path to save JSON report")
    args = parser.parse_args()

    print("========== STEP 1: Load native.csv ==========")
    native_map, total_rows = load_native_csv(args.csv)
    print(f"raw csv rows         : {total_rows}")
    print(f"unique native paulis : {len(native_map)}")
    print()

    print("========== STEP 2: Build N_Q ==========")
    bucket = build_bucket(native_map, args.pivot_index, args.basis)
    bucket_summary = summarize_bucket(bucket)

    print(f"pivot index          : {args.pivot_index}")
    print(f"pivot basis Q        : {args.basis}")
    print(f"|N_Q|                : {bucket_summary['bucket_size']}")
    print(f"tail first-char dist : {bucket_summary['tail_first_char_bucket']}")
    print()

    print("========== STEP 3: Build commute graph on tails ==========")
    commute_edges, anti_edges, adjacency = build_commute_graph(bucket)

    print(f"commuting pairs      : {len(commute_edges)}")
    print(f"anti-commuting pairs : {len(anti_edges)}")
    print()

    print("========== SAMPLE COMMUTING PAIRS ==========")
    for s in sample_pairs(bucket, commute_edges, k=5):
        print(f"full : {s['left_full']}   tail : {s['left_tail']}")
        print(f"full : {s['right_full']}   tail : {s['right_tail']}")
        print("-> tails commute in the fixed pivot-basis bucket")
        print()

    print("========== SAMPLE ANTI-COMMUTING PAIRS ==========")
    for s in sample_pairs(bucket, anti_edges, k=5):
        print(f"full : {s['left_full']}   tail : {s['left_tail']}")
        print(f"full : {s['right_full']}   tail : {s['right_tail']}")
        print("-> tails anti-commute in the fixed pivot-basis bucket")
        print()

    report = {
        "csv": args.csv,
        "total_rows": total_rows,
        "unique_native_count": len(native_map),
        "pivot_index": args.pivot_index,
        "pivot_basis": args.basis,
        "bucket_summary": bucket_summary,
        "commuting_pairs": len(commute_edges),
        "anti_commuting_pairs": len(anti_edges),
        "sample_commuting_pairs": sample_pairs(bucket, commute_edges, k=10),
        "sample_anti_commuting_pairs": sample_pairs(bucket, anti_edges, k=10),
        "bucket": [
            {
                "full_pauli": item["full_pauli"],
                "tail": item["tail"],
                "witness_count": item["witness_count"],
                "witnesses": item["witnesses"],
                "degree_in_commute_graph": len(adjacency[i]),
            }
            for i, item in enumerate(bucket)
        ],
    }

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved JSON report to {args.save_json}")


if __name__ == "__main__":
    main()
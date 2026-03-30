from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from .examples import build_demo_joint_table
from .io import (
    DEFAULT_JOINT_TABLE_PATH,
    DEFAULT_LOCAL_TABLE_PATH,
    RandomCorpus,
    load_joint_table,
    load_local_table,
    load_random_corpus,
    random_corpus_path,
    save_joint_table,
    save_local_table,
    save_random_corpus,
)
from .state import N_QUBITS, Tail
from .tables import (
    JointTable,
    LocalTable,
    NativeEntry,
    SCOPE_INTRA_BLOCK_NATIVE,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GROSS_NATIVE_SOURCE_CSV = REPO_ROOT / "results" / "native_11q_real.csv"
GROSS_LOCAL_FRAME_ID = "gross_native_11q_pivot0"
SEED_DIR = Path(__file__).resolve().parent / "seeds"
CROSS_LOGICAL_NATIVE_SEED_PATH = SEED_DIR / "cross_logical_native_seed.json"

GROSS_NATIVE_SOURCE_DESCRIPTION = (
    "results/native_11q_real.csv (export source: "
    "crates/bicycle_cliffords/src/bin/export_native_11q.rs using NativeMeasurement::all + GROSS_MEASUREMENT)"
)
RANDOM_RULE_DESCRIPTION = (
    "uniform per-qubit Pauli sampling from {I,X,Y,Z} with all-identity rejection; "
    "rule aligned with crates/bicycle_benchmark/src/random.rs and reused in "
    "crates/bicycle_compiler/src/random_check.rs"
)


def _decode_tail_bits(tail_bits: int, n_qubits: int = N_QUBITS) -> Tail:
    if n_qubits != 11:
        raise ValueError("this exporter currently assumes 11 data qubits")

    x_bits = tail_bits & ((1 << n_qubits) - 1)
    z_bits = (tail_bits >> n_qubits) & ((1 << n_qubits) - 1)

    paulis = []
    for idx in range(n_qubits):
        x = (x_bits >> idx) & 1
        z = (z_bits >> idx) & 1
        if x == 0 and z == 0:
            paulis.append("I")
        elif x == 1 and z == 0:
            paulis.append("X")
        elif x == 0 and z == 1:
            paulis.append("Z")
        else:
            paulis.append("Y")

    return Tail(tuple(paulis))


def _tail_support_labels(tail: Tail) -> Tuple[str, ...]:
    return tuple(f"data_q{idx}" for idx, p in enumerate(tail.paulis, start=1) if p != "I")


def _load_seed_entries(seed_path: Path, frame_id: str) -> List[NativeEntry]:
    if not seed_path.exists():
        return []

    with seed_path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    rows = payload.get("entries", [])
    out: List[NativeEntry] = []
    for row in rows:
        row_frame = row.get("frame_id", frame_id)
        out.append(
            NativeEntry(
                native_id=row["native_id"],
                frame_id=row_frame,
                head=row["head"],
                tail=Tail.from_str(row["tail"]),
                scope=row.get("scope", SCOPE_INTRA_BLOCK_NATIVE),
                support=tuple(row.get("support", [])),
                metadata=row.get("metadata", {}),
            )
        )
    return out


def _merge_entries(base: List[NativeEntry], extras: List[NativeEntry]) -> List[NativeEntry]:
    by_id: Dict[str, NativeEntry] = {entry.native_id: entry for entry in base}
    for entry in extras:
        if entry.native_id in by_id:
            raise ValueError(f"duplicate native_id while merging local table assets: {entry.native_id}")
        by_id[entry.native_id] = entry
    return list(by_id.values())


def build_local_table_from_gross_csv(
    csv_path: Path = GROSS_NATIVE_SOURCE_CSV,
    seed_path: Path = CROSS_LOGICAL_NATIVE_SEED_PATH,
) -> LocalTable:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"gross native source CSV not found: {csv_path}. "
            "Generate it via `cargo run -p bicycle_cliffords --bin export_native_11q -- --output results/native_11q_real.csv`."
        )

    entries = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index = int(row["index"])
            head = row["head"]
            tail_bits = int(row["tail_bits"])
            tail = _decode_tail_bits(tail_bits)
            entries.append(
                NativeEntry(
                    native_id=f"gross_native_{index}",
                    frame_id=GROSS_LOCAL_FRAME_ID,
                    head=head,
                    tail=tail,
                    scope=SCOPE_INTRA_BLOCK_NATIVE,
                    support=_tail_support_labels(tail),
                    metadata={
                        "source": "results/native_11q_real.csv",
                        "source_index": str(index),
                        "source_kind": "gross_native_csv",
                    },
                )
            )

    extras = _load_seed_entries(seed_path, frame_id=GROSS_LOCAL_FRAME_ID)
    merged = _merge_entries(entries, extras)
    return LocalTable(frame_id=GROSS_LOCAL_FRAME_ID, entries=merged)


def build_joint_table_asset() -> JointTable:
    return build_demo_joint_table()


def build_random_corpus(
    seed: int,
    count: int,
    n_qubits: int = N_QUBITS,
    max_weight_filter: int | None = None,
    allowed_targets: Set[str] | None = None,
) -> RandomCorpus:
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    targets = []

    while len(targets) < count:
        paulis = [rng.choice(("I", "X", "Y", "Z")) for _ in range(n_qubits)]
        target = "".join(paulis)
        if all(p == "I" for p in paulis):
            continue
        if max_weight_filter is not None:
            weight = sum(p != "I" for p in paulis)
            if weight > max_weight_filter:
                continue
        if allowed_targets is not None and target not in allowed_targets:
            continue
        targets.append(target)

    return RandomCorpus(
        seed=seed,
        n_qubits=n_qubits,
        sampling_rule=RANDOM_RULE_DESCRIPTION,
        targets=targets,
        source="crates/bicycle_benchmark/src/random.rs + crates/bicycle_compiler/src/random_check.rs",
        count=count,
        max_weight_filter=max_weight_filter,
    )


def refresh_local_table_asset(path: Path = DEFAULT_LOCAL_TABLE_PATH) -> LocalTable:
    local_table = build_local_table_from_gross_csv()
    save_local_table(local_table, path)
    return local_table


def refresh_joint_table_asset(path: Path = DEFAULT_JOINT_TABLE_PATH) -> JointTable:
    joint_table = build_joint_table_asset()
    save_joint_table(joint_table, path)
    return joint_table


def refresh_random_corpus_asset(
    *,
    seed: int,
    count: int,
    path: Path | None = None,
    max_weight_filter: int | None = None,
    allowed_targets: Set[str] | None = None,
) -> RandomCorpus:
    out_path = path or random_corpus_path(seed)
    corpus = build_random_corpus(
        seed=seed,
        count=count,
        max_weight_filter=max_weight_filter,
        allowed_targets=allowed_targets,
    )
    save_random_corpus(corpus, out_path)
    return corpus


def ensure_cached_assets(
    *,
    seed: int = 7,
    count: int = 20,
    max_weight_filter: int | None = 1,
) -> Tuple[LocalTable, JointTable, RandomCorpus]:
    if DEFAULT_LOCAL_TABLE_PATH.exists():
        local = load_local_table(DEFAULT_LOCAL_TABLE_PATH)
    else:
        local = refresh_local_table_asset(DEFAULT_LOCAL_TABLE_PATH)

    if DEFAULT_JOINT_TABLE_PATH.exists():
        joint = load_joint_table(DEFAULT_JOINT_TABLE_PATH)
    else:
        joint = refresh_joint_table_asset(DEFAULT_JOINT_TABLE_PATH)

    corpus_path = random_corpus_path(seed)
    if corpus_path.exists():
        corpus = load_random_corpus(corpus_path)
    else:
        corpus = refresh_random_corpus_asset(
            seed=seed,
            count=count,
            path=corpus_path,
            max_weight_filter=max_weight_filter,
        )

    return local, joint, corpus


def _print_local_summary(local_table: LocalTable, path: Path, mode: str) -> None:
    print(f"[{mode}] local table: {path} entries={len(local_table.entries)} frame={local_table.frame_id}")


def _print_joint_summary(joint_table: JointTable, path: Path, mode: str) -> None:
    print(
        f"[{mode}] joint table: {path} p2_rules={len(joint_table.p2_rules)} p3_rules={len(joint_table.p3_rules)}"
    )


def _print_corpus_summary(corpus: RandomCorpus, path: Path, mode: str) -> None:
    print(
        f"[{mode}] random corpus: {path} seed={corpus.seed} count={corpus.count} "
        f"max_weight_filter={corpus.max_weight_filter}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/cache two-table prototype assets")
    parser.add_argument("--refresh-all", action="store_true")
    parser.add_argument("--refresh-local", action="store_true")
    parser.add_argument("--refresh-joint", action="store_true")
    parser.add_argument("--refresh-random", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-weight", type=int, default=None)
    args = parser.parse_args()

    refresh_local = args.refresh_all or args.refresh_local
    refresh_joint = args.refresh_all or args.refresh_joint
    refresh_random = args.refresh_all or args.refresh_random

    if refresh_local:
        local = refresh_local_table_asset(DEFAULT_LOCAL_TABLE_PATH)
        _print_local_summary(local, DEFAULT_LOCAL_TABLE_PATH, "refreshed")
    elif DEFAULT_LOCAL_TABLE_PATH.exists():
        local = load_local_table(DEFAULT_LOCAL_TABLE_PATH)
        _print_local_summary(local, DEFAULT_LOCAL_TABLE_PATH, "cached")
    else:
        local = refresh_local_table_asset(DEFAULT_LOCAL_TABLE_PATH)
        _print_local_summary(local, DEFAULT_LOCAL_TABLE_PATH, "generated")

    if refresh_joint:
        joint = refresh_joint_table_asset(DEFAULT_JOINT_TABLE_PATH)
        _print_joint_summary(joint, DEFAULT_JOINT_TABLE_PATH, "refreshed")
    elif DEFAULT_JOINT_TABLE_PATH.exists():
        joint = load_joint_table(DEFAULT_JOINT_TABLE_PATH)
        _print_joint_summary(joint, DEFAULT_JOINT_TABLE_PATH, "cached")
    else:
        joint = refresh_joint_table_asset(DEFAULT_JOINT_TABLE_PATH)
        _print_joint_summary(joint, DEFAULT_JOINT_TABLE_PATH, "generated")

    corpus_path = random_corpus_path(args.seed)
    if refresh_random:
        corpus = refresh_random_corpus_asset(
            seed=args.seed,
            count=args.count,
            path=corpus_path,
            max_weight_filter=args.max_weight,
        )
        _print_corpus_summary(corpus, corpus_path, "refreshed")
    elif corpus_path.exists():
        corpus = load_random_corpus(corpus_path)
        _print_corpus_summary(corpus, corpus_path, "cached")
    else:
        corpus = refresh_random_corpus_asset(
            seed=args.seed,
            count=args.count,
            path=corpus_path,
            max_weight_filter=args.max_weight,
        )
        _print_corpus_summary(corpus, corpus_path, "generated")

    print(f"local source: {GROSS_NATIVE_SOURCE_DESCRIPTION}")
    print(f"local seed source: {CROSS_LOGICAL_NATIVE_SEED_PATH}")
    print(f"random source rule: {RANDOM_RULE_DESCRIPTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

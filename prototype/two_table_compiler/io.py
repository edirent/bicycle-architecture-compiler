from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .state import HiddenClass, Tail
from .tables import (
    JointTable,
    LocalTable,
    NativeEntry,
    P2_ALLOWED_SOURCE_SCOPES,
    P2_RELATION_ANY,
    P2_RELATION_COMMUTE,
    P2_TAIL_OP_PRODUCT,
    P2Rule,
    P3Rule,
    SCOPE_INTRA_BLOCK_NATIVE,
)

SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_LOCAL_TABLE_PATH = GENERATED_DIR / "gross_local_table.json"
DEFAULT_JOINT_TABLE_PATH = GENERATED_DIR / "joint_table.json"


def random_corpus_path(seed: int) -> Path:
    return GENERATED_DIR / f"random_corpus_seed{seed}.json"


@dataclass(frozen=True)
class RandomCorpus:
    seed: int
    n_qubits: int
    sampling_rule: str
    targets: List[str]
    source: str
    count: int
    max_weight_filter: int | None = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metadata_to_json(metadata: tuple[tuple[str, str], ...]) -> Dict[str, str]:
    return {key: value for key, value in metadata}


def save_local_table(local_table: LocalTable, path: Path | str) -> None:
    out_path = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frame_id": local_table.frame_id,
        "entry_count": len(local_table.entries),
        "entries": [
            {
                "native_id": entry.native_id,
                "frame_id": entry.frame_id,
                "head": entry.head,
                "tail": entry.tail.compact(),
                "scope": entry.scope,
                "support": list(entry.support),
                "metadata": _metadata_to_json(entry.metadata),
            }
            for entry in local_table.entries
        ],
    }
    _dump_json(out_path, payload)


def load_local_table(path: Path | str) -> LocalTable:
    src_path = Path(path)
    payload = _load_json(src_path)
    entries = [
        NativeEntry(
            native_id=row["native_id"],
            frame_id=row["frame_id"],
            head=row["head"],
            tail=Tail.from_str(row["tail"]),
            scope=row.get("scope", SCOPE_INTRA_BLOCK_NATIVE),
            support=tuple(row.get("support", [])),
            metadata=row.get("metadata", {}),
        )
        for row in payload["entries"]
    ]
    return LocalTable(frame_id=payload["frame_id"], entries=entries)


def save_joint_table(joint_table: JointTable, path: Path | str) -> None:
    out_path = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "p2_rules": [
            {
                "name": rule.name,
                "lhs_head": rule.lhs_head,
                "rhs_head": rule.rhs_head,
                "require_same_head": rule.require_same_head,
                "require_nonidentity_heads": rule.require_nonidentity_heads,
                "required_relation": rule.required_relation,
                "out_hidden": rule.out_hidden.name,
                "tail_op": rule.tail_op,
                "total_cost": rule.total_cost,
                "allowed_input_scopes": list(rule.allowed_input_scopes),
                "template": rule.template,
            }
            for rule in joint_table.p2_rules
        ],
        "p3_rules": [
            {
                "name": rule.name,
                "cur_hidden": rule.cur_hidden.name,
                "axis_head": rule.axis_head,
                "anticommute_required": rule.anticommute_required,
                "next_hidden": rule.next_hidden.name,
                "delta_cost": rule.delta_cost,
                "template": rule.template,
            }
            for rule in joint_table.p3_rules
        ],
    }
    _dump_json(out_path, payload)


def load_joint_table(path: Path | str) -> JointTable:
    src_path = Path(path)
    payload = _load_json(src_path)

    def _p2_rule_from_row(row: Dict[str, Any]) -> P2Rule:
        # Backward compatibility with schema_version=1 style rows.
        if "total_cost" not in row:
            out_hidden_name = row["out_hidden"]
            same_head = bool(row.get("required_same_nonidentity_head", True))
            lhs_head = out_hidden_name if same_head else None
            rhs_head = out_hidden_name if same_head else None
            relation = P2_RELATION_COMMUTE if row.get("commute_required", True) else P2_RELATION_ANY
            return P2Rule(
                name=row["name"],
                lhs_head=lhs_head,
                rhs_head=rhs_head,
                require_same_head=same_head,
                require_nonidentity_heads=same_head,
                required_relation=relation,
                out_hidden=HiddenClass(out_hidden_name),
                tail_op=P2_TAIL_OP_PRODUCT,
                total_cost=row["delta_cost"],
                allowed_input_scopes=P2_ALLOWED_SOURCE_SCOPES,
                template=row["template"],
            )

        return P2Rule(
            name=row["name"],
            lhs_head=row.get("lhs_head"),
            rhs_head=row.get("rhs_head"),
            require_same_head=row.get("require_same_head", False),
            require_nonidentity_heads=row.get("require_nonidentity_heads", False),
            required_relation=row.get("required_relation", P2_RELATION_ANY),
            out_hidden=HiddenClass(row["out_hidden"]),
            tail_op=row.get("tail_op", P2_TAIL_OP_PRODUCT),
            total_cost=row["total_cost"],
            allowed_input_scopes=tuple(row.get("allowed_input_scopes", P2_ALLOWED_SOURCE_SCOPES)),
            template=row["template"],
        )

    p2_rules = [
        _p2_rule_from_row(row)
        for row in payload["p2_rules"]
    ]
    p3_rules = [
        P3Rule(
            name=row["name"],
            cur_hidden=HiddenClass(row["cur_hidden"]),
            axis_head=row["axis_head"],
            anticommute_required=row["anticommute_required"],
            next_hidden=HiddenClass(row["next_hidden"]),
            delta_cost=row["delta_cost"],
            template=row["template"],
        )
        for row in payload["p3_rules"]
    ]
    return JointTable(p2_rules=p2_rules, p3_rules=p3_rules)


def save_random_corpus(corpus: RandomCorpus, path: Path | str) -> None:
    out_path = Path(path)
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": corpus.seed,
        "n_qubits": corpus.n_qubits,
        "sampling_rule": corpus.sampling_rule,
        "source": corpus.source,
        "count": corpus.count,
        "targets": corpus.targets,
    }
    if corpus.max_weight_filter is not None:
        payload["max_weight_filter"] = corpus.max_weight_filter
    _dump_json(out_path, payload)


def load_random_corpus(path: Path | str) -> RandomCorpus:
    src_path = Path(path)
    payload = _load_json(src_path)
    return RandomCorpus(
        seed=payload["seed"],
        n_qubits=payload["n_qubits"],
        sampling_rule=payload["sampling_rule"],
        source=payload["source"],
        count=payload["count"],
        targets=list(payload["targets"]),
        max_weight_filter=payload.get("max_weight_filter"),
    )

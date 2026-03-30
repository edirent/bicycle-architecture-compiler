from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .state import Tail
from .tables import LocalTable, NativeEntry

_LOGICAL_TOKEN = re.compile(r"^([XYZ])(\d+)(P'?)$")


@dataclass(frozen=True)
class TargetClassification:
    input_target: str
    canonical_tail: Tail | None
    canonical_logical: str | None
    matched_native_entries: Tuple[NativeEntry, ...]
    is_logical_notation: bool
    frame_unreachable: bool
    reason: str | None

    def matched_native_ids(self) -> Tuple[str, ...]:
        return tuple(entry.native_id for entry in self.matched_native_entries)


def _normalize_target_text(target: str) -> str:
    normalized = target.strip().upper()
    normalized = normalized.replace("’", "'")
    normalized = normalized.replace("PRIME", "'")
    normalized = normalized.replace("×", " ")
    normalized = normalized.replace("⊗", " ")
    normalized = normalized.replace("·", " ")
    normalized = normalized.replace("*", " ")
    return " ".join(token for token in normalized.split() if token)


def _parse_logical_tokens(target: str) -> Tuple[Tuple[str, int, str], ...]:
    normalized = _normalize_target_text(target)
    if not normalized:
        raise ValueError("empty target")
    out: List[Tuple[str, int, str]] = []
    for token in normalized.split():
        m = _LOGICAL_TOKEN.match(token)
        if not m:
            raise ValueError(f"invalid logical token: {token!r}")
        pauli = m.group(1)
        logical_index = int(m.group(2))
        sheet = m.group(3)
        if logical_index <= 0:
            raise ValueError(f"logical index must be positive: {token!r}")
        out.append((pauli, logical_index, sheet))
    if not out:
        raise ValueError("no logical tokens parsed")
    return tuple(out)


def canonicalize_logical_expression(target: str) -> str:
    tokens = _parse_logical_tokens(target)
    ordered = sorted(tokens, key=lambda item: (item[1], 0 if item[2] == "P" else 1, item[0]))
    return "⊗".join(f"{pauli}{index}{sheet}" for pauli, index, sheet in ordered)


def _logical_aliases(entry: NativeEntry) -> Sequence[str]:
    md = dict(entry.metadata)
    out: List[str] = []
    canonical = md.get("logical_canonical")
    if canonical:
        out.append(canonical)
    aliases = md.get("logical_aliases")
    if aliases:
        out.extend([alias.strip() for alias in aliases.split(";") if alias.strip()])
    return out


def _lookup_logical_family(canonical: str, local_table: LocalTable) -> Tuple[NativeEntry, ...]:
    hits: List[NativeEntry] = []
    for entry in local_table.entries:
        for alias in _logical_aliases(entry):
            try:
                alias_canonical = canonicalize_logical_expression(alias)
            except ValueError:
                continue
            if alias_canonical == canonical:
                hits.append(entry)
                break
    return tuple(hits)


def classify_target(target: Tail | str, local_table: LocalTable) -> TargetClassification:
    if isinstance(target, Tail):
        hits = tuple(local_table.native_family_lookup(target))
        return TargetClassification(
            input_target=target.compact(),
            canonical_tail=target,
            canonical_logical=None,
            matched_native_entries=hits,
            is_logical_notation=False,
            frame_unreachable=False,
            reason=None,
        )

    raw = target.strip()
    if not raw:
        raise ValueError("target must be non-empty")

    try:
        tail = Tail.from_str(raw)
    except ValueError:
        tail = None

    if tail is not None:
        hits = tuple(local_table.native_family_lookup(tail))
        return TargetClassification(
            input_target=raw,
            canonical_tail=tail,
            canonical_logical=None,
            matched_native_entries=hits,
            is_logical_notation=False,
            frame_unreachable=False,
            reason=None,
        )

    canonical = canonicalize_logical_expression(raw)
    logical_hits = _lookup_logical_family(canonical, local_table)
    if logical_hits:
        hit_tails = {entry.tail for entry in logical_hits}
        if len(hit_tails) != 1:
            raise ValueError(
                f"logical family {canonical!r} maps to multiple tails in frame {local_table.frame_id}"
            )
        canonical_tail = next(iter(hit_tails))
        return TargetClassification(
            input_target=raw,
            canonical_tail=canonical_tail,
            canonical_logical=canonical,
            matched_native_entries=logical_hits,
            is_logical_notation=True,
            frame_unreachable=False,
            reason=None,
        )

    return TargetClassification(
        input_target=raw,
        canonical_tail=None,
        canonical_logical=canonical,
        matched_native_entries=(),
        is_logical_notation=True,
        frame_unreachable=True,
        reason=(
            f"frame {local_table.frame_id} has no native family for logical target "
            f"{canonical}; no automatic Bell fallback"
        ),
    )

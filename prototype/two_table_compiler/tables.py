from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .state import HEADS, HiddenClass, NON_IDENTITY_HEADS, Tail

SCOPE_INTRA_BLOCK_NATIVE = "intra_block_native"
SCOPE_CROSS_LOGICAL_BLOCK_NATIVE = "cross_logical_block_native"
SCOPE_INTER_MODULE_BELL = "inter_module_bell"
SCOPE_DERIVED_COMPOSITION = "derived_composition"
SCOPE_CLOSURE_TRANSITION = "closure_transition"
VALID_NATIVE_SCOPES = (
    SCOPE_INTRA_BLOCK_NATIVE,
    SCOPE_CROSS_LOGICAL_BLOCK_NATIVE,
    SCOPE_INTER_MODULE_BELL,
)
P2_ALLOWED_SOURCE_SCOPES = (
    SCOPE_INTRA_BLOCK_NATIVE,
    SCOPE_CROSS_LOGICAL_BLOCK_NATIVE,
)
P2_RELATION_COMMUTE = "commute"
P2_RELATION_ANTICOMMUTE = "anticommute"
P2_RELATION_ANY = "any"
VALID_P2_RELATIONS = (
    P2_RELATION_COMMUTE,
    P2_RELATION_ANTICOMMUTE,
    P2_RELATION_ANY,
)
P2_TAIL_OP_PRODUCT = "tail_product"
VALID_P2_TAIL_OPS = (P2_TAIL_OP_PRODUCT,)


@dataclass(frozen=True)
class NativeEntry:
    native_id: str
    frame_id: str
    head: str
    tail: Tail
    scope: str = SCOPE_INTRA_BLOCK_NATIVE
    support: Tuple[str, ...] = ()
    metadata: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        normalized_head = self.head.strip().upper()
        if normalized_head not in HEADS:
            raise ValueError(f"invalid native head: {self.head!r}")
        normalized_scope = self.scope.strip().lower()
        if normalized_scope not in VALID_NATIVE_SCOPES:
            raise ValueError(f"invalid native scope: {self.scope!r}")
        normalized_support = tuple(str(item) for item in self.support)
        normalized_metadata = _normalize_metadata(self.metadata)
        object.__setattr__(self, "head", normalized_head)
        object.__setattr__(self, "scope", normalized_scope)
        object.__setattr__(self, "support", normalized_support)
        object.__setattr__(self, "metadata", normalized_metadata)
        if not self.native_id:
            raise ValueError("native_id must be non-empty")
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")


@dataclass
class LocalTable:
    frame_id: str
    entries: List[NativeEntry]

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        for entry in self.entries:
            if entry.frame_id != self.frame_id:
                raise ValueError(
                    f"native entry {entry.native_id!r} has frame_id={entry.frame_id!r}, "
                    f"expected {self.frame_id!r}"
                )

    def by_head(self) -> Dict[str, List[NativeEntry]]:
        grouped: Dict[str, List[NativeEntry]] = defaultdict(list)
        for entry in self.entries:
            grouped[entry.head].append(entry)
        return dict(grouped)

    def native_axes(self) -> List[NativeEntry]:
        return list(self.entries)

    def nonidentity_entries(self) -> List[NativeEntry]:
        return [entry for entry in self.entries if entry.head in NON_IDENTITY_HEADS]

    def by_tail(self) -> Dict[Tail, List[NativeEntry]]:
        grouped: Dict[Tail, List[NativeEntry]] = defaultdict(list)
        for entry in self.entries:
            grouped[entry.tail].append(entry)
        return dict(grouped)

    def native_family_lookup(self, tail: Tail) -> List[NativeEntry]:
        return list(self.by_tail().get(tail, []))


@dataclass(frozen=True)
class P2Rule:
    name: str
    lhs_head: str | None
    rhs_head: str | None
    require_same_head: bool
    require_nonidentity_heads: bool
    required_relation: str
    out_hidden: HiddenClass
    tail_op: str
    total_cost: int
    allowed_input_scopes: Tuple[str, ...]
    template: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("P2 rule name must be non-empty")
        normalized_lhs = None if self.lhs_head is None else self.lhs_head.strip().upper()
        normalized_rhs = None if self.rhs_head is None else self.rhs_head.strip().upper()
        if normalized_lhs is not None and normalized_lhs not in HEADS:
            raise ValueError(f"invalid P2 lhs_head: {self.lhs_head!r}")
        if normalized_rhs is not None and normalized_rhs not in HEADS:
            raise ValueError(f"invalid P2 rhs_head: {self.rhs_head!r}")
        normalized_relation = self.required_relation.strip().lower()
        if normalized_relation not in VALID_P2_RELATIONS:
            raise ValueError(f"invalid P2 required_relation: {self.required_relation!r}")
        normalized_tail_op = self.tail_op.strip().lower()
        if normalized_tail_op not in VALID_P2_TAIL_OPS:
            raise ValueError(f"invalid P2 tail_op: {self.tail_op!r}")
        if self.total_cost <= 0:
            raise ValueError("P2 total_cost must be positive")
        normalized_scopes = tuple(sorted({scope.strip().lower() for scope in self.allowed_input_scopes}))
        if not normalized_scopes:
            raise ValueError("P2 allowed_input_scopes must be non-empty")
        invalid_scopes = [scope for scope in normalized_scopes if scope not in P2_ALLOWED_SOURCE_SCOPES]
        if invalid_scopes:
            raise ValueError(
                f"invalid P2 allowed_input_scopes={invalid_scopes}; "
                f"P2 supports only {P2_ALLOWED_SOURCE_SCOPES}"
            )
        object.__setattr__(self, "lhs_head", normalized_lhs)
        object.__setattr__(self, "rhs_head", normalized_rhs)
        object.__setattr__(self, "required_relation", normalized_relation)
        object.__setattr__(self, "tail_op", normalized_tail_op)
        object.__setattr__(self, "allowed_input_scopes", normalized_scopes)


@dataclass(frozen=True)
class P3Rule:
    name: str
    cur_hidden: HiddenClass
    axis_head: str
    anticommute_required: bool
    next_hidden: HiddenClass
    delta_cost: int
    template: str

    def __post_init__(self) -> None:
        normalized_axis_head = self.axis_head.strip().upper()
        if normalized_axis_head not in HEADS:
            raise ValueError(f"invalid axis head for P3: {self.axis_head!r}")
        if self.delta_cost <= 0:
            raise ValueError("P3 delta_cost must be positive")
        if not self.name:
            raise ValueError("P3 rule name must be non-empty")
        object.__setattr__(self, "axis_head", normalized_axis_head)


@dataclass
class JointTable:
    p2_rules: List[P2Rule]
    p3_rules: List[P3Rule]

    def __post_init__(self) -> None:
        if not self.p2_rules:
            raise ValueError("joint table must contain at least one P2 rule")
        if not self.p3_rules:
            raise ValueError("joint table must contain at least one P3 rule")


def _normalize_metadata(metadata: object) -> Tuple[Tuple[str, str], ...]:
    if isinstance(metadata, dict):
        return tuple(sorted((str(key), str(value)) for key, value in metadata.items()))
    if isinstance(metadata, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 for item in metadata
    ):
        return tuple((str(key), str(value)) for key, value in metadata)
    if isinstance(metadata, list) and all(
        isinstance(item, tuple) and len(item) == 2 for item in metadata
    ):
        return tuple((str(key), str(value)) for key, value in metadata)
    if metadata in (None, ()):
        return ()
    raise ValueError(f"metadata must be dict or tuple pairs, got {type(metadata).__name__}")

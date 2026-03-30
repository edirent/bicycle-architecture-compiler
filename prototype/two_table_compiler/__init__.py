"""Independent two-table Python prototype (full M/N/P0/P2/P3 semantics)."""

from .examples import DEMO_JOINT_TABLE, DEMO_LOCAL_TABLE
from .io import (
    DEFAULT_JOINT_TABLE_PATH,
    DEFAULT_LOCAL_TABLE_PATH,
    RandomCorpus,
    load_joint_table,
    load_local_table,
    load_random_corpus,
    save_joint_table,
    save_local_table,
    save_random_corpus,
)
from .search import (
    SearchStatus,
    SynthesisOutcome,
    SynthesisPlan,
    SynthesisSearchResult,
    TargetAuditReport,
    audit_target_execution,
    synthesize,
    synthesize_search,
    synthesize_target,
)
from .state import HiddenClass, State, Tail
from .tables import (
    JointTable,
    LocalTable,
    NativeEntry,
    P2Rule,
    P3Rule,
    SCOPE_CROSS_LOGICAL_BLOCK_NATIVE,
    SCOPE_INTER_MODULE_BELL,
    SCOPE_INTRA_BLOCK_NATIVE,
)
from .targeting import TargetClassification, canonicalize_logical_expression, classify_target

__all__ = [
    "DEMO_JOINT_TABLE",
    "DEMO_LOCAL_TABLE",
    "DEFAULT_LOCAL_TABLE_PATH",
    "DEFAULT_JOINT_TABLE_PATH",
    "JointTable",
    "LocalTable",
    "NativeEntry",
    "P2Rule",
    "P3Rule",
    "HiddenClass",
    "State",
    "Tail",
    "RandomCorpus",
    "load_local_table",
    "save_local_table",
    "load_joint_table",
    "save_joint_table",
    "load_random_corpus",
    "save_random_corpus",
    "SynthesisPlan",
    "SynthesisOutcome",
    "SynthesisSearchResult",
    "SearchStatus",
    "TargetAuditReport",
    "audit_target_execution",
    "synthesize",
    "synthesize_search",
    "synthesize_target",
    "TargetClassification",
    "classify_target",
    "canonicalize_logical_expression",
    "SCOPE_INTRA_BLOCK_NATIVE",
    "SCOPE_CROSS_LOGICAL_BLOCK_NATIVE",
    "SCOPE_INTER_MODULE_BELL",
]

from __future__ import annotations

from typing import Dict, List

from . import pauli
from .state import HIDDEN_I, HIDDEN_X, HIDDEN_Y, HIDDEN_Z, HiddenClass, Tail
from .tables import (
    JointTable,
    LocalTable,
    NativeEntry,
    P2_ALLOWED_SOURCE_SCOPES,
    P2_RELATION_COMMUTE,
    P2_TAIL_OP_PRODUCT,
    P2Rule,
    P3Rule,
)

DEMO_FRAME_ID = "demo_frame_11q"


def build_demo_local_table() -> LocalTable:
    return LocalTable(
        frame_id=DEMO_FRAME_ID,
        entries=[
            NativeEntry(native_id="N_I_Z2", frame_id=DEMO_FRAME_ID, head="I", tail=Tail.from_str("IZIIIIIIIII")),
            NativeEntry(native_id="N_X_X1", frame_id=DEMO_FRAME_ID, head="X", tail=Tail.from_str("XIIIIIIIIII")),
            NativeEntry(native_id="N_X_X8", frame_id=DEMO_FRAME_ID, head="X", tail=Tail.from_str("IIIIIIIXIII")),
            NativeEntry(native_id="N_Z_Z8", frame_id=DEMO_FRAME_ID, head="Z", tail=Tail.from_str("IIIIIIIZIII")),
        ],
    )


def _build_demo_p3_rules() -> List[P3Rule]:
    rules: List[P3Rule] = []
    axis_cost: Dict[str, int] = {"X": 2, "Y": 3, "Z": 2}

    for cur_hidden in (HIDDEN_I, HIDDEN_X, HIDDEN_Y, HIDDEN_Z):
        for axis_head in ("X", "Y", "Z"):
            next_hidden = HiddenClass(pauli.single_product(cur_hidden.name, axis_head))
            rules.append(
                P3Rule(
                    name=f"P3_{cur_hidden.name}_AXIS_{axis_head}_TO_{next_hidden.name}",
                    cur_hidden=cur_hidden,
                    axis_head=axis_head,
                    anticommute_required=True,
                    next_hidden=next_hidden,
                    delta_cost=axis_cost[axis_head],
                    template=(
                        "if state.tail anticommutes with axis.tail: "
                        f"({cur_hidden.name}, M) + ({axis_head}, P) -> ({next_hidden.name}, M*P)"
                    ),
                )
            )

    return rules


def build_demo_joint_table() -> JointTable:
    return JointTable(
        p2_rules=[
            P2Rule(
                name="P2_SAME_NONIDENTITY_HEAD_COMMUTE_TO_X",
                lhs_head="X",
                rhs_head="X",
                require_same_head=True,
                require_nonidentity_heads=True,
                required_relation=P2_RELATION_COMMUTE,
                out_hidden=HIDDEN_X,
                tail_op=P2_TAIL_OP_PRODUCT,
                total_cost=4,
                allowed_input_scopes=P2_ALLOWED_SOURCE_SCOPES,
                template="(X,P),(X,P') commute => source State(X,P*P') with cost 4",
            ),
            P2Rule(
                name="P2_SAME_NONIDENTITY_HEAD_COMMUTE_TO_Y",
                lhs_head="Y",
                rhs_head="Y",
                require_same_head=True,
                require_nonidentity_heads=True,
                required_relation=P2_RELATION_COMMUTE,
                out_hidden=HIDDEN_Y,
                tail_op=P2_TAIL_OP_PRODUCT,
                total_cost=4,
                allowed_input_scopes=P2_ALLOWED_SOURCE_SCOPES,
                template="(Y,P),(Y,P') commute => source State(Y,P*P') with cost 4",
            ),
            P2Rule(
                name="P2_SAME_NONIDENTITY_HEAD_COMMUTE_TO_Z",
                lhs_head="Z",
                rhs_head="Z",
                require_same_head=True,
                require_nonidentity_heads=True,
                required_relation=P2_RELATION_COMMUTE,
                out_hidden=HIDDEN_Z,
                tail_op=P2_TAIL_OP_PRODUCT,
                total_cost=4,
                allowed_input_scopes=P2_ALLOWED_SOURCE_SCOPES,
                template="(Z,P),(Z,P') commute => source State(Z,P*P') with cost 4",
            ),
        ],
        p3_rules=_build_demo_p3_rules(),
    )


DEMO_LOCAL_TABLE = build_demo_local_table()
DEMO_JOINT_TABLE = build_demo_joint_table()

TARGET_REACHABLE = Tail.from_str("XIIIIIIXIII")
TARGET_UNREACHABLE = Tail.from_str("YIIIIIIIIII")

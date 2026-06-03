from __future__ import annotations

from dataclasses import dataclass

from .tables import JointTable, NativeEntry, P2Rule

P0_SOURCE_COST = 1
P2_DIRECT_SOURCE_COST = 4
P3_I_TO_NONI_COST = 4
P3_NONI_COST = 6

HEAD_TO_DIGIT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
DIGIT_TO_HEAD = ("I", "X", "Y", "Z")


def head_xor(lhs: str, rhs: str) -> str:
    lhs_digit = HEAD_TO_DIGIT[lhs]
    rhs_digit = HEAD_TO_DIGIT[rhs]
    return DIGIT_TO_HEAD[lhs_digit ^ rhs_digit]


def derive_p0_source_cost(entry: NativeEntry) -> int:
    _ = entry
    return P0_SOURCE_COST


def derive_p2_source_cost(lhs: NativeEntry, rhs: NativeEntry, rule: P2Rule | None = None) -> int:
    if lhs.head != rhs.head or lhs.head not in ("X", "Y", "Z"):
        raise ValueError("paper P2 source requires same-basis non-identity inputs")
    if not lhs.tail.commute(rhs.tail):
        raise ValueError("paper P2 source requires commuting tails")
    if rule is not None and int(rule.total_cost) != P2_DIRECT_SOURCE_COST:
        raise ValueError(f"unexpected paper P2 total_cost={rule.total_cost}; expected {P2_DIRECT_SOURCE_COST}")
    return P2_DIRECT_SOURCE_COST


@dataclass(frozen=True)
class P3TransitionCost:
    next_head: str
    measurement_cost: int


def derive_p3_transition(prev_head: str, axis_head: str) -> P3TransitionCost:
    if axis_head not in ("X", "Y", "Z"):
        raise ValueError("paper P3 transition requires non-identity axis head")
    next_head = head_xor(prev_head, axis_head)
    cost = P3_I_TO_NONI_COST if prev_head == "I" else P3_NONI_COST
    return P3TransitionCost(next_head=next_head, measurement_cost=cost)


def validate_joint_table_matches_paper_skeleton(joint_table: JointTable) -> None:
    same_basis_rules: dict[str, int] = {}
    for rule in joint_table.p2_rules:
        if rule.lhs_head != rule.rhs_head or rule.lhs_head not in ("X", "Y", "Z"):
            continue
        if not rule.require_same_head or rule.required_relation != "commute":
            continue
        same_basis_rules[rule.lhs_head] = int(rule.total_cost)

    for head in ("X", "Y", "Z"):
        total_cost = same_basis_rules.get(head)
        if total_cost is None:
            raise ValueError(f"joint table is missing same-basis paper P2 rule for head={head}")
        if total_cost != P2_DIRECT_SOURCE_COST:
            raise ValueError(
                f"joint table P2 rule for head={head} has total_cost={total_cost}; "
                f"expected {P2_DIRECT_SOURCE_COST}"
            )

    seen_pairs = {(rule.cur_hidden.name, rule.axis_head) for rule in joint_table.p3_rules}
    for prev_head in ("I", "X", "Y", "Z"):
        for axis_head in ("X", "Y", "Z"):
            if (prev_head, axis_head) not in seen_pairs:
                raise ValueError(f"joint table is missing P3 skeleton rule for ({prev_head}, {axis_head})")

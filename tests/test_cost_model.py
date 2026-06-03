from lgm_compiler.cost_model import Placement, estimate_gross_cost
from lgm_compiler.lgm_ir import FramedCNOT, Measure2, Rot
from lgm_compiler.frame import CliffordFrame


def test_cost_model_counts_direct_pivot_rotations() -> None:
    report = estimate_gross_cost(
        [Rot(q=0, axis="Z", a=-1), Rot(q=1, axis="X", a=1)],
        Placement(module_id={0: 0, 1: 0}, logical_index={0: 1, 1: 7}),
    )

    assert report.direct_t_injection == 2
    assert report.nonpivot_rotation == 0
    assert report.fallback_or_route == 0


def test_cost_model_counts_nonpivot_and_measurement_macros() -> None:
    report = estimate_gross_cost(
        [
            Rot(q=0, axis="Z", a=-1),
            Measure2(0, "Z", 1, "X"),
            FramedCNOT(0, 1, CliffordFrame.identity(), CliffordFrame.identity()),
        ],
        Placement(module_id={0: 0, 1: 1}, logical_index={0: 2, 1: 7}),
    )

    assert report.nonpivot_rotation == 1
    assert report.fallback_or_route == 1
    assert report.inter_module_measurement == 1
    assert report.cnot_macro == 1

from lgm_compiler.frame import CliffordFrame
from lgm_compiler.pauli import SignedPauli


def test_h_update_conjugates_axes() -> None:
    frame = CliffordFrame.identity().updated("H")

    assert frame.image("X") == SignedPauli(1, "Z")
    assert frame.image("Y") == SignedPauli(-1, "Y")
    assert frame.image("Z") == SignedPauli(1, "X")


def test_s_then_h_composes_in_local_frame_order() -> None:
    frame = CliffordFrame.identity().updated("S").updated("H")

    assert frame.image("X") == SignedPauli(1, "Z")
    assert frame.image("Y") == SignedPauli(-1, "X")
    assert frame.image("Z") == SignedPauli(-1, "Y")

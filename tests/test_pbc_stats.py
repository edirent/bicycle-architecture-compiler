import json

import pytest

from lgm_compiler.pbc_stats import PBCStatsError, infer_pauli_string, load_pbc_stats


def test_pbc_stats_reads_rotation_basis_shape(tmp_path) -> None:
    path = tmp_path / "sample.pbc.jsonl"
    records = [
        {"Rotation": {"basis": ["Z", "I", "I"], "angle": "0.1"}},
        {"Rotation": {"basis": ["X", "Z", "I"], "angle": "0.2"}},
        {"Rotation": {"basis": ["X", "Y", "Z"], "angle": "0.3"}},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))

    stats = load_pbc_stats(path)

    assert stats.num_rotations == 3
    assert stats.pauli_weight_histogram == {1: 1, 2: 1, 3: 1}
    assert stats.num_weight_1 == 1
    assert stats.num_weight_2 == 1
    assert stats.num_weight_gt_2 == 1
    assert stats.mean_weight == 2.0
    assert stats.max_weight == 3


def test_infer_pauli_string_supports_multiple_shapes() -> None:
    assert infer_pauli_string({"pauli_string": "IXYZ"}) == ["I", "X", "Y", "Z"]
    assert infer_pauli_string({"P": {"0": "X", "2": "Z"}}) == ["X", "I", "Z"]
    assert infer_pauli_string({"ops": [{"qubit": 1, "pauli": "Y"}]}) == ["I", "Y"]


def test_pbc_stats_reports_useful_error_for_unknown_shape(tmp_path) -> None:
    path = tmp_path / "bad.pbc.jsonl"
    path.write_text(json.dumps({"Rotation": {"angle": "0.1"}}))

    with pytest.raises(PBCStatsError, match="could not infer Pauli string"):
        load_pbc_stats(path)

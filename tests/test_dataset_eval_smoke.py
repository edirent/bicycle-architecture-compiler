import json

from lgm_compiler.dataset_eval import evaluate_dataset, write_reports


def test_dataset_eval_smoke_on_tiny_dataset(tmp_path) -> None:
    root = tmp_path / "qasmbench_gross"
    (root / "pbc").mkdir(parents=True)
    (root / "metadata").mkdir()

    qasm = tmp_path / "tiny.qasm"
    qasm.write_text(
        "\n".join(
            [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                "qreg q[2];",
                "creg c[2];",
                "h q[0];",
                "cx q[0],q[1];",
                "u1(pi/4) q[1];",
                "measure q[0] -> c[0];",
            ]
        )
    )
    (root / "metadata" / "tiny.conversion.json").write_text(
        json.dumps({"benchmark": "tiny", "qasm_path": str(qasm), "gross_modules": 1})
    )
    (root / "pbc" / "tiny.pbc.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"Rotation": {"basis": ["Z", "I"], "angle": "0.1"}}),
                json.dumps({"Rotation": {"basis": ["X", "Z"], "angle": "0.2"}}),
                json.dumps({"Rotation": {"basis": ["X", "Y"], "angle": "0.3"}}),
            ]
        )
    )

    reports = evaluate_dataset(root)

    assert len(reports) == 1
    report = reports[0]
    assert report.benchmark == "tiny"
    assert report.lgm_available
    assert report.input_h == 1
    assert report.input_cx == 1
    assert report.unsupported_gate_count == 1
    assert report.unsupported_gate_names == ["u1"]
    assert report.lgm_framed_cnot == 1
    assert report.pbc_rotations == 3
    assert report.pbc_weight_gt_2 == 0

    json_path, csv_path = write_reports(reports, root)
    assert json_path.exists()
    assert csv_path.exists()


def test_dataset_eval_marks_missing_source_but_keeps_pbc_stats(tmp_path) -> None:
    root = tmp_path / "qasmbench_gross"
    (root / "pbc").mkdir(parents=True)
    (root / "pbc" / "missing.pbc.jsonl").write_text(
        json.dumps({"Rotation": {"basis": ["X", "Y", "Z"], "angle": "0.1"}})
    )

    report = evaluate_dataset(root)[0]

    assert report.benchmark == "missing"
    assert report.source_qasm is None
    assert not report.lgm_available
    assert report.pbc_rotations == 1
    assert report.pbc_weight_gt_2 == 1
    assert report.estimated_arbitrary_synthesis_units_pbc == 18.5

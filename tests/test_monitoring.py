import json

from src import lab_monitoring
from src import crop_workflow


def test_simulated_drift_is_detected_but_does_not_trigger_retraining(tmp_path, monkeypatch):
    monitoring_dir = tmp_path / "monitoring"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(lab_monitoring, "MONITORING_DIR", monitoring_dir)
    monkeypatch.setattr(lab_monitoring, "REPORT_DIR", reports_dir)
    monkeypatch.setattr(lab_monitoring, "LOG_PATH", tmp_path / "missing-inference.jsonl")

    lab_monitoring.simulate_input_drift()
    drift = lab_monitoring.data_drift()
    trigger = lab_monitoring.monitor_trigger()

    assert drift["source"] == "simulated_input_drift"
    assert drift["drift_detected"] is True
    assert trigger["retraining_recommended"] is False
    assert trigger["automatic_retraining"] is False
    assert trigger["status"] == "simulated_drift_test"
    saved_report = json.loads((reports_dir / "monitor_trigger_report.json").read_text())
    assert saved_report["automatic_retraining"] is False


def test_concept_drift_waits_for_real_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(lab_monitoring, "MONITORING_DIR", tmp_path / "monitoring")
    monkeypatch.setattr(lab_monitoring, "REPORT_DIR", tmp_path / "reports")

    report = lab_monitoring.concept_drift()

    assert report["status"] == "awaiting_feedback"
    assert report["drift_detected"] is False


def test_equal_candidate_does_not_replace_production_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(crop_workflow, "ROOT", tmp_path)
    monkeypatch.setattr(crop_workflow, "quality_gate", lambda: {"status": "passed", "value": 0.9})

    registry = {
        "versions": [
            {"version": 1, "metrics": {"macro_f1": 0.9}, "status": "production"},
            {"version": 2, "metrics": {"macro_f1": 0.9}, "status": "candidate"},
        ],
        "production_version": 1,
    }
    (tmp_path / "models").mkdir()
    (tmp_path / "deployment").mkdir()
    (tmp_path / "models" / "registry.json").write_text(json.dumps(registry))
    (tmp_path / "models" / "candidate_metadata.json").write_text(
        json.dumps({"model_name": "crop_random_forest", "model_version": "1.0.0"})
    )
    production_model = tmp_path / "deployment" / "model.json"
    production_model.write_text('{"active": "production"}')

    report = crop_workflow.lifecycle()

    assert report["status"] == "not_promoted"
    assert production_model.read_text() == '{"active": "production"}'

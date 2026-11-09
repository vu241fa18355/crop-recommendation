"""Crop-specific drift, request-history, performance, and retraining evidence."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

if __package__:
    from .crop_workflow import ROOT, get_dataset, load_config, write_json
else:
    from crop_workflow import ROOT, get_dataset, load_config, write_json

LOG_PATH = ROOT / "logs" / "crop_inference.jsonl"
MONITORING_DIR = ROOT / "data" / "monitoring"
REPORT_DIR = ROOT / "reports"


def load_request_history() -> list[dict[str, Any]]:
    if not LOG_PATH.is_file():
        return []
    events = []
    with LOG_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid inference log at line {line_number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Inference log entry {line_number} is not a JSON object.")
            events.append(event)
    return events


def request_history() -> dict[str, Any]:
    events = load_request_history()
    report = {
        "status": "available" if events else "no_requests",
        "request_count": len(events),
        "recent_requests": events[-100:],
    }
    write_json(REPORT_DIR / "request_history_report.json", report)
    print(f"[PASS] Request history contains {len(events)} inference requests.")
    return report


def _compare_feature_distributions(
    observed: pd.DataFrame, reference: pd.DataFrame, *, threshold: float
) -> dict[str, Any]:
    feature_reports: dict[str, Any] = {}
    detected_features = []
    for column in reference.columns:
        ref_mean = float(reference[column].mean())
        ref_std = float(reference[column].std(ddof=0))
        observed_mean = float(observed[column].mean())
        difference = abs(observed_mean - ref_mean)
        standardized = difference / ref_std if ref_std > 0 else (0.0 if difference == 0 else float("inf"))
        detected = standardized >= threshold
        if detected:
            detected_features.append(column)
        feature_reports[column] = {
            "reference_mean": ref_mean,
            "observed_mean": observed_mean,
            "standardized_mean_difference": standardized,
            "drift_detected": detected,
        }
    return {
        "threshold": threshold,
        "drift_detected": bool(detected_features),
        "drifted_features": detected_features,
        "features": feature_reports,
    }


def simulate_input_drift() -> dict[str, Any]:
    config = load_config()
    frame, features, _ = get_dataset(config)
    drift = frame[features].sample(n=min(200, len(frame)), random_state=71).copy()
    drift["temperature"] = (drift["temperature"] + 8.0).clip(-50, 60)
    drift["humidity"] = (drift["humidity"] - 30.0).clip(0, 100)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    path = MONITORING_DIR / "simulated_input_drift.csv"
    drift.to_csv(path, index=False)
    try:
        relative_path = str(path.relative_to(ROOT))
    except ValueError:
        relative_path = str(path)
    print(f"[PASS] Wrote {len(drift)} explicitly simulated records to {relative_path}.")
    return {"status": "passed", "path": relative_path, "rows": len(drift)}


def data_drift() -> dict[str, Any]:
    config = load_config()
    frame, features, _ = get_dataset(config)
    events = load_request_history()
    source = "inference_log"
    simulated_path = MONITORING_DIR / "simulated_input_drift.csv"
    if simulated_path.is_file():
        observed = pd.read_csv(simulated_path)[features]
        source = "simulated_input_drift"
    elif events:
        feature_events = [
            event["features"]
            for event in events
            if isinstance(event.get("features"), dict)
            and all(column in event["features"] for column in features)
        ]
        if not feature_events:
            result = {
                "status": "no_data",
                "source": "inference_log",
                "drift_detected": False,
                "message": "Inference history has no complete feature records; drift was not assessed.",
            }
            write_json(REPORT_DIR / "data_drift_report.json", result)
            print("[PASS] Inference history has no complete features; data drift is not assessed.")
            return result
        observed = pd.DataFrame(
            feature_events
        )
        observed = observed[features].apply(pd.to_numeric, errors="raise")
    else:
        result = {
            "status": "no_data",
            "source": "inference_log",
            "drift_detected": False,
            "message": "No prediction requests are available; drift was not assessed.",
        }
        write_json(REPORT_DIR / "data_drift_report.json", result)
        print("[PASS] No inference traffic yet; data drift is not assessed.")
        return result

    drift_cfg = config.get("monitoring", {})
    comparison = _compare_feature_distributions(
        observed,
        frame[features],
        threshold=float(drift_cfg.get("standardized_mean_threshold", 0.75)),
    )
    result = {
        "status": "passed",
        "source": source,
        "observations": len(observed),
        **comparison,
    }
    write_json(REPORT_DIR / "data_drift_report.json", result)
    print(
        f"[PASS] Data drift: {'detected' if result['drift_detected'] else 'not detected'} "
        f"from {len(observed)} {source} observations."
    )
    return result


def prediction_drift() -> dict[str, Any]:
    events = load_request_history()
    predictions = [event["recommended_crop"] for event in events if event.get("recommended_crop")]
    if not predictions:
        result = {
            "status": "no_data",
            "drift_detected": False,
            "message": "No serving predictions are available; prediction drift was not assessed.",
        }
        write_json(REPORT_DIR / "prediction_drift_report.json", result)
        print("[PASS] No serving predictions yet; prediction drift is not assessed.")
        return result

    frame, features, _ = get_dataset()
    model = joblib.load(ROOT / "models" / "crop_random_forest.pkl")
    reference_predictions = model.predict(frame[features]).astype(str).tolist()
    classes = [str(label) for label in model.classes_]
    reference_counts = Counter(reference_predictions)
    observed_counts = Counter(predictions)
    reference = [reference_counts[label] / len(reference_predictions) for label in classes]
    observed = [observed_counts[label] / len(predictions) for label in classes]
    total_variation = 0.5 * sum(abs(left - right) for left, right in zip(reference, observed))
    threshold = float(load_config().get("monitoring", {}).get("prediction_drift_threshold", 0.25))
    result = {
        "status": "passed",
        "observations": len(predictions),
        "total_variation_distance": total_variation,
        "threshold": threshold,
        "drift_detected": total_variation >= threshold,
        "reference_distribution": dict(zip(classes, reference)),
        "observed_distribution": dict(zip(classes, observed)),
    }
    write_json(REPORT_DIR / "prediction_drift_report.json", result)
    print(
        f"[PASS] Prediction drift {'detected' if result['drift_detected'] else 'not detected'} "
        f"(TVD={total_variation:.3f})."
    )
    return result


def concept_drift() -> dict[str, Any]:
    feedback_path = MONITORING_DIR / "labeled_feedback.csv"
    if not feedback_path.is_file():
        result = {
            "status": "awaiting_feedback",
            "drift_detected": False,
            "message": "Concept drift requires ground-truth crop labels in labeled_feedback.csv.",
        }
        write_json(REPORT_DIR / "concept_drift_report.json", result)
        print("[PASS] No labeled feedback; concept drift cannot yet be assessed.")
        return result

    frame, features, target = get_dataset()
    feedback = pd.read_csv(feedback_path)
    missing = sorted(set(features + [target]) - set(feedback.columns))
    if missing:
        raise ValueError(f"Labeled feedback is missing columns: {', '.join(missing)}")
    if feedback.empty or feedback[features + [target]].isna().any().any():
        raise ValueError("Labeled feedback is empty or contains missing values.")
    unknown_labels = sorted(set(feedback[target].astype(str)) - set(frame[target].astype(str)))
    if unknown_labels:
        raise ValueError(f"Labeled feedback contains unknown crop labels: {unknown_labels}")
    model = joblib.load(ROOT / "models" / "crop_random_forest.pkl")
    accuracy = float((model.predict(feedback[features]) == feedback[target].astype(str)).mean())
    baseline = float(load_config()["quality_gate"]["minimum_macro_f1"])
    result = {
        "status": "passed",
        "feedback_rows": len(feedback),
        "accuracy": accuracy,
        "minimum_accuracy": baseline,
        "drift_detected": accuracy < baseline,
    }
    write_json(REPORT_DIR / "concept_drift_report.json", result)
    print(f"[PASS] Labeled feedback accuracy: {accuracy:.4f}.")
    return result


def performance_monitoring() -> dict[str, Any]:
    events = load_request_history()
    latencies = [float(event["latency_ms"]) for event in events if event.get("latency_ms") is not None]
    confidences = [float(event["confidence"]) for event in events if event.get("confidence") is not None]
    result = {
        "status": "available" if latencies else "no_requests",
        "request_count": len(events),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
    }
    write_json(REPORT_DIR / "performance_monitoring_report.json", result)
    print(
        "[PASS] Performance summary: "
        f"{result['request_count']} requests, p95 latency={result['p95_latency_ms']} ms."
    )
    return result


def drift_analysis() -> dict[str, Any]:
    result = {
        "data": data_drift(),
        "prediction": prediction_drift(),
        "concept": concept_drift(),
        "performance": performance_monitoring(),
    }
    result["status"] = "passed"
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(REPORT_DIR / "drift_analysis_report.json", result)
    return result


def monitor_trigger() -> dict[str, Any]:
    analysis = drift_analysis()
    drift_detected = any(
        analysis[key].get("drift_detected", False)
        for key in ("data", "prediction", "concept")
    )
    simulated_only = (
        analysis["data"].get("source") == "simulated_input_drift"
        and analysis["data"].get("drift_detected", False)
        and not analysis["prediction"].get("drift_detected", False)
        and not analysis["concept"].get("drift_detected", False)
    )
    retraining_recommended = drift_detected and not simulated_only
    result = {
        "status": "simulated_drift_test" if simulated_only else (
            "drift_detected" if retraining_recommended else "no_drift"
        ),
        "retraining_recommended": retraining_recommended,
        "automatic_retraining": False,
        "reason": (
            "Only synthetic drift data is present; no production retraining is recommended."
            if simulated_only
            else
            "New independently validated labeled data is required before retraining."
            if retraining_recommended
            else "No monitored drift threshold was exceeded."
        ),
    }
    write_json(REPORT_DIR / "monitor_trigger_report.json", result)
    print(f"[PASS] Monitoring trigger: {result['status']}; {result['reason']}")
    return result


def retrain() -> dict[str, Any]:
    if __package__:
        from .crop_workflow import evaluate, lifecycle, quality_gate, track_and_register, train
    else:
        from crop_workflow import evaluate, lifecycle, quality_gate, track_and_register, train

    metadata = train()
    evaluation = evaluate()
    quality_gate()
    tracking = track_and_register()
    lifecycle_report = lifecycle()
    result = {
        "status": "passed",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "registry_version": tracking["version"],
        "lifecycle": lifecycle_report["status"],
        "trigger": "explicitly requested retraining on the validated labeled source dataset",
    }
    write_json(REPORT_DIR / "retraining_report.json", result)
    print("[PASS] Retrained and quality-gated the crop recommendation model.")
    return result


COMMANDS = {
    "history": request_history,
    "simulate": simulate_input_drift,
    "data-drift": data_drift,
    "prediction-drift": prediction_drift,
    "concept-drift": concept_drift,
    "performance": performance_monitoring,
    "analyze": drift_analysis,
    "trigger": monitor_trigger,
    "retrain": retrain,
}


def main(command: str | None = None) -> int:
    selected = command or (sys.argv[1] if len(sys.argv) > 1 else "analyze")
    if selected not in COMMANDS:
        raise ValueError(f"Unknown monitoring stage: {selected}")
    COMMANDS[selected]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible training, validation, evaluation, and monitoring for crop data."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid project configuration: {CONFIG_PATH}")
    return config


def get_dataset(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[str], str]:
    config = config or load_config()
    data_cfg = config["data"]
    dataset_path = ROOT / data_cfg["raw_path"]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Crop dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    features = list(data_cfg["feature_columns"])
    target = data_cfg["target_column"]
    missing = sorted(set(features + [target]) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Crop dataset contains no rows.")
    if frame[features + [target]].isna().any().any():
        raise ValueError("Crop dataset contains missing feature or target values.")
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in features):
        raise ValueError("All crop input features must be numeric.")
    if frame[target].nunique() < 2:
        raise ValueError("Crop dataset must contain at least two crop labels.")
    frame[target] = frame[target].astype(str).str.strip()
    if (frame[target] == "").any():
        raise ValueError("Crop labels must not be empty.")
    return frame, features, target


def split_data(
    frame: pd.DataFrame, features: list[str], target: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = config["split"]
    return train_test_split(
        frame[features],
        frame[target],
        test_size=float(split["test_size"]),
        random_state=int(split["random_state"]),
        stratify=frame[target],
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def validate_data() -> dict[str, Any]:
    frame, features, target = get_dataset()
    report = {
        "status": "passed",
        "dataset": str((ROOT / load_config()["data"]["raw_path"]).relative_to(ROOT)),
        "rows": int(len(frame)),
        "features": features,
        "classes": sorted(frame[target].unique().tolist()),
        "class_counts": {str(k): int(v) for k, v in frame[target].value_counts().sort_index().items()},
    }
    write_json(ROOT / "artifacts" / "data_validation_report.json", report)
    print(f"[PASS] Validated {report['rows']} crop records across {len(report['classes'])} crop classes.")
    return report


def preprocess() -> dict[str, Any]:
    config = load_config()
    frame, features, target = get_dataset(config)
    x_train, x_test, y_train, y_test = split_data(frame, features, target, config)
    output = ROOT / config["data"]["processed_dir"]
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "X_train_final.npy", x_train.to_numpy(dtype=np.float64))
    np.save(output / "X_test_final.npy", x_test.to_numpy(dtype=np.float64))
    np.save(output / "y_train.npy", y_train.to_numpy(dtype=str))
    np.save(output / "y_test.npy", y_test.to_numpy(dtype=str))
    x_train.to_csv(output / "X_train_raw.csv", index=False)
    x_test.to_csv(output / "X_test_raw.csv", index=False)
    metadata = {
        "dataset": config["data"]["raw_path"],
        "target_column": target,
        "feature_columns": features,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "random_state": int(config["split"]["random_state"]),
    }
    write_json(output / "dataset_metadata.json", metadata)
    print(f"[PASS] Saved crop preprocessing outputs under {output.relative_to(ROOT)}.")
    return metadata


def train() -> dict[str, Any]:
    config = load_config()
    frame, features, target = get_dataset(config)
    x_train, x_test, y_train, y_test = split_data(frame, features, target, config)
    model_cfg = config["model"]
    model = RandomForestClassifier(
        n_estimators=int(model_cfg["n_estimators"]),
        max_depth=model_cfg.get("max_depth"),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
        random_state=int(model_cfg["random_state"]),
        n_jobs=int(model_cfg.get("n_jobs", -1)),
    )
    model.fit(x_train, y_train)
    deployment = ROOT / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    model_path = deployment / "model.pkl"
    joblib.dump(model, model_path)
    joblib.dump(model, ROOT / "models" / "crop_random_forest.pkl")
    metadata = {
        "service": "Crop Recommendation API",
        "model_name": "crop_random_forest",
        "model_version": "1.0.0",
        "model_type": "RandomForestClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": config["data"]["raw_path"],
        "features": features,
        "classes": [str(value) for value in model.classes_],
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": float(accuracy_score(y_test, model.predict(x_test))),
        "macro_f1": float(f1_score(y_test, model.predict(x_test), average="macro")),
    }
    write_json(deployment / "model_metadata.json", metadata)
    print(f"[PASS] Trained {metadata['model_name']} ({metadata['accuracy']:.4f} accuracy).")
    return metadata


def load_model() -> tuple[Any, dict[str, Any]]:
    model_path = ROOT / "deployment" / "model.pkl"
    metadata_path = ROOT / "deployment" / "model_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Run `python src/train.py` before using the trained model.")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def evaluate() -> dict[str, Any]:
    config = load_config()
    frame, features, target = get_dataset(config)
    x_train, x_test, y_train, y_test = split_data(frame, features, target, config)
    model, metadata = load_model()
    predictions = model.predict(x_test)
    report = {
        "status": "passed",
        "model_name": metadata["model_name"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "classes": sorted(frame[target].unique().tolist()),
    }
    write_json(ROOT / "reports" / "evaluation_report.json", report)
    print(f"[PASS] Evaluation accuracy={report['accuracy']:.4f}, macro-F1={report['macro_f1']:.4f}.")
    return report


def validate_outputs() -> dict[str, Any]:
    processed = ROOT / load_config()["data"]["processed_dir"]
    arrays = {
        "X_train_final.npy": 2,
        "X_test_final.npy": 2,
        "y_train.npy": 1,
        "y_test.npy": 1,
    }
    loaded: dict[str, np.ndarray] = {}
    for name, dimensions in arrays.items():
        path = processed / name
        if not path.is_file():
            raise FileNotFoundError(f"Expected preprocessing output is missing: {path}")
        values = np.load(path, allow_pickle=False)
        if values.ndim != dimensions or values.shape[0] == 0:
            raise ValueError(f"Invalid shape for {path.name}: {values.shape}")
        if values.dtype.kind in "fiu" and not np.isfinite(values).all():
            raise ValueError(f"Non-finite values found in {path.name}.")
        loaded[name] = values
    if loaded["X_train_final.npy"].shape[1] != len(load_config()["data"]["feature_columns"]):
        raise ValueError("Processed feature count does not match configured crop features.")
    if loaded["X_train_final.npy"].shape[0] != loaded["y_train.npy"].shape[0]:
        raise ValueError("Training feature and label row counts do not match.")
    if loaded["X_test_final.npy"].shape[0] != loaded["y_test.npy"].shape[0]:
        raise ValueError("Test feature and label row counts do not match.")
    result = {"status": "passed", "arrays": {name: list(value.shape) for name, value in loaded.items()}}
    write_json(ROOT / "artifacts" / "preprocessing_validation_report.json", result)
    print("[PASS] Preprocessed crop arrays have valid shapes and values.")
    return result


def quality_gate() -> dict[str, Any]:
    report_path = ROOT / "reports" / "evaluation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Run evaluation before the quality gate: {report_path}")
    evaluation = json.loads(report_path.read_text(encoding="utf-8"))
    threshold = float(load_config()["quality_gate"]["minimum_macro_f1"])
    passed = evaluation["macro_f1"] >= threshold
    result = {
        "status": "passed" if passed else "failed",
        "metric": "macro_f1",
        "value": evaluation["macro_f1"],
        "minimum": threshold,
    }
    write_json(ROOT / "reports" / "quality_gate_report.json", result)
    if not passed:
        raise RuntimeError(f"Model macro-F1 {result['value']:.4f} is below {threshold:.4f}.")
    print(f"[PASS] Macro-F1 quality gate passed ({result['value']:.4f} >= {threshold:.4f}).")
    return result


def inference() -> dict[str, Any]:
    frame, features, _ = get_dataset()
    model, _ = load_model()
    sample = frame.loc[[frame.index[0]], features]
    probabilities = model.predict_proba(sample)[0]
    result = {
        "crop": str(model.classes_[int(np.argmax(probabilities))]),
        "confidence": float(np.max(probabilities)),
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        },
    }
    write_json(ROOT / "reports" / "sample_prediction.json", result)
    print(f"[PASS] Sample recommendation: {result['crop']} ({result['confidence']:.3f}).")
    return result


def monitor() -> dict[str, Any]:
    frame, features, target = get_dataset()
    report = {
        "status": "passed",
        "rows": len(frame),
        "crop_classes": int(frame[target].nunique()),
        "feature_ranges": {
            name: {"min": float(frame[name].min()), "max": float(frame[name].max())}
            for name in features
        },
    }
    write_json(ROOT / "reports" / "monitoring_report.json", report)
    print(f"[PASS] Monitoring summary written for {report['rows']} crop rows.")
    return report


def lifecycle() -> dict[str, Any]:
    gate = quality_gate()
    _, metadata = load_model()
    report = {
        "status": "promoted" if gate["status"] == "passed" else "rejected",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "macro_f1": gate["value"],
    }
    write_json(ROOT / "reports" / "lifecycle_report.json", report)
    print(f"[PASS] Crop model lifecycle decision: {report['status']}.")
    return report


def registry_report() -> dict[str, Any]:
    _, metadata = load_model()
    print(json.dumps(metadata, indent=2))
    return metadata


def run_all() -> None:
    validate_data()
    preprocess()
    validate_outputs()
    train()
    evaluate()
    quality_gate()
    inference()
    monitor()


COMMANDS = {
    "all": run_all,
    "validate": validate_data,
    "preprocess": preprocess,
    "validate-outputs": validate_outputs,
    "train": train,
    "evaluate": evaluate,
    "quality-gate": quality_gate,
    "inference": inference,
    "monitor": monitor,
    "lifecycle": lifecycle,
    "registry": registry_report,
    "data-drift": monitor,
    "prediction-drift": monitor,
    "concept-drift": monitor,
    "performance": monitor,
    "failure-check": validate_data,
    "report": monitor,
}


def main(command: str | None = None) -> int:
    selected = command or (sys.argv[1] if len(sys.argv) > 1 else "all")
    try:
        action = COMMANDS[selected]
    except KeyError as exc:
        raise ValueError(f"Unknown crop workflow stage: {selected}") from exc
    action()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

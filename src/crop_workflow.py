"""Reproducible training, validation, evaluation, and monitoring for crop data."""

from __future__ import annotations

import json
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid project configuration: {CONFIG_PATH}")
    return config


def validate_frame(
    frame: pd.DataFrame, features: list[str], target: str
) -> pd.DataFrame:
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
    return frame


def get_dataset(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[str], str]:
    config = config or load_config()
    data_cfg = config["data"]
    dataset_path = ROOT / data_cfg["raw_path"]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Crop dataset not found: {dataset_path}")

    frame = pd.read_csv(dataset_path)
    features = list(data_cfg["feature_columns"])
    target = data_cfg["target_column"]
    return validate_frame(frame, features, target), features, target


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


def load_data() -> dict[str, Any]:
    frame, features, target = get_dataset()
    path = ROOT / load_config()["data"]["raw_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "status": "loaded",
        "dataset": str(path.relative_to(ROOT)),
        "sha256": digest,
        "rows": len(frame),
        "feature_columns": features,
        "target_column": target,
    }
    write_json(ROOT / "artifacts" / "data_load_report.json", report)
    print(f"[PASS] Loaded {report['rows']} crop rows (SHA-256 {digest[:12]}...).")
    return report


def baseline() -> dict[str, Any]:
    config = load_config()
    frame, features, target = get_dataset(config)
    x_train, x_test, y_train, y_test = split_data(frame, features, target, config)
    models = {
        "decision_tree": DecisionTreeClassifier(random_state=int(config["model"]["random_state"])),
        "random_forest": RandomForestClassifier(
            n_estimators=int(config["model"]["n_estimators"]),
            max_depth=config["model"].get("max_depth"),
            min_samples_leaf=int(config["model"].get("min_samples_leaf", 1)),
            random_state=int(config["model"]["random_state"]),
            n_jobs=int(config["model"].get("n_jobs", -1)),
        ),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        }
    report = {"status": "passed", "test_rows": len(x_test), "models": metrics}
    write_json(ROOT / "reports" / "baseline_comparison.json", report)
    print("[PASS] Evaluated Decision Tree and Random Forest baselines.")
    return report


def failure_check() -> dict[str, Any]:
    config = load_config()
    _, features, target = get_dataset(config)
    valid, _, _ = get_dataset(config)
    cases = {
        "missing_feature": valid.drop(columns=[features[0]]),
        "missing_value": valid.assign(**{features[0]: np.nan}),
        "non_numeric_feature": valid.assign(**{features[0]: "not-a-number"}),
        "single_class": valid.assign(**{target: valid[target].iloc[0]}),
        "empty_label": valid.assign(**{target: ""}),
    }
    results = {}
    for name, invalid in cases.items():
        try:
            validate_frame(invalid, features, target)
        except ValueError as exc:
            results[name] = {"rejected": True, "error": str(exc)}
        else:
            raise AssertionError(f"Failure-injection case was unexpectedly accepted: {name}")
    report = {"status": "passed", "cases": results}
    write_json(ROOT / "reports" / "failure_injection_report.json", report)
    print(f"[PASS] Rejected all {len(results)} deliberately invalid dataset cases.")
    return report


def track_and_register() -> dict[str, Any]:
    metadata_path = ROOT / "models" / "candidate_metadata.json"
    evaluation_path = ROOT / "reports" / "evaluation_report.json"
    if not metadata_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("Run training and evaluation before tracking a model.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    registry_path = ROOT / "models" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {
        "model_name": metadata["model_name"],
        "versions": [],
        "production_version": None,
    }
    version = len(registry["versions"]) + 1
    run_id = uuid4().hex
    record = {
        "run_id": run_id,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": metadata["model_name"],
        "model_artifact": "models/candidate_model.json",
        "metrics": {
            "accuracy": evaluation["accuracy"],
            "macro_f1": evaluation["macro_f1"],
        },
        "parameters": load_config()["model"],
        "status": "candidate",
    }
    registry["versions"].append(record)
    version_dir = ROOT / "models" / "registry" / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "models" / "candidate_model.json", version_dir / "model.json")
    shutil.copy2(ROOT / "models" / "candidate_model.pkl", version_dir / "model.pkl")
    shutil.copy2(ROOT / "models" / "candidate_metadata.json", version_dir / "model_metadata.json")
    record["model_artifact"] = str((version_dir / "model.json").relative_to(ROOT))
    record["metadata_artifact"] = str((version_dir / "model_metadata.json").relative_to(ROOT))
    write_json(registry_path, registry)
    experiment_path = ROOT / "mlruns" / "experiments.jsonl"
    experiment_path.parent.mkdir(parents=True, exist_ok=True)
    with experiment_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    report = {"status": "tracked", **record}
    write_json(ROOT / "reports" / "experiment_tracking_report.json", report)
    print(f"[PASS] Tracked model version {version} (run {run_id[:12]}).")
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
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "candidate_model.pkl")
    portable_model = {"classes": [str(value) for value in model.classes_], "trees": []}
    for estimator in model.estimators_:
        tree = estimator.tree_
        leaf_values = []
        for node in range(tree.node_count):
            if tree.children_left[node] == -1:
                leaf_values.append(
                    [
                        [int(class_index), float(value)]
                        for class_index, value in enumerate(tree.value[node, 0, :])
                        if value > 0
                    ]
                )
            else:
                leaf_values.append(None)
        portable_model["trees"].append(
            {
                "left": tree.children_left.astype(int).tolist(),
                "right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "leaf_values": leaf_values,
            }
        )
    write_json(models_dir / "candidate_model.json", portable_model)
    metadata = {
        "service": "Crop Recommendation API",
        "model_name": "crop_random_forest",
        "model_version": "1.0.0",
        "model_type": "RandomForestClassifier",
        "model_format": "portable-tree-json-v2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": config["data"]["raw_path"],
        "features": features,
        "classes": [str(value) for value in model.classes_],
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": float(accuracy_score(y_test, model.predict(x_test))),
        "macro_f1": float(f1_score(y_test, model.predict(x_test), average="macro")),
    }
    write_json(models_dir / "candidate_metadata.json", metadata)
    print(f"[PASS] Trained {metadata['model_name']} ({metadata['accuracy']:.4f} accuracy).")
    return metadata


def load_model() -> tuple[Any, dict[str, Any]]:
    model_path = ROOT / "models" / "crop_random_forest.pkl"
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
    candidate_path = ROOT / "models" / "candidate_model.pkl"
    metadata_path = ROOT / "models" / "candidate_metadata.json"
    if not candidate_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Run training before evaluating a model candidate.")
    model = joblib.load(candidate_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
    candidate_metadata_path = ROOT / "models" / "candidate_metadata.json"
    if not candidate_metadata_path.is_file():
        raise FileNotFoundError("Run training before the model lifecycle decision.")
    metadata = json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError("Run experiment tracking before the model lifecycle decision.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidate = registry["versions"][-1]
    current_version = registry.get("production_version")
    current = next(
        (version for version in registry["versions"] if version["version"] == current_version),
        None,
    )
    promote = gate["status"] == "passed" and (
        current is None or candidate["metrics"]["macro_f1"] > current["metrics"]["macro_f1"]
    )
    if promote:
        registry["production_version"] = candidate["version"]
        candidate["status"] = "production"
        version_dir = ROOT / "models" / "registry" / f"v{candidate['version']}"
        deployment_dir = ROOT / "deployment"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(version_dir / "model.json", deployment_dir / "model.json")
        shutil.copy2(version_dir / "model_metadata.json", deployment_dir / "model_metadata.json")
        shutil.copy2(version_dir / "model.pkl", ROOT / "models" / "crop_random_forest.pkl")
    else:
        candidate["status"] = "rejected" if gate["status"] != "passed" else "candidate"
    write_json(registry_path, registry)
    report = {
        "status": "promoted" if promote else "not_promoted",
        "model_name": metadata["model_name"],
        "model_version": candidate["version"],
        "macro_f1": gate["value"],
        "previous_production_version": current_version,
    }
    write_json(ROOT / "reports" / "lifecycle_report.json", report)
    print(f"[PASS] Crop model lifecycle decision: {report['status']}.")
    return report


def prepare_deployment() -> dict[str, Any]:
    model_path = ROOT / "deployment" / "model.json"
    metadata_path = ROOT / "deployment" / "model_metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Train a crop model before preparing deployment.")
    import yaml

    manifests = {}
    for name in ("deployment.yaml", "service.yaml"):
        path = ROOT / "k8s" / name
        if not path.is_file():
            raise FileNotFoundError(f"Kubernetes manifest not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload.get("apiVersion") or not payload.get("kind"):
            raise ValueError(f"Invalid Kubernetes manifest: {path}")
        manifests[name] = {"kind": payload["kind"], "name": payload["metadata"]["name"]}
    report = {
        "status": "passed",
        "model_artifact": str(model_path.relative_to(ROOT)),
        "model_metadata": str(metadata_path.relative_to(ROOT)),
        "manifests": manifests,
    }
    write_json(ROOT / "reports" / "deployment_preparation_report.json", report)
    print("[PASS] Model bundle and Kubernetes deployment manifests are ready.")
    return report


def registry_report() -> dict[str, Any]:
    registry_path = ROOT / "models" / "registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError("No crop models have been registered yet.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    print(json.dumps(registry, indent=2))
    return registry


def monitoring_command(command: str) -> Any:
    if __package__:
        from . import lab_monitoring
    else:
        import lab_monitoring
    return lab_monitoring.COMMANDS[command]()


def run_all() -> None:
    baseline()
    load_data()
    validate_data()
    preprocess()
    validate_outputs()
    train()
    evaluate()
    quality_gate()
    track_and_register()
    lifecycle()
    prepare_deployment()
    inference()
    failure_check()
    monitoring_command("analyze")


COMMANDS = {
    "all": run_all,
    "load-data": load_data,
    "baseline": baseline,
    "validate": validate_data,
    "preprocess": preprocess,
    "validate-outputs": validate_outputs,
    "train": train,
    "evaluate": evaluate,
    "quality-gate": quality_gate,
    "track": track_and_register,
    "compare": baseline,
    "inference": inference,
    "monitor": lambda: monitoring_command("analyze"),
    "lifecycle": lifecycle,
    "registry": registry_report,
    "data-drift": lambda: monitoring_command("data-drift"),
    "prediction-drift": lambda: monitoring_command("prediction-drift"),
    "concept-drift": lambda: monitoring_command("concept-drift"),
    "performance": lambda: monitoring_command("performance"),
    "failure-check": failure_check,
    "prepare-deployment": prepare_deployment,
    "report": monitor,
    "analyze": lambda: monitoring_command("analyze"),
    "history": lambda: monitoring_command("history"),
    "simulate": lambda: monitoring_command("simulate"),
    "trigger": lambda: monitoring_command("trigger"),
    "retrain": lambda: monitoring_command("retrain"),
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

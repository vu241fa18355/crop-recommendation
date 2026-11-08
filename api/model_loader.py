"""Load the model bundle produced by the crop training workflow."""

import json
from pathlib import Path
from typing import Any

import joblib

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


class ModelLoader:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.model = None
        self.metadata: dict[str, Any] = {}
        self.model_name: str | None = None
        self.model_version: str | None = None

    def load(self) -> None:
        model_path = self.project_root / "deployment" / "model.pkl"
        metadata_path = self.project_root / "deployment" / "model_metadata.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"Crop model not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Crop model metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("features") != FEATURE_COLUMNS:
            raise ValueError("Model feature metadata does not match the crop API schema.")
        model = joblib.load(model_path)
        if not hasattr(model, "predict_proba") or list(map(str, model.classes_)) != metadata.get("classes"):
            raise ValueError("The bundled model does not match its metadata.")

        self.model = model
        self.metadata = metadata
        self.model_name = metadata.get("model_name")
        self.model_version = metadata.get("model_version")

    def is_ready(self) -> bool:
        return self.model is not None

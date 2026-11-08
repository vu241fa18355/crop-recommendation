"""Load and evaluate the portable crop forest bundle."""

import json
import math
from pathlib import Path
from typing import Any

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


class CropForest:
    def __init__(self, classes: list[str], trees: list[dict[str, Any]]) -> None:
        if not classes or not trees:
            raise ValueError("Portable crop model must contain classes and decision trees.")
        self.classes_ = classes
        self.trees = trees
        for tree in trees:
            lengths = {
                len(tree.get(key, []))
                for key in ("left", "right", "feature", "threshold", "leaf_values")
            }
            if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
                raise ValueError("Portable crop model contains an invalid decision tree.")
            for node, leaf_values in enumerate(tree["leaf_values"]):
                if tree["left"][node] == -1:
                    if not leaf_values or any(
                        not 0 <= class_index < len(classes) or value <= 0
                        for class_index, value in leaf_values
                    ):
                        raise ValueError("Portable crop tree contains invalid leaf values.")
                elif leaf_values is not None:
                    raise ValueError("Portable crop tree contains values on an internal node.")

    def predict_proba(self, rows: list[list[float]]) -> list[list[float]]:
        if not rows:
            raise ValueError("At least one crop feature row is required.")
        probabilities = [[0.0] * len(self.classes_) for _ in rows]
        for tree in self.trees:
            for row_index, row in enumerate(rows):
                if len(row) != len(FEATURE_COLUMNS):
                    raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {len(row)}.")
                node = 0
                while tree["left"][node] != -1:
                    feature_index = tree["feature"][node]
                    if not 0 <= feature_index < len(row):
                        raise ValueError("Portable crop tree contains an invalid feature index.")
                    branch = "left" if row[feature_index] <= tree["threshold"][node] else "right"
                    node = tree[branch][node]
                class_values = tree["leaf_values"][node]
                total = sum(value for _, value in class_values)
                if total <= 0 or not math.isfinite(total):
                    raise ValueError("Portable crop tree contains invalid class values.")
                for class_index, value in class_values:
                    probabilities[row_index][class_index] += value / total
        tree_count = len(self.trees)
        return [
            [value / tree_count for value in row_probabilities]
            for row_probabilities in probabilities
        ]


class ModelLoader:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.model: CropForest | None = None
        self.metadata: dict[str, Any] = {}
        self.model_name: str | None = None
        self.model_version: str | None = None

    def load(self) -> None:
        model_path = self.project_root / "deployment" / "model.json"
        metadata_path = self.project_root / "deployment" / "model_metadata.json"
        if not model_path.is_file():
            raise FileNotFoundError(f"Portable crop model not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Crop model metadata not found: {metadata_path}")

        payload = json.loads(model_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_format") != "portable-tree-json-v2":
            raise ValueError("Unsupported crop model format in metadata.")
        if metadata.get("features") != FEATURE_COLUMNS:
            raise ValueError("Model feature metadata does not match the crop API schema.")
        if payload.get("classes") != metadata.get("classes"):
            raise ValueError("Crop model classes do not match the metadata.")
        model = CropForest(payload["classes"], payload["trees"])

        self.model = model
        self.metadata = metadata
        self.model_name = metadata.get("model_name")
        self.model_version = metadata.get("model_version")

    def is_ready(self) -> bool:
        return self.model is not None

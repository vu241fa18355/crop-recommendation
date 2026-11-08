"""Verify the crop dataset, bundled model, and one prediction."""

import json

from api.model_loader import FEATURE_COLUMNS, ModelLoader
from src.crop_workflow import get_dataset


def main() -> int:
    frame, features, _ = get_dataset()
    if features != FEATURE_COLUMNS:
        raise ValueError(f"Dataset features do not match the API: {features}")
    loader = ModelLoader()
    loader.load()
    if loader.model is None:
        raise RuntimeError("Crop model failed to load.")
    row = frame.loc[frame.index[0], features].astype(float).tolist()
    probabilities = loader.model.predict_proba([row])[0]
    crop = str(loader.model.classes_[probabilities.index(max(probabilities))])
    print(f"[PASS] Dataset contains {len(frame)} crop examples.")
    print(f"[PASS] Loaded {loader.model_name} v{loader.model_version}.")
    print(f"[PASS] Sample crop recommendation: {crop} ({max(probabilities):.3f}).")
    print(json.dumps(loader.metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

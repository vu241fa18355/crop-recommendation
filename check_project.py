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
    probabilities = loader.model.predict_proba(frame.loc[[frame.index[0]], features])[0]
    crop = str(loader.model.classes_[probabilities.argmax()])
    print(f"[PASS] Dataset contains {len(frame)} crop examples.")
    print(f"[PASS] Loaded {loader.model_name} v{loader.model_version}.")
    print(f"[PASS] Sample crop recommendation: {crop} ({probabilities.max():.3f}).")
    print(json.dumps(loader.metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

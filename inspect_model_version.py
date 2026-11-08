"""Display the versioned crop model metadata."""

import json
from pathlib import Path

metadata_path = Path(__file__).resolve().parent / "deployment" / "model_metadata.json"
if not metadata_path.is_file():
    raise FileNotFoundError(f"Train the crop model first: {metadata_path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
print(f"MODEL NAME: {metadata['model_name']}")
print(f"VERSION: {metadata['model_version']}")
print(f"TRAINED: {metadata['trained_at']}")
print(f"ACCURACY: {metadata['accuracy']:.4f}")

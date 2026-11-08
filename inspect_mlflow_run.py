"""Display the active crop model metadata (MLflow is not required)."""

import json
from pathlib import Path

metadata_path = Path(__file__).resolve().parent / "deployment" / "model_metadata.json"
if not metadata_path.is_file():
    raise FileNotFoundError(f"Train the crop model first: {metadata_path}")
print(json.dumps(json.loads(metadata_path.read_text(encoding="utf-8")), indent=2))

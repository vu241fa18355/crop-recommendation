"""Append crop inference metadata to a local JSON Lines log."""

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "crop_inference.jsonl"


def log_inference(
    *,
    features: dict[str, float],
    prediction: str,
    confidence: float,
    latency_ms: float,
    model_name: str | None,
    model_version: str | None,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "recommended_crop": prediction,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "model_name": model_name,
        "model_version": model_version,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")

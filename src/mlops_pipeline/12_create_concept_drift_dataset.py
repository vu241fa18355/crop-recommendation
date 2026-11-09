"""Crop-compatible entry point for the 12_create_concept_drift_dataset workflow."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.crop_workflow import main

if __name__ == "__main__":
    raise SystemExit(main("concept-drift"))

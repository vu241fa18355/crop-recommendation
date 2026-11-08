"""Crop-compatible entry point for the 06_model_evaluation workflow."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.crop_workflow import main

if __name__ == "__main__":
    raise SystemExit(main("evaluate"))

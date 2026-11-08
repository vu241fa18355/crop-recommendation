"""Crop-compatible entry point for the run_lab4_tracking workflow."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crop_workflow import main

if __name__ == "__main__":
    raise SystemExit(main("evaluate"))

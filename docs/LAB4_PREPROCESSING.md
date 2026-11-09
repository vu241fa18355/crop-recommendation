# Lab 4 — Reproducible Preprocessing

The workflow uses a stratified, deterministic train/test split configured in `config.yaml`. It saves feature arrays, labels, human-readable feature CSVs, and split metadata to `data/processed/`. Output validation checks dimensions, finite measurements, and corresponding row counts.

```powershell
python src/mlops_pipeline/03_preprocessing.py
python src/mlops_pipeline/04_validate_outputs.py
```

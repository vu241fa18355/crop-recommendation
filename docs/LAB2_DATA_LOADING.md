# Lab 2 — Dataset Loading and Provenance

The configured crop CSV is loaded by the shared workflow. A SHA-256 digest, relative dataset path, row count, feature list, and target name are written to `artifacts/data_load_report.json`.

```powershell
python src/mlops_pipeline/01_load_data.py
```

The loader never overwrites or silently falls back to another dataset.

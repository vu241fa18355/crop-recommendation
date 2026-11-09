# Lab 8 — Experiment Tracking, Registry, and CI

Training parameters, validation metrics, timestamps, run IDs, and model artifact references are recorded to an append-only local experiment log at `mlruns/experiments.jsonl`. `models/registry.json` maintains candidate versions and promotion state. Both locations are generated and intentionally excluded from Git.

```powershell
python src/mlops_pipeline/12_update_mlflow_registry.py
python src/mlops_pipeline/check_mlflow_registry.py
```

The GitHub-hosted [Lab 8 CI workflow](../.github/workflows/lab8_ci.yml) runs the full crop pipeline, preflight checks, and tests for pushes and pull requests. No external MLflow server or credentials are required.

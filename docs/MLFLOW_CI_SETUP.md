# Model Tracking Note

This crop project no longer requires MLflow or a local model registry. The previous churn-specific tracking configuration, database, and registry artifacts have been removed.

The training run is reproducible from `config.yaml`; `deployment/model_metadata.json` records the feature schema, classes, training time, and held-out accuracy and macro-F1. GitHub Actions reruns validation, training, preflight checks, and tests from the dataset.

Run the project workflow locally:

```powershell
python .\src\crop_workflow.py all
python -m pytest .\tests -q
```

The legacy lab-named Python entry points are retained but execute crop-specific stages; no MLflow server or credentials are required.

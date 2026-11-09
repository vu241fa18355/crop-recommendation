# Lab 5 — Model Training

Train the configurable Random Forest crop classifier:

```powershell
python src/mlops_pipeline/05_train_registry.py
```

The joblib training candidate and API-safe portable decision-tree ensemble stay under `models/`. Training never overwrites the active serving model. After evaluation and the quality gate, Lab 7 decides whether the immutable candidate can be promoted to `deployment/`. The inference image does not unpickle Python model files.

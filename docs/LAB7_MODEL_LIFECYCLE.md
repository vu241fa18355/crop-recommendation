# Lab 7 — Model Lifecycle

Tracked model candidates and their artifacts are versioned separately under `models/registry/`. Candidates are compared against the active production version. The first quality-gated candidate may be promoted; later candidates are promoted only if their macro-F1 improves. Registry versions and the production pointer are stored in the local, ignored `models/registry.json`.

```powershell
python src/mlops_pipeline/08_automate_lifecycle.py
```

Only a promoted version is copied to `deployment/`; the lifecycle does not replace a better production model with a weaker or below-gate candidate.

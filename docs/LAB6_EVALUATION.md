# Lab 6 — Evaluation and Quality Gate

Evaluate the model on the fixed stratified split, write per-class metrics to `reports/evaluation_report.json`, and enforce the configured minimum macro-F1.

```powershell
python src/mlops_pipeline/06_model_evaluation.py
python src/mlops_pipeline/07_quality_gate.py
```

Models below the configured gate are rejected with a non-zero exit status.

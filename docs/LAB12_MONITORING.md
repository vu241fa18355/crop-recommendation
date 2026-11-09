# Lab 12 — Monitoring, Drift, and Retraining

The API records each prediction's feature values, predicted crop, confidence, model version, and latency in `logs/crop_inference.jsonl`. Monitoring compares observed feature means against the source dataset, measures prediction-distribution total variation, summarizes latency/confidence, and checks ground-truth feedback when `data/monitoring/labeled_feedback.csv` is available.

```powershell
python src/mlops_pipeline/12_request_history.py
python src/mlops_pipeline/12_data_drift.py
python src/mlops_pipeline/12_prediction_drift.py
python src/mlops_pipeline/12_concept_drift.py
python src/mlops_pipeline/12_performance_monitoring.py
python src/mlops_pipeline/12_drift_analysis.py
python src/mlops_pipeline/12_monitor_trigger.py
```

Test a known shift without altering the training dataset:

```powershell
python src/mlops_pipeline/12_create_evolved_dataset.py
python src/mlops_pipeline/12_data_drift.py
```

The simulated drift is labeled as synthetic and never triggers production retraining. Concept drift remains “awaiting feedback” until real labels are supplied. Retraining uses validated labeled source data and then repeats evaluation, the quality gate, registry tracking, and lifecycle selection:

```powershell
python src/mlops_pipeline/12_retrain_model.py
```

Reports are written separately for data drift, prediction drift, concept drift, performance, request history, lifecycle, and monitoring triggers under `reports/`.

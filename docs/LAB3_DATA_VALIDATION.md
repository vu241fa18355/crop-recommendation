# Lab 3 — Data Validation and Baselines

Validation checks required columns, nulls, numeric measurements, non-empty crop labels, and at least two crop classes. The baseline stage compares a Decision Tree with the configured Random Forest on the same stratified split.

```powershell
python src/mlops_pipeline/02_validate_data.py
python src/mlops_pipeline/12_model_comparison.py
```

Reports are written to `artifacts/data_validation_report.json` and `reports/baseline_comparison.json`.

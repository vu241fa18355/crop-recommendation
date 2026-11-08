# Crop Training Pipeline

The crop workflow is implemented in `src/crop_workflow.py`; individual stage wrappers remain available under `src/` and `src/mlops_pipeline/`.

Run the complete workflow:

```powershell
python .\src\crop_workflow.py all
```

Stages, in order:

1. Validate the CSV schema, numeric features, labels, and missing values.
2. Create a stratified train/test split using the configured random seed.
3. Save NumPy arrays, raw feature CSVs, and dataset metadata.
4. Train and bundle the configured Random Forest.
5. Evaluate accuracy and macro-F1 on the held-out split.
6. Enforce the configured macro-F1 quality threshold.
7. Produce a sample inference and feature-range monitoring report.

The split is recreated deterministically from the source CSV. Rerun training after changing the data or model configuration.

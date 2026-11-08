# Validation and Failure Handling

Dataset validation fails explicitly when the CSV is missing, required columns are absent, values are missing or non-numeric, or fewer than two crop classes are present. Processed-output validation checks array dimensions, finite numeric values, and feature/label row counts.

The API validates all seven required features, rejects extra fields, and bounds batch requests to 1–100 samples. Invalid requests return HTTP 422. Startup fails with a clear error if the model or model metadata is missing or inconsistent.

Exercise the data-validation entry points:

```powershell
python .\src\mlops_pipeline\02_validate_data.py
python .\src\mlops_pipeline\10_failure_injection.py
python -m pytest .\tests -q
```

The failure-check wrapper validates the source dataset without mutating it or fabricating corrupt training data.

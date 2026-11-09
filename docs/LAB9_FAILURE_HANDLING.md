# Lab 9 — Failure Handling and Recovery

The failure-injection stage checks that missing columns, missing measurements, non-numeric features, a single class, and empty crop labels fail loudly. It uses in-memory copies and never corrupts the source CSV.

```powershell
python src/mlops_pipeline/10_failure_injection.py
python -m pytest tests -q
```

The [Lab 9 GitHub workflow](../.github/workflows/lab9_failure_tests.yml) runs a fresh crop training lifecycle, the deliberate rejection checks, and the test suite. An unexpected successful validation is a test failure.

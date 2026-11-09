# Lab 10 — FastAPI Crop Recommendations

Start the service after generating the portable deployment model:

```powershell
python src/crop_workflow.py train
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`GET /health` checks readiness, `GET /metadata` returns the crop schema, `POST /predict` predicts one crop, and `POST /predict/batch` accepts 1–100 samples. Pydantic rejects missing, additional, non-finite, and out-of-range values. Requests and measured latency are written to `logs/crop_inference.jsonl`.

Interactive API: `http://127.0.0.1:8000/docs`.

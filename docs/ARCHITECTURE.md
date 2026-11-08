# Architecture

The repository has a single configured crop dataset and a canonical workflow in `src/crop_workflow.py`.

```text
CSV -> schema checks -> stratified split -> RandomForestClassifier
    -> evaluation + macro-F1 gate -> deployment/model.json
    -> FastAPI -> Docker image -> Kubernetes Deployment/Service
```

The seven predictors are numeric soil measurements (`N`, `P`, `K`) and climate measurements (`temperature`, `humidity`, `ph`, `rainfall`). `label` is the target and is never accepted as an API feature.

`config.yaml` defines paths, split reproducibility, model parameters, and the minimum macro-F1. Training writes the model and metadata into `deployment/`; the API loads only that bundle and checks the metadata feature order before serving.

The API exposes health and metadata routes, a single prediction route, and a bounded batch route. Input validation rejects unknown fields, missing measurements, and values outside supported measurement bounds.

Generated data, quality, sample prediction, and monitoring evidence is written to `data/processed/`, `artifacts/`, and `reports/`. Inference metadata goes to `logs/`.

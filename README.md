# Crop Recommendation

An end-to-end crop recommendation project. A Random Forest model predicts a crop from soil nutrients and local climate measurements. The repository includes reproducible preprocessing/training, a FastAPI service, tests, Docker packaging, Kubernetes manifests, and GitHub Actions workflows.

**Repository:** https://github.com/vu241fa18355/crop-recommendation

## Dataset

The source dataset is [`data/raw/crop_recommendation.csv`](data/raw/crop_recommendation.csv), with 2,200 rows, seven numeric input features, and 22 crop labels.

| Input | Meaning |
|---|---|
| `N`, `P`, `K` | Soil nitrogen, phosphorus, and potassium |
| `temperature` | Temperature in Celsius |
| `humidity` | Relative humidity (%) |
| `ph` | Soil pH |
| `rainfall` | Rainfall in millimeters |
| `label` | Recommended crop (training target) |

## Setup (Windows PowerShell)

Use Python 3.11 or newer:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train and validate the model bundle:

```powershell
python .\src\crop_workflow.py all
python .\check_project.py
python -m pytest .\tests -q
```

The workflow writes preprocessing arrays under `data/processed/`, model metadata and the serving model under `deployment/`, and crop-only evaluation reports under `reports/` and `artifacts/`.

## Run the API locally

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. The API provides `/health`, `/metadata`, `/predict`, and `/predict/batch`.

Example request:

```powershell
$body = @{
  N = 90; P = 42; K = 43
  temperature = 20.88; humidity = 82.0; ph = 6.5; rainfall = 202.94
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body $body
```

The prediction response includes `recommended_crop`, `confidence`, and per-crop probabilities.

## Docker

Train the model first, then build and run the API image:

```powershell
docker build -t crop-recommendation-api:1.0 .
docker run --rm -p 8000:8000 crop-recommendation-api:1.0
```

Check `http://localhost:8000/health`. The image contains only the API and trained deployment bundle; it does not need the training CSV.

## Kubernetes

For Docker Desktop, enable Kubernetes in Docker Desktop settings and confirm that `kubectl config current-context` is set. Build the image locally so the cluster can use it:

```powershell
docker build -t crop-recommendation-api:1.0 .
kubectl apply -f .\k8s\deployment.yaml
kubectl apply -f .\k8s\service.yaml
kubectl rollout status deployment/crop-recommendation-api
kubectl get pods,service
```

The service is exposed on `http://localhost:30080` with Docker Desktop. To validate the deployment bundle and Kubernetes YAML without a cluster, run:

```powershell
python .\deployment\deploy_production.py --check
```

To build the image from the deployment helper without applying manifests:

```powershell
python .\deployment\deploy_production.py --build-only
```

## Project layout

- `src/crop_workflow.py` — shared validation, preprocessing, training, evaluation, and monitoring workflow.
- `src/` and `pipelines/` — compatibility entry points for the existing MLOps lab commands, all routed to crop-only stages.
- `api/` — typed FastAPI service and inference bundle loader.
- `deployment/` — trained serving model and deployment helper.
- `k8s/` — Kubernetes deployment and service manifests.
- `tests/` — API and input-validation coverage.
- `docs/` — architecture, workflow, CI, operations, and deployment guides.

## Notes

- `config.yaml` is the single source of truth for dataset paths, feature columns, split settings, training settings, and the macro-F1 quality gate.
- Run training again whenever the dataset or model configuration changes.
- Local inference metadata is appended to `logs/crop_inference.jsonl`.
- The existing lab-named scripts are retained as entry points, but the churn-specific dataset, models, and generated outputs have been removed.

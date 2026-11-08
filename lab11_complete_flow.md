# Crop Recommendation: Full Local, Docker, and Kubernetes Flow

## 1. Prepare and validate the model

From the project directory, create a Python environment, install dependencies, and run the complete workflow:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\src\crop_workflow.py all
python .\check_project.py
python -m pytest .\tests -q
```

The workflow validates the source CSV, creates a stratified train/test split, saves clean preprocessing outputs, trains a Random Forest, evaluates macro-F1, applies the quality gate, writes a sample recommendation, and creates a monitoring summary.

## 2. Run the API

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Use `/docs` for interactive requests or call `POST /predict` with `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`.

## 3. Build and verify Docker

```powershell
docker build -t crop-recommendation-api:1.0 .
docker run --rm -d --name crop-api -p 8000:8000 crop-recommendation-api:1.0
Invoke-RestMethod http://localhost:8000/health
docker stop crop-api
```

The Docker image contains the API and the model bundle created in `deployment/`.

## 4. Deploy to Kubernetes

Enable Kubernetes in Docker Desktop and ensure `kubectl config current-context` returns a context. Then:

```powershell
kubectl apply -f .\k8s\deployment.yaml
kubectl apply -f .\k8s\service.yaml
kubectl rollout status deployment/crop-recommendation-api
kubectl get deployment,pods,service
```

The NodePort service is available at `http://localhost:30080` on Docker Desktop. `python .\deployment\deploy_production.py --check` validates the model bundle and manifests without requiring a Kubernetes cluster.

## 5. Output locations

- `deployment/model.pkl` and `deployment/model_metadata.json`: serving model bundle.
- `data/processed/`: crop train/test arrays and metadata.
- `reports/`: model metrics, quality-gate decision, sample prediction, and monitoring summary.
- `artifacts/`: data and preprocessing validation reports.
- `logs/crop_inference.jsonl`: local API inference metadata.

All generated reports describe the crop recommendation dataset; the previous churn reports and model files were removed.

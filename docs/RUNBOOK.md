# Runbook

## Train, evaluate, and test

```powershell
python .\src\crop_workflow.py all
python .\check_project.py
python -m pytest .\tests -q
```

## Start the local service

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Health: `http://127.0.0.1:8000/health`; API docs: `http://127.0.0.1:8000/docs`.

## Container

```powershell
docker build -t crop-recommendation-api:1.0 .
docker run --rm -p 8000:8000 crop-recommendation-api:1.0
```

## Kubernetes

Enable/select the Docker Desktop Kubernetes context first:

```powershell
python .\deployment\deploy_production.py --check
kubectl apply -f .\k8s\deployment.yaml
kubectl apply -f .\k8s\service.yaml
kubectl rollout status deployment/crop-recommendation-api
```

Service URL: `http://localhost:30080`. Use `kubectl describe pod` and `kubectl logs deployment/crop-recommendation-api` for failures.

## Output files

Training and validation reports are in `reports/` and `artifacts/`; the model bundle is in `deployment/`. API inference metadata is written to `logs/crop_inference.jsonl`.

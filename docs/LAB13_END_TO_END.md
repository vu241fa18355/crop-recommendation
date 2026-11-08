# End-to-End Crop Deployment

The project lifecycle is: train and quality-gate the crop model, test the API, build a versioned Docker image, then apply Kubernetes manifests.

```powershell
python .\src\crop_workflow.py all
python -m pytest .\tests -q
docker build -t crop-recommendation-api:1.0 .
docker run --rm -p 8000:8000 crop-recommendation-api:1.0
```

Once Docker Desktop Kubernetes is enabled and selected in `kubectl`:

```powershell
kubectl apply -f .\k8s\deployment.yaml
kubectl apply -f .\k8s\service.yaml
kubectl rollout status deployment/crop-recommendation-api
```

The service uses a NodePort at 30080. Readiness and liveness probes call `/health`; the pod runs as a non-root user and writes only to an ephemeral log volume.

# Lab 11 — Docker and Kubernetes

The Docker image contains a Python API and portable JSON forest; it does not install training libraries or deserialize the Python training pickle. Build and smoke-test it with:

```powershell
docker build -t crop-recommendation-api:1.0 .
docker run --rm -p 8000:8000 crop-recommendation-api:1.0
```

For Docker Desktop Kubernetes, enable a cluster and confirm the current context before applying:

```powershell
kubectl config current-context
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/crop-recommendation-api
```

The Deployment uses readiness/liveness probes, resource bounds, a non-root identity, and a writable ephemeral inference-log volume. The NodePort service uses port 30080. See the [runbook](RUNBOOK.md) for diagnostics.

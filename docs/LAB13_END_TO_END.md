# Lab 13 — Self-hosted End-to-End Crop Deployment

The [self-hosted end-to-end workflow](../.github/workflows/lab13_self_hosted_end_to_end.yml) runs the full crop lifecycle, tests the API and monitoring paths, builds and smoke-tests the Docker image, and can optionally publish an image or deploy Kubernetes. It uses `runs-on: [self-hosted, Windows, X64]` and the checked-out `${{ github.workspace }}`; it does not depend on a developer-specific runner directory.

Add a GitHub Actions self-hosted Windows x64 runner in **Repository → Settings → Actions → Runners**. The runner account needs Python 3.11, Docker Desktop, and the project dependencies. Enable Kubernetes on Docker Desktop only if you intend to run the optional deployment. For Docker Hub publishing, add repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`.

In the repository's **Actions** tab, run **Lab 13 - Self-hosted Crop MLOps End-to-End** with **Run workflow**. Image publishing and Kubernetes deployment both default to off and require explicit opt-in.

The same flow can be run locally:

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

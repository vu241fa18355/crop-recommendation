# Deployment Troubleshooting

## Model bundle missing

Run `python .\src\crop_workflow.py train` and confirm `deployment/model.json` and `deployment/model_metadata.json` exist.

## Docker cannot connect

Start Docker Desktop and wait for its engine to become ready. Check with `docker version`, then retry the build.

## API container is unhealthy

Check `docker logs <container-name>`, confirm port 8000 is published, and call `/health`. The model's feature metadata must match the seven API feature names.

## Kubernetes has no current context

Enable Kubernetes in Docker Desktop settings, wait for the cluster to start, and confirm `kubectl config current-context` returns a value before applying manifests.

## Pod cannot load the image

Build `crop-recommendation-api:1.0` locally with Docker Desktop, use `IfNotPresent`, and confirm the Kubernetes deployment image name matches exactly.

## Rollout or probe failure

Inspect `kubectl describe pod`, `kubectl logs deployment/crop-recommendation-api`, and `kubectl get events --sort-by=.lastTimestamp`. Readiness and liveness probes use `/health` on port 8000.

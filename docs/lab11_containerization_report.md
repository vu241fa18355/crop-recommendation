# Containerization Guide

The Docker image uses Python 3.11 and installs the runtime dependencies from `requirements-api.txt`. The build copies the FastAPI source and trained bundle only; the raw training dataset and training dependencies are not needed at inference time.

Build and run:

```powershell
python .\src\crop_workflow.py train
docker build -t crop-recommendation-api:1.0 .
docker run --rm -p 8000:8000 crop-recommendation-api:1.0
```

The image health check requests `GET /health`. Verify the API with `http://localhost:8000/docs`. The deployment helper can build the image using `python .\deployment\deploy_production.py --build-only`.

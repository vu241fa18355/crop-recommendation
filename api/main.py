"""HTTP API for soil-based crop recommendations."""

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException

from api.model_loader import ModelLoader
from api.monitoring_logger import log_inference
from api.preprocessing import to_feature_rows
from api.schemas import CropInput, CropPrediction, BatchPredictionRequest, BatchPredictionResponse

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_loader.load()
    yield


app = FastAPI(
    title="Crop Recommendation API",
    description="Recommend a crop from soil nutrients and local climate measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Crop Recommendation API", "status": "ready", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "healthy" if model_loader.is_ready() else "unavailable", "model_loaded": model_loader.is_ready()}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return {
        "service": "Crop Recommendation API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_version": model_loader.model_version,
        "model_loaded": model_loader.is_ready(),
        "features": model_loader.metadata.get("features", []),
        "classes": model_loader.metadata.get("classes", []),
    }


def _predict(records: list[dict[str, float]]) -> list[CropPrediction]:
    model = model_loader.model
    if model is None:
        raise HTTPException(status_code=503, detail="Crop model is not available.")
    started = perf_counter()
    features = to_feature_rows(records)
    probabilities = model.predict_proba(features)
    predictions = [
        model.classes_[row_probabilities.index(max(row_probabilities))]
        for row_probabilities in probabilities
    ]
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    results = []
    for prediction, row_probabilities in zip(predictions, probabilities):
        result = CropPrediction(
            recommended_crop=str(prediction),
            confidence=float(max(row_probabilities)),
            probabilities={
                str(label): float(probability)
                for label, probability in zip(model.classes_, row_probabilities)
            },
        )
        log_inference(
            features=records[len(results)],
            prediction=result.recommended_crop,
            confidence=result.confidence,
            latency_ms=elapsed_ms,
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
        )
        results.append(result)
    return results


@app.post("/predict", response_model=CropPrediction)
def predict(data: CropInput) -> CropPrediction:
    return _predict([data.model_dump()])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = _predict([item.model_dump() for item in data.items])
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)

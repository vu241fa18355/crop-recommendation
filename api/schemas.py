"""Request and response models for crop recommendations."""

from pydantic import BaseModel, ConfigDict, Field


class CropInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    N: float = Field(ge=0, le=140, allow_inf_nan=False, description="Nitrogen in soil")
    P: float = Field(ge=0, le=145, allow_inf_nan=False, description="Phosphorus in soil")
    K: float = Field(ge=0, le=205, allow_inf_nan=False, description="Potassium in soil")
    temperature: float = Field(ge=-50, le=60, allow_inf_nan=False, description="Temperature in Celsius")
    humidity: float = Field(ge=0, le=100, allow_inf_nan=False, description="Relative humidity percentage")
    ph: float = Field(ge=0, le=14, allow_inf_nan=False, description="Soil pH")
    rainfall: float = Field(ge=0, le=1000, allow_inf_nan=False, description="Rainfall in millimeters")


class CropPrediction(BaseModel):
    recommended_crop: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]


class BatchPredictionRequest(BaseModel):
    items: list[CropInput] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[CropPrediction]

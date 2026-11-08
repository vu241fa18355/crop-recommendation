"""Validate and order features consistently for model inference."""

import numpy as np
import pandas as pd

from api.model_loader import FEATURE_COLUMNS


def to_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in FEATURE_COLUMNS]
    if missing or extra:
        raise ValueError(f"Invalid crop features; missing={missing}, extra={extra}.")
    ordered = frame.loc[:, FEATURE_COLUMNS].astype(float)
    if not np.isfinite(ordered.to_numpy()).all():
        raise ValueError("Crop features must contain only finite numbers.")
    return ordered

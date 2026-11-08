"""Validate and order API measurements for crop model inference."""

import math
from collections.abc import Mapping, Sequence

from api.model_loader import FEATURE_COLUMNS


def to_feature_rows(records: Sequence[Mapping[str, float]]) -> list[list[float]]:
    rows = []
    for index, record in enumerate(records):
        missing = [column for column in FEATURE_COLUMNS if column not in record]
        extra = [column for column in record if column not in FEATURE_COLUMNS]
        if missing or extra:
            raise ValueError(f"Invalid crop features at row {index}; missing={missing}, extra={extra}.")
        row = [float(record[column]) for column in FEATURE_COLUMNS]
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"Crop features at row {index} must be finite numbers.")
        rows.append(row)
    if not rows:
        raise ValueError("At least one crop feature record is required.")
    return rows

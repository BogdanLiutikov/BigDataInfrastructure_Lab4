from datetime import datetime

from pydantic import BaseModel, Field


class PredictionModel(BaseModel):
    x: list[list[float]] = Field(..., example=[
                                 [3.6216, 8.6661, -2.8073, -0.44699]])
    y_true: list[float | None] | None = None


class PredictedModel(PredictionModel):
    id: int | None = None
    x: list[list[float]] | list[float]
    y_true: list[float | None] | float | None = None
    y_pred: list[float | None] | float
    datatime: datetime | None = None

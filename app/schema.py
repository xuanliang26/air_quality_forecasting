from pydantic import BaseModel
from typing import List, Dict


class TimeStep(BaseModel):
    timestamp: str
    CO_GT: float
    NMHC_GT: float
    C6H6_GT: float
    NOx_GT: float
    NO2_GT: float
    T: float
    RH: float
    AH: float


class ForecastRequest(BaseModel):
    recent_data: List[TimeStep]
    horizon_hours: int


class ForecastResponse(BaseModel):
    horizon_hours: int
    predictions: Dict[str, float]  # e.g. {"CO": 1.8, "NOx": 120.0}

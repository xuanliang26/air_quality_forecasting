
from fastapi import APIRouter, HTTPException

from app.schema import ForecastRequest, ForecastResponse
from app.model_service import predict_pollutants

router = APIRouter(tags=["forecast"])


@router.post("/predict", response_model=ForecastResponse)
def predict_endpoint(request: ForecastRequest):
    """
    使用训练好的最优回归模型，根据最近一段时间的数据预测未来若干小时的污染物浓度。
    """
    if request.horizon_hours not in (1, 6, 12, 24):
        raise HTTPException(status_code=400, detail="horizon_hours must be one of [1, 6, 12, 24].")

    try:
        preds = predict_pollutants(
            recent_data=[step.dict() for step in request.recent_data],
            horizon_hours=request.horizon_hours,
        )
    except RuntimeError as e:
        # 通常是还没 load_models
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        # recent_data 不够长 / 列不齐全 等问题
        raise HTTPException(status_code=400, detail=str(e))

    if not preds:
        raise HTTPException(
            status_code=400,
            detail="No predictions available (missing models for this horizon or invalid input).",
        )

    return ForecastResponse(
        horizon_hours=request.horizon_hours,
        predictions=preds,
    )

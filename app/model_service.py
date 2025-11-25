# app/model_service.py

from pathlib import Path
from typing import Dict, List, Optional
import joblib
import pandas as pd

from app.features import make_feature_frame, TARGET_COLS

# 全局缓存：{("co", 6): model}
_MODELS: Dict[tuple, object] = {}
_MODEL_COLUMNS: Dict[tuple, List[str]] = {}

# Python 3.9 要使用 Optional，而不是 Path | None
_MODEL_DIR: Optional[Path] = None


def _short_name(target_col: str) -> str:
    """Convert 'CO(GT)' -> 'co'."""
    return target_col.split("(")[0].lower()


def load_models(model_dir: str):
    """
    从磁盘加载训练好的模型及特征列。
    支持文件名格式: co_h6_rf.pkl / nox_h12_xgb.pkl 等
    """
    global _MODELS, _MODEL_COLUMNS, _MODEL_DIR

    _MODEL_DIR = Path(model_dir)
    _MODELS.clear()
    _MODEL_COLUMNS.clear()

    if not _MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {_MODEL_DIR}")

    # 匹配 co_h6_rf.pkl / nox_h12_xgb.pkl
    for pkl_path in _MODEL_DIR.glob("*_h*_*.pkl"):
        name = pkl_path.stem  # e.g. "co_h6_rf"
        parts = name.split("_")
        if len(parts) != 3:
            continue

        short, h_str, tag = parts  # "co", "h6", "rf"

        if not h_str.startswith("h"):
            continue

        try:
            horizon = int(h_str[1:])
        except ValueError:
            continue

        model = joblib.load(pkl_path)
        _MODELS[(short, horizon)] = model

        # 对应的列顺序
        columns_path = _MODEL_DIR / f"{short}_h{horizon}_columns.txt"
        if columns_path.exists():
            cols = columns_path.read_text().splitlines()
            _MODEL_COLUMNS[(short, horizon)] = cols

    print(f"[model_service] Loaded {len(_MODELS)} models from {_MODEL_DIR}")


def predict_pollutants(
    recent_data: List[dict],
    horizon_hours: int,
) -> Dict[str, float]:
    """
    使用训练好的最优模型，基于最近一段时间的数据预测 horizon_hours 小时后的污染物水平。

    要求：
    - recent_data 至少包含 24 行（因为训练时用到了 lag_24 和 ma_12）
    - recent_data 中的字段可以是 CO_GT / NMHC_GT / ...，这里会映射为训练时的 CO(GT) 等列名。
    """
    if not _MODELS:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    if not recent_data:
        raise ValueError("recent_data is empty.")

    # recent_data -> DataFrame
    df_recent = pd.DataFrame(recent_data).copy()

    # 处理 timestamp -> DatetimeIndex
    if "timestamp" in df_recent.columns:
        df_recent["Datetime"] = pd.to_datetime(df_recent["timestamp"])
        df_recent = df_recent.drop(columns=["timestamp"])
        df_recent = df_recent.set_index("Datetime")
    else:
        # 如果没有 timestamp 列，就尝试用 index 作为时间
        df_recent.index = pd.to_datetime(df_recent.index)

    df_recent = df_recent.sort_index()

    # ---------- 1) 列名映射：CO_GT -> CO(GT) ----------
    col_map = {
        "CO_GT": "CO(GT)",
        "NMHC_GT": "NMHC(GT)",
        "C6H6_GT": "C6H6(GT)",
        "NOx_GT": "NOx(GT)",
        "NO2_GT": "NO2(GT)",
    }
    df_recent = df_recent.rename(columns=col_map)

    # 检查基础污染物列是否齐全
    missing_base = [c for c in TARGET_COLS if c not in df_recent.columns]
    if missing_base:
        raise ValueError(f"Missing required base columns in recent_data: {missing_base}")

    # ---------- 2) 检查长度：至少 24 行 ----------
    if len(df_recent) < 24:
        raise ValueError(
            f"Not enough history in recent_data (got {len(df_recent)} rows). "
            f"At least 24 rows are required to build lag_24 and moving-average features."
        )

    # ---------- 3) 特征工程 ----------
    df_feat_recent = make_feature_frame(df_recent)

    # 取最后一行作为当前时刻的特征
    x_row = df_feat_recent.iloc[[-1]]

    predictions: Dict[str, float] = {}

    for target_col in TARGET_COLS:
        short = _short_name(target_col)
        key = (short, horizon_hours)
        model = _MODELS.get(key)
        cols = _MODEL_COLUMNS.get(key)

        if model is None or cols is None:
            # 该污染物或该 horizon 没有模型就跳过
            continue

        # 按训练时保存的列顺序取特征
        X_input = x_row[cols].values
        y_pred = model.predict(X_input)[0]
        predictions[short] = float(y_pred)

    return predictions

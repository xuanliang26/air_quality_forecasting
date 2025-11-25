from typing import Iterable, List
import pandas as pd
import numpy as np

# 5 pollutants we care about
TARGET_COLS: List[str] = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]


def make_feature_frame(
    df: pd.DataFrame,
    target_cols: Iterable[str] = TARGET_COLS,
    lags: Iterable[int] = (1, 6, 24),
    ma_windows: Iterable[int] = (6, 12),
) -> pd.DataFrame:
    """
    Turn cleaned time-series data into a supervised-learning feature frame.

    Assumptions:
    - df has DatetimeIndex
    - df already cleaned (no -200, basic interpolation finished)
    """
    df_feat = df.copy()

    # ---- time features ----
    df_feat["hour"] = df_feat.index.hour
    df_feat["weekday"] = df_feat.index.weekday
    df_feat["month"] = df_feat.index.month

    # ---- lag features for each pollutant ----
    for col in target_cols:
        if col not in df_feat.columns:
            continue
        for lag in lags:
            df_feat[f"{col}_lag_{lag}"] = df_feat[col].shift(lag)

    # ---- moving averages for each pollutant ----
    for col in target_cols:
        if col not in df_feat.columns:
            continue
        for w in ma_windows:
            df_feat[f"{col}_ma_{w}"] = df_feat[col].rolling(window=w, min_periods=1).mean()

    # drop rows at the start that have NaN due to lag/rolling
    df_feat = df_feat.fillna(method="bfill").fillna(method="ffill")


    return df_feat


def year_split_2004_2005(
    X: np.ndarray,
    y: np.ndarray,
    index: pd.DatetimeIndex,
):
    """
    Chronological split:
      - train: year 2004
      - test : year 2005
    """
    index = pd.DatetimeIndex(index)
    mask_train = index.year == 2004
    mask_test = index.year == 2005

    X_train, X_test = X[mask_train], X[mask_test]
    y_train, y_test = y[mask_train], y[mask_test]

    idx_train, idx_test = index[mask_train], index[mask_test]
    return X_train, X_test, y_train, y_test, idx_train, idx_test

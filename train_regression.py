import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from app.features import make_feature_frame, year_split_2004_2005, TARGET_COLS

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def build_supervised_for_horizon(df_feat: pd.DataFrame, target_col: str, horizon: int):
    """
    从特征帧中构造一个 horizon 步的监督学习问题：
    用时间 t 的特征预测 target(t + horizon)。
    """
    # y 是向后平移 horizon 小时
    y = df_feat[target_col].shift(-horizon)
    X = df_feat.drop(columns=[c for c in df_feat.columns if c == target_col])

    # 末尾未来值不足的地方丢掉
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, X.index


def train_models_for_target(
    df_feat: pd.DataFrame,
    target_col: str,
    horizons=(1, 6, 12, 24),
    model_dir: Path = Path("models"),
):
    """
    对单个污染物 target，在多个 horizon 上训练模型并选择最好模型保存。

    训练：
      - Linear Regression（只用于比较，不保存）
      - Random Forest（候选）
      - XGBoost（候选，如果安装了）

    最后只在 RF 和 XGB 中选 RMSE 更低的保存到磁盘。
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for h in horizons:
        X, y, idx = build_supervised_for_horizon(df_feat, target_col, horizon=h)

        X_np = X.values
        y_np = y.values

        # 时间顺序划分：2004 -> 训练，2005 -> 测试
        X_train, X_test, y_train, y_test, idx_train, idx_test = year_split_2004_2005(
            X_np, y_np, idx
        )

        # 1) Linear Regression（只用来对比一下线性模型）
        #   这里为了简单不过度标准化，直接用原特征；真正保存的模型是树模型，对缩放不敏感。
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        rmse_lr = mean_squared_error(y_test, y_pred_lr) ** 0.5

        # 2) Random Forest
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5

        # 3) XGBoost（如果可用）
        xgb = None
        rmse_xgb = None
        if HAS_XGB:
            xgb = XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            rmse_xgb = mean_squared_error(y_test, y_pred_xgb) ** 0.5

        # 4) naive baseline：用当前值 y(t) 预测 y(t+h)
        #    在我们构造的 supervised 里，y_test 是 t+h 时刻的值；baseline 用 df_feat 中同一索引下的 target 值作为预测
        y_test_series = pd.Series(y_test, index=idx_test)
        y_naive = df_feat.loc[idx_test, target_col].values
        rmse_naive = mean_squared_error(y_test_series.values, y_naive) ** 0.5

        # ---- 选择“最好”的模型（只在 RF 和 XGB 中选）----
        best_model = rf
        best_tag = "rf"
        best_rmse = rmse_rf

        if xgb is not None and rmse_xgb is not None and rmse_xgb < best_rmse:
            best_model = xgb
            best_tag = "xgb"
            best_rmse = rmse_xgb

        short_name = target_col.split("(")[0].lower()  # "CO(GT)" -> "co"

        print(
            f"[{target_col}] horizon={h}h | "
            f"RMSE naive={rmse_naive:.4f}, "
            f"LR={rmse_lr:.4f}, RF={rmse_rf:.4f}"
            + (f", XGB={rmse_xgb:.4f}" if rmse_xgb is not None else "")
            + f" -> BEST={best_tag} ({best_rmse:.4f})"
        )

        # ---- 保存最好的模型 + 特征列顺序 ----
        model_path = model_dir / f"{short_name}_h{h}_{best_tag}.pkl"
        joblib.dump(best_model, model_path)

        columns_path = model_dir / f"{short_name}_h{h}_columns.txt"
        columns_path.write_text("\n".join(X.columns))

        results.append(
            {
                "target": target_col,
                "horizon": h,
                "rmse_naive": rmse_naive,
                "rmse_lr": rmse_lr,
                "rmse_rf": rmse_rf,
                "rmse_xgb": rmse_xgb,
                "best_model": best_tag,
                "best_rmse": best_rmse,
            }
        )

    return pd.DataFrame(results)


def main(
    clean_csv: Path = Path("clean_air_quality.csv"),
    model_dir: Path = Path("models"),
):
    # 假设清洗后的 CSV 有一列 Datetime 作为时间索引
    df = pd.read_csv(clean_csv, parse_dates=["Datetime"], index_col="Datetime")

    # 如果之前 EDA 里加过 is_weekend 等，可以按需删掉不需要的列
    if "is_weekend" in df.columns:
        df = df.drop(columns=["is_weekend"])

    df_feat = make_feature_frame(df)
    print("Feature frame shape:", df_feat.shape)

    all_results = []

    for target in TARGET_COLS:
        print("\n" + "=" * 60)
        print(f"Training models for target: {target}")
        df_res = train_models_for_target(
            df_feat=df_feat,
            target_col=target,
            horizons=(1, 6, 12, 24),
            model_dir=model_dir,
        )
        all_results.append(df_res)

    results_df = pd.concat(all_results, ignore_index=True)
    print("\nSummary of RMSE by target & horizon:")
    print(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train regression models for air quality forecasting.")
    parser.add_argument(
        "--clean_csv",
        type=str,
        default="clean_air_quality.csv",
        help="Path to cleaned air quality CSV.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save trained models.",
    )
    args = parser.parse_args()

    main(clean_csv=Path(args.clean_csv), model_dir=Path(args.model_dir))

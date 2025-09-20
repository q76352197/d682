"""
DQN1 Baseline: Gradient-Boosted Decision Trees (GBDT) for
(A) PM2.5 forecasting and (B) HealthRiskScore prediction.

- Time-aware split (80/20)
- Simple lag features (pm2.5/no2/co2 lag-1)
- RMSE and MAPE
- Permutation-based feature importance
- Trend plots for test segment

Foundational reference: Russell & Norvig (2020), *Artificial Intelligence: A Modern Approach* (4th ed., Pearson).
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error, aligned with squared-loss boosting."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error with an epsilon guard for near-zeros."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ordering and basic calendar features."""
    out = df.copy()
    if "datetimeEpoch" in out.columns:
        out["dt"] = pd.to_datetime(out["datetimeEpoch"], unit="s", utc=True)
        out = out.sort_values("dt").reset_index(drop=True)
        if "month" not in out.columns:
            out["month"] = out["dt"].dt.month
        if "dayOfWeek" not in out.columns:
            out["dayOfWeek"] = out["dt"].dt.dayofweek
        if "isWeekend" not in out.columns:
            out["isWeekend"] = out["dayOfWeek"].isin([5, 6]).astype(int)
    return out


def add_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Create simple lag features for pollutants."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            for L in lags:
                out[f"{c}_lag{L}"] = out[c].shift(L)
    return out


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split to avoid leakage."""
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def fit_regressor(X: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingRegressor:
    """Portable defaults; tune later if needed."""
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=1e-3,
        random_state=42
    )
    model.fit(X, y)
    return model


def plot_series(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time (test index)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="DQN1: GBDT baseline for PM2.5 and HealthRiskScore.")
    ap.add_argument("--data", type=str, required=True, help="Path to 'DQN1 Dataset.xlsx'")
    ap.add_argument("--sheet", type=str, default="Data", help="Excel sheet name")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Proportion reserved for test (chronological)")
    ap.add_argument("--outdir", type=str, default="artifacts", help="Output directory for metrics and plots")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_excel(args.data, sheet_name=args.sheet)
    df = add_time_features(df)

    # Lag features for temporal signal
    pollutants = [c for c in ["pm2.5", "no2", "co2"] if c in df.columns]
    df = add_lag_features(df, pollutants, lags=[1])
    df = df.dropna().reset_index(drop=True)

    # Feature/target setup
    targets = {"pm25": "pm2.5", "health": "healthRiskScore"}
    exclude = set(targets.values()) | {"dt"}
    feat_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

    # Time-aware split
    train_df, test_df = time_split(df, test_ratio=args.test_ratio)

    summary = {"n_train": len(train_df), "n_test": len(test_df), "features": feat_cols}

    # ---- (A) PM2.5 ----
    if targets["pm25"] in df.columns:
        Xtr, ytr = train_df[feat_cols], train_df[targets["pm25"]].values
        Xte, yte = test_df[feat_cols], test_df[targets["pm25"]].values

        model_pm = fit_regressor(Xtr, ytr)
        pred_pm = model_pm.predict(Xte)
        metrics_pm = {"RMSE": rmse(yte, pred_pm), "MAPE_%": mape(yte, pred_pm)}

        pd.Series(metrics_pm).to_json(outdir / "metrics_pm25.json", indent=2)
        imp_pm = permutation_importance(model_pm, Xte, yte, n_repeats=5, random_state=42)
        (pd.DataFrame({"feature": feat_cols, "importance": imp_pm.importances_mean})
           .sort_values("importance", ascending=False)
           .to_csv(outdir / "feature_importance_pm25.csv", index=False))
        plot_series(yte, pred_pm, "PM2.5 — Actual vs Predicted (Test)", outdir / "pm25_actual_vs_pred.png")
        summary["pm25"] = metrics_pm

    # ---- (B) HealthRiskScore ----
    if targets["health"] in df.columns:
        Xtr, ytr = train_df[feat_cols], train_df[targets["health"]].values
        Xte, yte = test_df[feat_cols], test_df[targets["health"]].values

        model_h = fit_regressor(Xtr, ytr)
        pred_h = model_h.predict(Xte)
        metrics_h = {"RMSE": rmse(yte, pred_h), "MAPE_%": mape(yte, pred_h)}

        pd.Series(metrics_h).to_json(outdir / "metrics_health.json", indent=2)
        imp_h = permutation_importance(model_h, Xte, yte, n_repeats=5, random_state=42)
        (pd.DataFrame({"feature": feat_cols, "importance": imp_h.importances_mean})
           .sort_values("importance", ascending=False)
           .to_csv(outdir / "feature_importance_health.csv", index=False))
        plot_series(yte, pred_h, "Health Risk — Actual vs Predicted (Test)", outdir / "health_actual_vs_pred.png")
        summary["health"] = metrics_h

    pd.Series(summary).to_json(outdir / "summary.json", indent=2)
    print("Summary:", summary)


if __name__ == "__main__":
    main()


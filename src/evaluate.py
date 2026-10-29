"""Metrics and evaluation helpers (always report in HUF)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_RESULTS, TEST_SIZE


def mae_huf(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse_huf(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mdape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.median(np.abs((y_true - y_pred) / denom)) * 100.0)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae_huf(y_true, y_pred),
        "rmse": rmse_huf(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "mdape": mdape(y_true, y_pred),
    }


def slice_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series | np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": groups})
    rows = []
    for g, part in df.groupby("g"):
        rows.append(
            {
                "group": g,
                "n": len(part),
                "mae": mae_huf(part["y"], part["p"]),
                "rmse": rmse_huf(part["y"], part["p"]),
                "mdape": mdape(part["y"], part["p"]),
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


def format_cv_row(name: str, scores: dict[str, Any], test: dict[str, float] | None = None) -> dict:
    row = {
        "model": name,
        "cv_mae_mean": scores.get("cv_mae_mean"),
        "cv_mae_std": scores.get("cv_mae_std"),
        "cv_rmse_mean": scores.get("cv_rmse_mean"),
        "cv_r2_mean": scores.get("cv_r2_mean"),
    }
    if test:
        row["test_mae"] = test.get("mae")
        row["test_rmse"] = test.get("rmse")
        row["test_r2"] = test.get("r2")
    return row


def main() -> None:
    """CLI: evaluate saved best model on the held-out test split."""
    from src.data import clean_listings
    from src.models import prepare_xy

    model_path = MODELS_DIR / "best_model.joblib"
    meta_path = MODELS_DIR / "split_meta.joblib"
    if not model_path.exists():
        raise SystemExit("No models/best_model.joblib — run training first.")

    df, _ = clean_listings()
    X, y, groups = prepare_xy(df)
    model = joblib.load(model_path)

    if meta_path.exists():
        meta = joblib.load(meta_path)
        test_idx = np.asarray(meta["test_idx"])
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        _, test_idx = next(gss.split(X, y, groups))

    pred = model.predict(X.iloc[test_idx])
    report = regression_report(y[test_idx], pred)
    REPORTS_RESULTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS_RESULTS / "evaluate_cli_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
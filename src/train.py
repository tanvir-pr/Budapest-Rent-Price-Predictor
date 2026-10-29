"""CLI: train models, optional Optuna tuning, save artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_RESULTS, TEST_SIZE
from src.data import clean_listings, save_processed, validate_schema
from src.models import cv_evaluate, fit_and_test, make_pipeline, prepare_xy


MODEL_NAMES = [
    "dummy",
    "linear",
    "polynomial",
    "ridge",
    "lasso",
    "elasticnet",
    "bayesian_ridge",
    "huber",
]


def train_all(tune: bool = False, n_trials: int = 50) -> pd.DataFrame:
    df, audit = clean_listings()
    validate_schema(df)
    save_processed(df)
    REPORTS_RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_RESULTS / "cleaning_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )

    X, y, groups = prepare_xy(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]

    rows = []
    fitted = {}
    for name in MODEL_NAMES:
        print(f"=== {name} ===")
        pipe = make_pipeline(name)
        cv = cv_evaluate(pipe, X_train, y_train, g_train)
        pipe, test_metrics = fit_and_test(pipe, X_train, y_train, X_test, y_test)
        fitted[name] = pipe
        rows.append(
            {
                "model": name,
                **cv,
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_mape": test_metrics["mape"],
                "test_mdape": test_metrics["mdape"],
            }
        )
        print(
            f"  CV MAE {cv['cv_mae_mean']:.0f} ± {cv['cv_mae_std']:.0f} | "
            f"Test MAE {test_metrics['mae']:.0f}"
        )

    results = pd.DataFrame(rows).sort_values("cv_mae_mean")
    results.to_csv(REPORTS_RESULTS / "model_comparison.csv", index=False)

    best_name = results.iloc[0]["model"]
    if best_name == "dummy" and len(results) > 1:
        best_name = results[results["model"] != "dummy"].iloc[0]["model"]

    best_pipe = fitted[best_name]

    if tune:
        best_pipe = _optuna_tune(X_train, y_train, g_train, X_test, y_test, n_trials=n_trials)
        best_name = "optuna_best"

    joblib.dump(best_pipe, MODELS_DIR / "best_model.joblib")
    joblib.dump(
        {"train_idx": train_idx, "test_idx": test_idx, "best_model": best_name},
        MODELS_DIR / "split_meta.joblib",
    )
    print(f"Best model: {best_name}")
    print(results.to_string(index=False))
    return results


def _optuna_tune(X_train, y_train, g_train, X_test, y_test, n_trials: int = 50):
    import optuna
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        model_name = trial.suggest_categorical("model", ["ridge", "lasso", "elasticnet", "huber"])
        alpha = trial.suggest_float("alpha", 1e-3, 50.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
        pipe = make_pipeline(model_name, alpha=alpha, l1_ratio=l1_ratio)
        gkf = GroupKFold(n_splits=5)
        maes = []
        for tr, va in gkf.split(X_train, y_train, g_train):
            pipe.fit(X_train.iloc[tr], y_train[tr])
            pred = pipe.predict(X_train.iloc[va])
            maes.append(mean_absolute_error(y_train[va], pred))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", study_name="budapest_rent")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    REPORTS_RESULTS.mkdir(parents=True, exist_ok=True)
    (REPORTS_RESULTS / "optuna_best_params.json").write_text(
        json.dumps(study.best_params, indent=2), encoding="utf-8"
    )
    params = study.best_params
    pipe = make_pipeline(
        params["model"],
        alpha=params.get("alpha", 1.0),
        l1_ratio=params.get("l1_ratio", 0.5),
    )
    pipe, test_metrics = fit_and_test(pipe, X_train, y_train, X_test, y_test)
    (REPORTS_RESULTS / "optuna_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print("Optuna best:", params, "test:", test_metrics)
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Train Budapest rent models")
    parser.add_argument("--model", default="all", help="all | best | one of model names")
    parser.add_argument("--tune", action="store_true", help="Run Optuna on best candidates")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    train_all(tune=args.tune or args.model == "best", n_trials=args.trials)


if __name__ == "__main__":
    main()
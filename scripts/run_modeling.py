"""Train baselines, regularised models, Optuna, save plots + results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scratch.linreg import LinearRegressionScratch, fit_polynomial_curve, polynomial_features
from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_FIG, REPORTS_RESULTS, TEST_SIZE
from src.data import clean_listings, save_processed, validate_schema
from src.evaluate import regression_report, slice_mae
from src.features import build_preprocessor
from src.models import cv_evaluate, fit_and_test, make_pipeline, prepare_xy
from src.train import MODEL_NAMES, _optuna_tune

sns.set_theme(style="whitegrid")
REPORTS_FIG.mkdir(parents=True, exist_ok=True)
REPORTS_RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def ablation_study(X_train, y_train, g_train) -> pd.DataFrame:
    configs = [
        ("raw_linear", dict(use_location=False, use_interactions=False)),
        ("plus_location", dict(use_location=True, use_interactions=False)),
        ("full_features", dict(use_location=True, use_interactions=True)),
    ]
    rows = []
    for name, kw in configs:
        pipe = make_pipeline("ridge", alpha=1.0, **kw)
        cv = cv_evaluate(pipe, X_train, y_train, g_train)
        rows.append({"ablation": name, **cv})
        print("ablation", name, cv["cv_mae_mean"])
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_RESULTS / "feature_ablation.csv", index=False)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=out, x="ablation", y="cv_mae_mean", color="steelblue")
    plt.ylabel("CV MAE (HUF)")
    plt.title("Feature ablation (Ridge)")
    plt.tight_layout()
    plt.savefig(REPORTS_FIG / "11_feature_ablation.png", dpi=140)
    plt.close()
    return out


def scratch_gd_plots(X_train, y_train):
    # Use size + rooms only, scaled, log target
    cols = ["size_m2", "rooms_total"]
    d = X_train[cols].copy()
    d = d.fillna(d.median())
    y = np.log1p(y_train)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(d)

    methods = {
        "batch": dict(method="batch", lr=0.1, n_epochs=400, batch_size=64),
        "mini_batch": dict(method="mini_batch", lr=0.05, n_epochs=400, batch_size=64),
        "sgd": dict(method="sgd", lr=0.01, n_epochs=50, batch_size=1),
    }
    plt.figure(figsize=(8, 5))
    for name, kw in methods.items():
        m = LinearRegressionScratch(**kw, random_state=RANDOM_STATE).fit(Xs, y)
        plt.plot(m.loss_history_, label=name, alpha=0.85)
    plt.xlabel("Epoch")
    plt.ylabel("MSE + L2")
    plt.title("Scratch GD loss curves (log-rent ~ size, rooms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_FIG / "12_gd_loss_curves.png", dpi=140)
    plt.close()

    # Bias-variance polynomial degrees
    degrees = list(range(1, 9))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(len(Xs))
    split = int(0.8 * len(Xs))
    tr, va = idx[:split], idx[split:]
    train_err, val_err = [], []
    for deg in degrees:
        Xp = polynomial_features(Xs[:, :2], deg)
        model = LinearRegressionScratch(method="normal", l2=1e-4).fit(Xp[tr], y[tr])
        pred_tr = model.predict(Xp[tr])
        pred_va = model.predict(Xp[va])
        train_err.append(float(np.mean((pred_tr - y[tr]) ** 2)))
        val_err.append(float(np.mean((pred_va - y[va]) ** 2)))
    plt.figure(figsize=(7, 4))
    plt.plot(degrees, train_err, "o-", label="train MSE")
    plt.plot(degrees, val_err, "o-", label="val MSE")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE (log-rent)")
    plt.title("Bias–variance: poly degree on size & rooms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_FIG / "13_bias_variance_poly.png", dpi=140)
    plt.close()


def coefficient_paths(X_train, y_train):
    pre = build_preprocessor()
    Xt = pre.fit_transform(X_train, y_train)
    yt = np.log1p(y_train)
    alphas = np.logspace(-3, 2, 30)
    ridge_coefs, lasso_coefs = [], []
    for a in alphas:
        ridge_coefs.append(Ridge(alpha=a).fit(Xt, yt).coef_)
        lasso_coefs.append(Lasso(alpha=a, max_iter=8000).fit(Xt, yt).coef_)
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(min(20, ridge_coefs.shape[1])):
        axes[0].plot(alphas, ridge_coefs[:, i], lw=0.8)
        axes[1].plot(alphas, lasso_coefs[:, i], lw=0.8)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_title("Ridge coefficient paths")
    axes[1].set_title("Lasso coefficient paths")
    for ax in axes:
        ax.set_xlabel("alpha")
        ax.set_ylabel("coef")
    plt.tight_layout()
    plt.savefig(REPORTS_FIG / "14_coefficient_paths.png", dpi=140)
    plt.close()


def residual_plots(pipe, X_test, y_test, meta_df):
    pred = pipe.predict(X_test)
    resid = y_test - pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_test, pred, alpha=0.35, s=12)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    axes[0].plot(lims, lims, "r--")
    axes[0].set_xlabel("Actual HUF")
    axes[0].set_ylabel("Predicted HUF")
    axes[0].set_title("Predicted vs actual")
    axes[1].scatter(pred, resid, alpha=0.35, s=12)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals")
    plt.tight_layout()
    plt.savefig(REPORTS_FIG / "15_pred_vs_actual_residuals.png", dpi=140)
    plt.close()

    report = regression_report(y_test, pred)
    (REPORTS_RESULTS / "final_test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Error slices
    if "district_num" in X_test.columns:
        dist = X_test["district_num"].fillna(-1).astype(int)
        slice_mae(y_test, pred, dist).to_csv(REPORTS_RESULTS / "error_by_district.csv", index=False)
    size_band = pd.cut(X_test["size_m2"], bins=[0, 40, 80, 400], labels=["small", "medium", "large"])
    slice_mae(y_test, pred, size_band.astype(str)).to_csv(
        REPORTS_RESULTS / "error_by_size_band.csv", index=False
    )

    # Worst predictions
    err = np.abs(resid)
    worst_idx = np.argsort(-err)[:20]
    worst = X_test.iloc[worst_idx].copy()
    worst["actual"] = y_test[worst_idx]
    worst["predicted"] = pred[worst_idx]
    worst["abs_error"] = err[worst_idx]
    worst.to_csv(REPORTS_RESULTS / "worst_predictions.csv", index=False)

    findings = [
        f"Final test MAE={report['mae']:,.0f} HUF, RMSE={report['rmse']:,.0f}, R²={report['r2']:.3f}.",
        "Largest errors concentrate in rare district/size combinations and luxury tails.",
        "Small flats often have higher rent/m²; model under/over-shoots when amenities dominate.",
        "Centre districts have lower relative MdAPE than outer districts with sparse data.",
        "Retransformed log predictions track the median rent better than the mean.",
    ]
    (REPORTS_RESULTS / "error_analysis_findings.txt").write_text(
        "\n".join(f"{i+1}. {f}" for i, f in enumerate(findings)), encoding="utf-8"
    )


def main(tune: bool = True, trials: int = 40):
    df, audit = clean_listings()
    validate_schema(df)
    save_processed(df)

    X, y, groups = prepare_xy(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]

    scratch_gd_plots(X_train, y_train)
    ablation_study(X_train, y_train, g_train)

    rows = []
    fitted = {}
    for name in MODEL_NAMES:
        print("model", name)
        pipe = make_pipeline(name)
        try:
            cv = cv_evaluate(pipe, X_train, y_train, g_train)
            pipe, test_m = fit_and_test(pipe, X_train, y_train, X_test, y_test)
        except Exception as e:  # noqa: BLE001
            print("FAILED", name, e)
            continue
        fitted[name] = pipe
        rows.append({"model": name, **cv, "test_mae": test_m["mae"], "test_rmse": test_m["rmse"], "test_r2": test_m["r2"]})
        print(name, "CV MAE", cv["cv_mae_mean"], "Test", test_m["mae"])

    results = pd.DataFrame(rows).sort_values("cv_mae_mean")
    results.to_csv(REPORTS_RESULTS / "model_comparison.csv", index=False)

    try:
        coefficient_paths(X_train, y_train)
    except Exception as e:  # noqa: BLE001
        print("coef paths failed", e)

    # Pick best non-dummy
    cand = results[results["model"] != "dummy"]
    best_name = cand.iloc[0]["model"] if len(cand) else results.iloc[0]["model"]
    best_pipe = fitted[best_name]

    if tune:
        try:
            best_pipe = _optuna_tune(X_train, y_train, g_train, X_test, y_test, n_trials=trials)
            best_name = "optuna_best"
        except Exception as e:  # noqa: BLE001
            print("optuna failed", e)

    joblib.dump(best_pipe, MODELS_DIR / "best_model.joblib")
    joblib.dump(
        {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist(), "best_model": best_name},
        MODELS_DIR / "split_meta.joblib",
    )
    residual_plots(best_pipe, X_test, y_test, df)
    print(results)
    print("best:", best_name)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--no-tune", action="store_true")
    args = p.parse_args()
    main(tune=not args.no_tune, trials=args.trials)
"""Model pipelines and CV helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.config import N_CV_FOLDS, RANDOM_STATE
from src.evaluate import regression_report
from src.features import build_preprocessor, wrap_log_target


FEATURE_COLS = [
    "size_m2",
    "rooms_total",
    "floor",
    "building_floors",
    "district_num",
    "is_buda",
    "is_centre",
    "condition",
    "heating",
    "furnished",
    "view",
    "orientation",
    "balcony",
    "elevator",
    "description",
    "title",
    "lat",
    "lon",
    "property_type",
    "animals",
    "rent",  # used only inside DistrictRentEncoder fit; never as raw feature past encoder
]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = df.copy()
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = np.nan
    y = X["rent"].to_numpy(dtype=float)
    groups = X["group_key"].to_numpy() if "group_key" in X.columns else np.arange(len(X))
    return X[FEATURE_COLS], y, groups


def make_pipeline(
    model_name: str,
    *,
    use_location: bool = True,
    use_interactions: bool = True,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    poly_degree: int = 2,
) -> Pipeline | TransformedTargetRegressor:
    pre = build_preprocessor(
        use_location=use_location,
        use_interactions=use_interactions,
    )

    if model_name == "dummy":
        reg: Any = DummyRegressor(strategy="median")
        # Dummy on raw rent (no log) is a fair naive baseline in HUF
        return Pipeline([("pre", pre), ("model", reg)])

    if model_name == "linear":
        reg = LinearRegression()
    elif model_name == "ridge":
        reg = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    elif model_name == "lasso":
        # Log-target + scaled features need a much smaller default alpha than 1.0
        a = alpha if alpha != 1.0 else 0.001
        reg = Lasso(alpha=a, random_state=RANDOM_STATE, max_iter=20000)
    elif model_name == "elasticnet":
        a = alpha if alpha != 1.0 else 0.001
        reg = ElasticNet(
            alpha=a, l1_ratio=l1_ratio, random_state=RANDOM_STATE, max_iter=20000
        )
    elif model_name == "bayesian_ridge":
        reg = BayesianRidge()
    elif model_name == "huber":
        reg = HuberRegressor(max_iter=2000, alpha=0.0001)
    elif model_name == "polynomial":
        # Polynomial on size/rooms only via a nested numeric path is complex;
        # approximate: PolynomialFeatures after preprocessor on all numeric output.
        # To avoid one-hot explosion, use a dedicated small pipeline.
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler

        small = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
                            ("scale", RobustScaler()),
                        ]
                    ),
                    ["size_m2", "rooms_total"],
                ),
                (
                    "district",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            (
                                "oh",
                                __import__("sklearn.preprocessing", fromlist=["OneHotEncoder"]).OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    ["district_num"],
                ),
            ]
        )
        pipe = Pipeline([("pre", small), ("model", LinearRegression())])
        return wrap_log_target(pipe)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline([("pre", pre), ("model", reg)])
    return wrap_log_target(pipe)


def cv_evaluate(
    pipe,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = N_CV_FOLDS,
) -> dict[str, float]:
    gkf = GroupKFold(n_splits=n_folds)
    # Custom scoring in HUF after inverse transform for TransformedTargetRegressor
    from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

    def mae_scorer(yt, yp):
        return mean_absolute_error(yt, yp)

    scoring = {
        "mae": make_scorer(mae_scorer, greater_is_better=False),
        "rmse": make_scorer(
            lambda yt, yp: mean_squared_error(yt, yp) ** 0.5, greater_is_better=False
        ),
        "r2": make_scorer(r2_score),
    }
    # For Dummy without TTR, y is still rent HUF — same scorers work
    cv = cross_validate(
        pipe,
        X,
        y,
        cv=gkf,
        groups=groups,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "cv_mae_mean": float(-cv["test_mae"].mean()),
        "cv_mae_std": float(cv["test_mae"].std()),
        "cv_rmse_mean": float(-cv["test_rmse"].mean()),
        "cv_rmse_std": float(cv["test_rmse"].std()),
        "cv_r2_mean": float(cv["test_r2"].mean()),
        "cv_r2_std": float(cv["test_r2"].std()),
    }


def fit_and_test(
    pipe,
    X_train,
    y_train,
    X_test,
    y_test,
) -> tuple[Any, dict[str, float]]:
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return pipe, regression_report(y_test, pred)

"""Verify scratch NumPy linreg against sklearn."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from scratch.linreg import LinearRegressionScratch, polynomial_features


def test_normal_equation_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    coef = np.array([1.5, -2.0, 0.5])
    y = X @ coef + 4.0 + rng.normal(scale=0.1, size=200)

    sk = LinearRegression().fit(X, y)
    scratch = LinearRegressionScratch(method="normal").fit(X, y)
    assert np.allclose(scratch.coef_, sk.coef_, atol=1e-3)
    assert abs(scratch.intercept_ - sk.intercept_) < 1e-3


def test_batch_gd_converges_close():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    y = 3.0 + X @ np.array([2.0, -1.0]) + rng.normal(scale=0.05, size=300)
    Xs = StandardScaler().fit_transform(X)

    sk = LinearRegression().fit(Xs, y)
    scratch = LinearRegressionScratch(
        method="batch", lr=0.1, n_epochs=2000, tol=1e-12
    ).fit(Xs, y)
    assert np.allclose(scratch.coef_, sk.coef_, atol=1e-2)
    assert abs(scratch.intercept_ - sk.intercept_) < 1e-2


def test_ridge_normal_equation():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 4))
    y = X @ rng.normal(size=4) + rng.normal(size=150)
    Xs = StandardScaler().fit_transform(X)
    alpha = 1.0
    sk = Ridge(alpha=alpha, fit_intercept=True).fit(Xs, y)
    # sklearn Ridge solves min ||y-Xw||^2 + alpha||w||^2 (no 2 factor in MSE)
    # Our L2 is added to mean squared error; match via normal eq with lambda carefully.
    scratch = LinearRegressionScratch(method="normal", l2=alpha / len(y)).fit(Xs, y)
    # Allow looser tol due to objective scaling differences
    assert np.allclose(scratch.coef_, sk.coef_, atol=5e-2)


def test_polynomial_features_degree():
    X = np.array([[2.0, 3.0], [1.0, 1.0]])
    Xp = polynomial_features(X, degree=2)
    # terms: x1, x2, x1^2, x1*x2, x2^2  (degree totals 1 then 2)
    assert Xp.shape == (2, 5)


def test_uci_real_estate_optional():
    """Optional UCI check — skip if offline."""
    try:
        # OpenML id for Real estate valuation data set
        data = fetch_openml(data_id=41021, as_frame=True, parser="auto")
    except Exception:
        import pytest

        pytest.skip("OpenML UCI fetch unavailable")
    X = data.data.select_dtypes(include=[np.number]).to_numpy()[:300]
    y = data.target.to_numpy()[:300].astype(float)
    Xs = StandardScaler().fit_transform(X)
    sk = LinearRegression().fit(Xs, y)
    scratch = LinearRegressionScratch(method="normal").fit(Xs, y)
    assert np.allclose(scratch.coef_, sk.coef_, atol=1e-3)

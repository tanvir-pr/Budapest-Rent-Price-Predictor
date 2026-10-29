"""From-scratch NumPy linear regression: normal equation, GD variants, L2, polynomials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

ScheduleName = Literal["constant", "inv_time", "exponential"]


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones for the intercept."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Expand 1 or 2 feature columns to polynomial terms (no interaction of all one-hots).

    For 1 feature: [x, x^2, ..., x^d]
    For 2 features: all monomials x1^a * x2^b with 1 <= a+b <= degree
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    if d > 2:
        raise ValueError("Scratch polynomial expansion supports at most 2 features.")
    if degree < 1:
        raise ValueError("degree must be >= 1")

    cols = []
    if d == 1:
        x = X[:, 0]
        for p in range(1, degree + 1):
            cols.append(x**p)
    else:
        x1, x2 = X[:, 0], X[:, 1]
        for total in range(1, degree + 1):
            for a in range(total + 1):
                b = total - a
                cols.append((x1**a) * (x2**b))
    return np.column_stack(cols)


def learning_rate_schedule(
    name: ScheduleName,
    lr0: float,
    epoch: int,
    decay: float = 0.01,
) -> float:
    if name == "constant":
        return lr0
    if name == "inv_time":
        return lr0 / (1.0 + decay * epoch)
    if name == "exponential":
        return lr0 * np.exp(-decay * epoch)
    raise ValueError(f"Unknown schedule: {name}")


@dataclass
class LinearRegressionScratch:
    """Linear regression with normal equation or gradient descent."""

    method: Literal["normal", "batch", "mini_batch", "sgd"] = "normal"
    lr: float = 0.01
    n_epochs: int = 1000
    batch_size: int = 32
    l2: float = 0.0
    schedule: ScheduleName = "constant"
    schedule_decay: float = 0.01
    tol: float = 1e-8
    random_state: int = 42
    fit_intercept: bool = True

    coef_: np.ndarray | None = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)
    lr_history_: list[float] = field(default_factory=list, init=False)

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return add_bias(X) if self.fit_intercept else X

    def _mse_l2(self, Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        resid = Xb @ w - y
        mse = float(np.mean(resid**2))
        if self.l2 > 0:
            # do not penalise intercept (index 0) when fit_intercept
            start = 1 if self.fit_intercept else 0
            mse += self.l2 * float(np.sum(w[start:] ** 2))
        return mse

    def _grad(self, Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = Xb.shape[0]
        resid = Xb @ w - y
        g = (2.0 / n) * (Xb.T @ resid)
        if self.l2 > 0:
            penalty = 2.0 * self.l2 * w
            if self.fit_intercept:
                penalty[0] = 0.0
            g = g + penalty
        return g

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        Xb = self._prepare_X(X)
        y = np.asarray(y, dtype=float).ravel()
        n, p = Xb.shape
        rng = np.random.default_rng(self.random_state)
        self.loss_history_ = []
        self.lr_history_ = []

        if self.method == "normal":
            # (X'X + λI) w = X'y ; do not regularise intercept
            xtx = Xb.T @ Xb
            if self.l2 > 0:
                reg = np.eye(p) * self.l2
                if self.fit_intercept:
                    reg[0, 0] = 0.0
                xtx = xtx + reg
            w = np.linalg.pinv(xtx) @ (Xb.T @ y)
            self.loss_history_.append(self._mse_l2(Xb, y, w))
        else:
            w = rng.normal(scale=0.01, size=p)
            indices = np.arange(n)
            for epoch in range(self.n_epochs):
                lr_t = learning_rate_schedule(
                    self.schedule, self.lr, epoch, self.schedule_decay
                )
                self.lr_history_.append(lr_t)

                if self.method == "batch":
                    g = self._grad(Xb, y, w)
                    w = w - lr_t * g
                elif self.method == "mini_batch":
                    rng.shuffle(indices)
                    for start in range(0, n, self.batch_size):
                        batch = indices[start : start + self.batch_size]
                        g = self._grad(Xb[batch], y[batch], w)
                        w = w - lr_t * g
                elif self.method == "sgd":
                    rng.shuffle(indices)
                    for i in indices:
                        xi = Xb[i : i + 1]
                        yi = y[i : i + 1]
                        g = self._grad(xi, yi, w)
                        w = w - lr_t * g
                else:
                    raise ValueError(self.method)

                loss = self._mse_l2(Xb, y, w)
                self.loss_history_.append(loss)
                if epoch > 0 and abs(self.loss_history_[-2] - loss) < self.tol:
                    break

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = w.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_

    @property
    def weights_with_bias_(self) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() first.")
        if self.fit_intercept:
            return np.concatenate([[self.intercept_], self.coef_])
        return self.coef_.copy()


def fit_polynomial_curve(
    X: np.ndarray,
    y: np.ndarray,
    degree: int,
    method: Literal["normal", "batch"] = "normal",
    **kwargs,
) -> LinearRegressionScratch:
    """Fit polynomial regression of given degree on 1–2 features."""
    Xp = polynomial_features(X, degree)
    model = LinearRegressionScratch(method=method, **kwargs)
    model.fit(Xp, y)
    return model

from __future__ import annotations

import numpy as np

from src.evaluate import mae_huf, mape, regression_report, rmse_huf


def test_perfect_predictions():
    y = np.array([100.0, 200.0, 300.0])
    assert mae_huf(y, y) == 0
    assert rmse_huf(y, y) == 0
    r = regression_report(y, y)
    assert r["r2"] == 1.0
    assert r["mape"] == 0.0


def test_mae_known():
    y = np.array([10.0, 20.0])
    p = np.array([12.0, 18.0])
    assert mae_huf(y, p) == 2.0
    assert abs(mape(y, p) - 15.0) < 1e-9

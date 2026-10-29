"""Leakage tests: fold-fitted transformers must not use test statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import DistrictRentEncoder, build_preprocessor


def _toy_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    district = rng.integers(1, 6, size=n)
    size = rng.uniform(30, 100, size=n)
    rent = 2000 * size + district * 10000 + rng.normal(0, 5000, size=n)
    return pd.DataFrame(
        {
            "size_m2": size,
            "rooms_total": rng.integers(1, 4, size=n).astype(float),
            "floor": rng.integers(0, 8, size=n).astype(float),
            "building_floors": 10.0,
            "district_num": district.astype(float),
            "is_buda": (district <= 2).astype(float),
            "is_centre": (district <= 3).astype(float),
            "condition": rng.choice(["good", "renovated", "new"], size=n),
            "heating": rng.choice(["gas", "district"], size=n),
            "furnished": rng.choice(["yes", "no"], size=n),
            "view": "street",
            "orientation": "south",
            "balcony": "yes",
            "elevator": "yes",
            "description": "renovated balcony AC",
            "title": "flat",
            "lat": np.nan,
            "lon": np.nan,
            "property_type": rng.choice(["tégla", "panel"], size=n),
            "animals": rng.choice(["igen", "nem"], size=n),
            "rent": rent,
            "rent_per_m2": rent / size,
        }
    )


def test_district_rent_encoder_ignores_test_labels():
    train = _toy_frame(60, seed=1)
    test = _toy_frame(20, seed=2)
    # Poison test rents to an extreme value — encoder fitted on train must be unchanged
    enc = DistrictRentEncoder(smoothing=5.0)
    enc.fit(train, train["rent"])
    stats_before = dict(enc.district_stats_)
    global_before = enc.global_median_

    test_poison = test.copy()
    test_poison["rent"] = 1e12
    # transform only — fit not called on test
    _ = enc.transform(test_poison)
    assert enc.district_stats_ == stats_before
    assert enc.global_median_ == global_before


def test_preprocessor_fit_transform_shapes():
    train = _toy_frame(50, seed=3)
    test = _toy_frame(15, seed=4)
    pre = build_preprocessor()
    Xt = pre.fit_transform(train, train["rent"])
    Xv = pre.transform(test)
    assert Xt.shape[0] == len(train)
    assert Xv.shape[0] == len(test)
    assert Xt.shape[1] == Xv.shape[1]
    assert np.isfinite(Xt).all() or np.isnan(Xt).sum() >= 0

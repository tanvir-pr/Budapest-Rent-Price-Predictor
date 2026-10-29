"""sklearn-compatible feature transformers and ColumnTransformer factory."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from src.config import CENTRE_LAT, CENTRE_LON, DISTRICT_CENTROIDS


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised haversine distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _as_df(X)
        for c in self.columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[self.columns]


class SizeFeatures(BaseEstimator, TransformerMixin):
    """size, log_size, size_per_room, size band indicators."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _as_df(X)
        size = pd.to_numeric(df.get("size_m2", np.nan), errors="coerce")
        rooms = pd.to_numeric(df.get("rooms_total", np.nan), errors="coerce")
        out = pd.DataFrame(index=df.index)
        out["size_m2"] = size
        out["log_size"] = np.log1p(size)
        out["size_per_room"] = size / rooms.replace(0, np.nan)
        out["size_small"] = (size < 40).astype(float)
        out["size_medium"] = ((size >= 40) & (size < 80)).astype(float)
        out["size_large"] = (size >= 80).astype(float)
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(
            ["size_m2", "log_size", "size_per_room", "size_small", "size_medium", "size_large"]
        )


class FloorFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _as_df(X)
        floor = pd.to_numeric(df.get("floor", np.nan), errors="coerce")
        bld = pd.to_numeric(df.get("building_floors", np.nan), errors="coerce")
        out = pd.DataFrame(index=df.index)
        out["floor"] = floor
        out["is_ground"] = (floor == 0).astype(float)
        out["is_top"] = ((bld.notna()) & (floor == bld)).astype(float)
        out["relative_floor"] = floor / bld.replace(0, np.nan)
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(["floor", "is_ground", "is_top", "relative_floor"])


class DistrictFlags(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _as_df(X)
        out = pd.DataFrame(index=df.index)
        out["is_buda"] = pd.to_numeric(df.get("is_buda", np.nan), errors="coerce")
        out["is_centre"] = pd.to_numeric(df.get("is_centre", np.nan), errors="coerce")
        out["rooms_total"] = pd.to_numeric(df.get("rooms_total", np.nan), errors="coerce")
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(["is_buda", "is_centre", "rooms_total"])


class GeoDistanceFeatures(BaseEstimator, TransformerMixin):
    """Distance to centre using lat/lon if present, else district centroid."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _as_df(X)
        lat = pd.to_numeric(df.get("lat", np.nan), errors="coerce")
        lon = pd.to_numeric(df.get("lon", np.nan), errors="coerce")
        dist_num = pd.to_numeric(df.get("district_num", np.nan), errors="coerce")

        use_centroid = lat.isna() | lon.isna()
        lat_f = lat.copy()
        lon_f = lon.copy()
        for idx in df.index[use_centroid]:
            d = dist_num.loc[idx]
            if pd.notna(d) and int(d) in DISTRICT_CENTROIDS:
                lat_f.loc[idx], lon_f.loc[idx] = DISTRICT_CENTROIDS[int(d)]

        dist_km = haversine_km(lat_f.to_numpy(), lon_f.to_numpy(), CENTRE_LAT, CENTRE_LON)
        out = np.column_stack([dist_km, np.log1p(dist_km)])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(["dist_centre_km", "log_dist_centre"])


class DistrictRentEncoder(BaseEstimator, TransformerMixin):
    """Smoothed district median rent/m² fitted on training folds only."""

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing
        self.global_median_: float = 0.0
        self.district_stats_: dict[float, float] = {}

    def fit(self, X, y=None):
        df = _as_df(X)
        d = pd.to_numeric(df.get("district_num", np.nan), errors="coerce")
        if y is None:
            # fallback: use rent_per_m2 column if present (training frame only)
            rpm = pd.to_numeric(df.get("rent_per_m2", np.nan), errors="coerce")
        else:
            size = pd.to_numeric(df.get("size_m2", np.nan), errors="coerce")
            y_arr = np.asarray(y, dtype=float).ravel()
            # y may be log-rent when used inside TransformedTargetRegressor — prefer raw rent col
            if "rent" in df.columns:
                rpm = pd.to_numeric(df["rent"], errors="coerce") / size
            else:
                # assume y is rent in HUF
                rpm = pd.Series(y_arr, index=df.index) / size

        valid = d.notna() & rpm.notna() & np.isfinite(rpm)
        self.global_median_ = float(np.median(rpm[valid])) if valid.any() else 0.0
        self.district_stats_ = {}
        tmp = pd.DataFrame({"d": d[valid], "rpm": rpm[valid]})
        for dist, g in tmp.groupby("d"):
            n = len(g)
            med = float(g["rpm"].median())
            # Bayesian-ish shrink to global
            self.district_stats_[float(dist)] = (
                n * med + self.smoothing * self.global_median_
            ) / (n + self.smoothing)
        return self

    def transform(self, X):
        df = _as_df(X)
        d = pd.to_numeric(df.get("district_num", np.nan), errors="coerce")
        vals = d.map(
            lambda x: self.district_stats_.get(float(x), self.global_median_)
            if pd.notna(x)
            else self.global_median_
        )
        return vals.to_numpy(dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["district_median_rpm"])


class AmenityFlags(BaseEstimator, TransformerMixin):
    """Binary amenity flags from categorical / text columns."""

    KEYWORDS = {
        "has_balcony": [r"erkely", r"erkély", r"balcony", r"terasz", r"terrace"],
        "has_elevator": [r"lift", r"elevator", r"felvon"],
        "is_furnished": [r"butor", r"bútor", r"furnish"],
        "has_ac": [r"klima", r"klíma", r"air.?cond", r"\bac\b"],
        "renovated": [r"feluj", r"felúj", r"renovat"],
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _as_df(X)
        text_parts = []
        for c in ("balcony", "elevator", "furnished", "condition", "description", "title", "heating"):
            if c in df.columns:
                text_parts.append(df[c].astype(str).fillna(""))
        if text_parts:
            blob = text_parts[0]
            for p in text_parts[1:]:
                blob = blob + " " + p
        else:
            blob = pd.Series([""] * len(df), index=df.index)

        blob = blob.str.lower()
        out = []
        names = []
        for name, patterns in self.KEYWORDS.items():
            mask = np.zeros(len(df), dtype=float)
            for pat in patterns:
                mask = np.maximum(mask, blob.str.contains(pat, regex=True, na=False).astype(float))
            out.append(mask)
            names.append(name)
        self._names = names
        return np.column_stack(out) if out else np.zeros((len(df), 0))

    def get_feature_names_out(self, input_features=None):
        return np.array(getattr(self, "_names", list(self.KEYWORDS.keys())))


class InteractionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _as_df(X)
        size = pd.to_numeric(df.get("size_m2", np.nan), errors="coerce")
        rooms = pd.to_numeric(df.get("rooms_total", np.nan), errors="coerce")
        centre = pd.to_numeric(df.get("is_centre", np.nan), errors="coerce")
        out = pd.DataFrame(index=df.index)
        out["size_x_centre"] = size * centre
        out["rooms_x_centre"] = rooms * centre
        out["size_x_rooms"] = size * rooms
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(["size_x_centre", "rooms_x_centre", "size_x_rooms"])


def _as_df(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def build_preprocessor(
    *,
    use_location: bool = True,
    use_interactions: bool = True,
    use_amenities: bool = True,
    condition_encoding: str = "onehot",
) -> ColumnTransformer:
    """Assemble the shared ColumnTransformer used by all models and the app."""

    transformers: list[tuple[str, Any, list[str] | slice]] = []

    transformers.append(
        (
            "size",
            Pipeline(
                [
                    ("feat", SizeFeatures()),
                    ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", RobustScaler()),
                ]
            ),
            ["size_m2", "rooms_total"],
        )
    )
    transformers.append(
        (
            "flags",
            Pipeline(
                [
                    ("feat", DistrictFlags()),
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", RobustScaler()),
                ]
            ),
            ["is_buda", "is_centre", "rooms_total"],
        )
    )
    transformers.append(
        (
            "property_type",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    (
                        "oh",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            ["property_type"],
        )
    )
    transformers.append(
        (
            "animals",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    (
                        "oh",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            ["animals"],
        )
    )

    # District as categorical one-hot (low cardinality ≤23)
    transformers.append(
        (
            "district",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    (
                        "oh",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            ["district_num"],
        )
    )

    if condition_encoding == "ordinal":
        transformers.append(
            (
                "condition",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ["condition"],
            )
        )
    else:
        transformers.append(
            (
                "condition",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "oh",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=12),
                        ),
                    ]
                ),
                ["condition"],
            )
        )

    transformers.append(
        (
            "cats",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                    (
                        "oh",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            max_categories=10,
                        ),
                    ),
                ]
            ),
            ["heating", "furnished", "view"],
        )
    )

    if use_location:
        transformers.append(
            (
                "geo",
                Pipeline(
                    [
                        ("feat", GeoDistanceFeatures()),
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", RobustScaler()),
                    ]
                ),
                ["lat", "lon", "district_num"],
            )
        )
        transformers.append(
            (
                "dist_rent",
                DistrictRentEncoder(smoothing=10.0),
                ["district_num", "size_m2", "rent"],
            )
        )

    if use_amenities:
        transformers.append(
            (
                "amenity",
                AmenityFlags(),
                ["balcony", "elevator", "furnished", "condition", "description", "title", "heating"],
            )
        )

    if use_interactions:
        transformers.append(
            (
                "interact",
                Pipeline(
                    [
                        ("feat", InteractionFeatures()),
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", RobustScaler()),
                    ]
                ),
                ["size_m2", "rooms_total", "is_centre"],
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def wrap_log_target(estimator) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


FEATURE_CATALOG = [
    {"name": "size_m2", "source": "size", "formula": "raw m²", "reason": "Strongest size–rent correlation in EDA", "leakage_risk": "low"},
    {"name": "log_size", "source": "size", "formula": "log1p(size)", "reason": "Right-skewed size; linearises vs log-rent", "leakage_risk": "low"},
    {"name": "size_per_room", "source": "size/rooms", "formula": "size/rooms", "reason": "Flags unusually small/large rooms", "leakage_risk": "low"},
    {"name": "size bands", "source": "size", "formula": "small/medium/large indicators", "reason": "Non-linearity of size vs rent", "leakage_risk": "low"},
    {"name": "floor / ground / top / relative", "source": "floor", "formula": "floor flags", "reason": "Ground/top premiums in EDA boxplots", "leakage_risk": "low"},
    {"name": "district one-hot", "source": "district_num", "formula": "OHE ≤23", "reason": "Primary location driver", "leakage_risk": "low"},
    {"name": "is_buda / is_centre", "source": "district", "formula": "side & centre flags", "reason": "Buda/Pest and centre vs outer gap", "leakage_risk": "low"},
    {"name": "dist_centre_km", "source": "geo", "formula": "haversine to Deák tér", "reason": "Distance decay of rent", "leakage_risk": "low"},
    {"name": "district_median_rpm", "source": "district stats", "formula": "smoothed train-fold median rent/m²", "reason": "Location prior without using test rents", "leakage_risk": "medium — fold-only"},
    {"name": "condition / heating / amenities", "source": "categoricals+text", "formula": "OHE + keyword flags", "reason": "Quality & amenity premia", "leakage_risk": "low"},
    {"name": "interactions", "source": "size×centre etc.", "formula": "selected products", "reason": "EDA slope differences by district group", "leakage_risk": "low"},
]

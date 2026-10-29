"""Load, clean, and validate Budapest rent listings."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.download import find_primary_csv
from src.config import (
    BUDA_DISTRICTS,
    CENTRE_DISTRICTS,
    DATA_PROCESSED,
    RENT_PER_M2_MAX,
    RENT_PER_M2_MIN,
    RANDOM_STATE,
)

# Candidate source-column names → canonical name (lowercase match)
COLUMN_ALIASES: dict[str, list[str]] = {
    "listing_id": ["unnamed: 0", "id", "url", "link", "listing_id", "index"],
    "rent": [
        "price",
        "rent",
        "rental_price",
        "monthly_rent",
        "ar",
        "ár",
        "berleti_dij",
        "bérleti_díj",
        "price_huf",
        "n_price",
    ],
    "size_m2": [
        "size",
        "area",
        "sqm",
        "square_meter",
        "squaremeter",
        "alapterulet",
        "alapterület",
        "nm",
        "m2",
        "area_size",
        "n_size",
    ],
    "rooms": ["rooms", "room", "szoba", "szobak", "szobák", "n_rooms", "room_count"],
    "half_rooms": ["half_rooms", "halfrooms", "felszoba", "félszoba", "n_half_rooms"],
    "floor": ["floor", "emelet", "n_floor"],
    "building_floors": [
        "building_floors",
        "floors_in_building",
        "max_floor",
        "epulet_szint",
        "épület_szintek",
        "number_of_floors",
    ],
    "district": [
        "location_str",
        "location",
        "district",
        "kerulet",
        "kerület",
        "city_district",
        "address",
        "cim",
        "cím",
    ],
    "condition": [
        "condition",
        "quality",
        "state",
        "allapot",
        "állapot",
        "comfort",
        "komfort",
    ],
    "heating": ["heating_type", "heating", "futes", "fűtés", "type_of_heating"],
    "furnished": ["furniture", "furnished", "butorozott", "bútorozott"],
    "elevator": ["elevator", "lift", "elevator_available"],
    "balcony": ["balcony", "erkely", "erkély", "terrace", "terasz", "garden"],
    "view": ["view_type", "view", "kilatas", "kilátás"],
    "orientation": ["orientation", "tajolas", "tájolás"],
    "year_built": ["year_built", "year", "epites_eve", "építés_éve", "built"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "lng", "longitude", "x"],
    "description": ["description", "desc", "leiras", "leírás", "text"],
    "title": ["title", "name", "cimke"],
    "listing_id": ["url", "unnamed: 0", "id", "link", "listing_id", "index"],
    "common_cost": ["common_cost", "kozos_koltseg", "közös_költség"],
    "animals": ["animals", "pets", "allatok", "állatok"],
    "property_type": ["type", "property_type", "tipus", "típus"],
    "date_scraped": ["date_scraped", "scraped_at", "date"],
}


def _norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _parse_number(val: Any) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "n/a", "na", "-", "unknown", "?"}:
        return np.nan
    s = s.replace("\xa0", " ")
    # Strip common unit tokens before removing letters (avoid "85 m2" → 852)
    s = re.sub(
        r"(?i)\b(m2|m²|nm|ft|huf|hu\s*f|ft/hó|ft/ho|eur|€)\b",
        "",
        s,
    )
    s = s.replace(" ", "")
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s or s in {"-", ".", ","}:
        return np.nan
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    elif s.count(",") > 0 and s.count(".") > 0:
        # thousand sep vs decimal: European style 1.250.000,5 or 1,250,000.5
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(".") > 1 and s.count(",") == 0:
        # thousand separators only: 1.250.000
        s = s.replace(".", "")
    elif s.count(",") > 1 and s.count(".") == 0:
        s = s.replace(",", "")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map raw columns to canonical names; return mapped frame + {canonical: source}."""
    norm_to_raw = {_norm_key(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    used_raw: set[str] = set()

    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _norm_key(alias)
            if key in norm_to_raw and norm_to_raw[key] not in used_raw:
                raw = norm_to_raw[key]
                mapping[canon] = raw
                used_raw.add(raw)
                break
        # fuzzy: alias as whole token inside column name (avoid lat⊂location)
        if canon not in mapping:
            for nkey, raw in norm_to_raw.items():
                if raw in used_raw:
                    continue
                tokens = set(nkey.split("_"))
                for alias in aliases:
                    a = _norm_key(alias)
                    if not a or len(a) < 3:
                        continue
                    if a == nkey or a in tokens:
                        mapping[canon] = raw
                        used_raw.add(raw)
                        break
                if canon in mapping:
                    break

    out = pd.DataFrame(index=df.index)
    for canon, raw in mapping.items():
        out[canon] = df[raw]
    # keep unmapped columns prefixed
    for c in df.columns:
        if c not in used_raw:
            out[f"raw__{_norm_key(c)}"] = df[c]
    return out, mapping


def parse_district(series: pd.Series) -> pd.Series:
    """Extract Budapest district number I–XXIII from messy text."""
    roman = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "viii": 8,
        "ix": 9,
        "x": 10,
        "xi": 11,
        "xii": 12,
        "xiii": 13,
        "xiv": 14,
        "xv": 15,
        "xvi": 16,
        "xvii": 17,
        "xviii": 18,
        "xix": 19,
        "xx": 20,
        "xxi": 21,
        "xxii": 22,
        "xxiii": 23,
    }

    def one(val: Any) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            v = int(val)
            return float(v) if 1 <= v <= 23 else np.nan
        s = str(val).lower().strip()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        m = re.search(r"\b([1-9]|1[0-9]|2[0-3])\b", s)
        if m and ("ker" in s or "district" in s or "bp" in s or "budapest" in s):
            return float(m.group(1))
        m = re.search(
            r"\b(xxiii|xxii|xxi|xx|xix|xviii|xvii|xvi|xv|xiv|xiii|xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)\b\.?",
            s,
        )
        if m:
            return float(roman[m.group(1)])
        m = re.search(r"\b([1-9]|1[0-9]|2[0-3])\b", s)
        if m:
            return float(m.group(1))
        return np.nan

    return series.map(one)


def parse_floor(series: pd.Series) -> pd.Series:
    """Parse floor labels including Hungarian ground/basement words."""
    ground_words = {"foldszint", "földszint", "fsz", "ground", "ground floor", "0"}
    basement_words = {"pince", "alagsor", "basement", "-1"}

    def one(val: Any) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        s = str(val).strip().lower()
        s_norm = unicodedata.normalize("NFKD", s)
        s_norm = "".join(c for c in s_norm if not unicodedata.combining(c))
        if s_norm in ground_words or "foldszint" in s_norm or s_norm == "fsz":
            return 0.0
        if s_norm in basement_words or "pince" in s_norm or "alagsor" in s_norm:
            return -1.0
        m = re.search(r"-?\d+", s_norm)
        return float(m.group()) if m else np.nan

    return series.map(one)


def load_raw(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or find_primary_csv()
    # Try utf-8 then latin-1 / cp1250 for Hungarian
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.attrs["source_path"] = str(path)
            df.attrs["encoding"] = enc
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to read {path}: {last_err}")


def clean_listings(
    raw: pd.DataFrame | None = None,
    *,
    drop_error_outliers: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full cleaning pipeline → modelling frame + audit dict."""
    if raw is None:
        raw = load_raw()

    mapped, col_map = map_columns(raw)
    audit: dict[str, Any] = {
        "n_raw": len(raw),
        "column_mapping": col_map,
        "source_columns": list(raw.columns),
    }

    df = mapped.copy()

    # Numeric parsing
    for col in ("rent", "size_m2", "rooms", "half_rooms", "building_floors", "year_built", "lat", "lon"):
        if col in df.columns:
            df[col] = df[col].map(_parse_number)

    if "floor" in df.columns:
        df["floor"] = parse_floor(df["floor"])

    if "district" in df.columns:
        df["district_num"] = parse_district(df["district"])
    else:
        for c in df.columns:
            if c.startswith("raw__") and any(k in c for k in ("address", "cim", "ker", "loc")):
                df["district_num"] = parse_district(df[c])
                break
        if "district_num" not in df.columns:
            df["district_num"] = np.nan

    # Parse "(lat, lon)" strings from raw location if lat/lon still missing
    if ("lat" not in df.columns or df["lat"].isna().all()) and "raw__location" in df.columns:
        coords = df["raw__location"].astype(str).str.extract(
            r"\(\s*([0-9.+-]+)\s*,\s*([0-9.+-]+)\s*\)"
        )
        df["lat"] = pd.to_numeric(coords[0], errors="coerce")
        df["lon"] = pd.to_numeric(coords[1], errors="coerce")
    elif "lat" in df.columns and df["lat"].dtype == object:
        coords = df["lat"].astype(str).str.extract(
            r"\(\s*([0-9.+-]+)\s*,\s*([0-9.+-]+)\s*\)"
        )
        if coords.notna().any().any():
            df["lat"] = pd.to_numeric(coords[0], errors="coerce")
            df["lon"] = pd.to_numeric(coords[1], errors="coerce")

    # Prefer lat/lon from raw__location when district mapped location_str
    if "raw__location" in df.columns:
        coords = df["raw__location"].astype(str).str.extract(
            r"\(\s*([0-9.+-]+)\s*,\s*([0-9.+-]+)\s*\)"
        )
        lat_p = pd.to_numeric(coords[0], errors="coerce")
        lon_p = pd.to_numeric(coords[1], errors="coerce")
        if "lat" not in df.columns:
            df["lat"] = lat_p
            df["lon"] = lon_p
        else:
            df["lat"] = df["lat"].fillna(lat_p)
            df["lon"] = df["lon"].fillna(lon_p) if "lon" in df.columns else lon_p

    # Combine half-rooms into rooms if present
    if "rooms" in df.columns and "half_rooms" in df.columns:
        df["rooms_total"] = df["rooms"].fillna(0) + 0.5 * df["half_rooms"].fillna(0)
        df.loc[df["rooms"].isna() & df["half_rooms"].isna(), "rooms_total"] = np.nan
    elif "rooms" in df.columns:
        df["rooms_total"] = df["rooms"]
    else:
        df["rooms_total"] = np.nan

    # Optional extras for this dataset
    if "property_type" in df.columns:
        df["property_type"] = (
            df["property_type"].astype(str).str.strip().str.lower().replace({"nan": np.nan})
        )
    if "animals" in df.columns:
        df["animals"] = (
            df["animals"].astype(str).str.strip().str.lower().replace({"nan": np.nan})
        )
    if "common_cost" in df.columns:
        df["common_cost"] = df["common_cost"].map(_parse_number)

    # Require target and size
    before = len(df)
    df = df.dropna(subset=["rent", "size_m2"]).copy()
    audit["dropped_missing_rent_or_size"] = before - len(df)

    df = df[(df["rent"] > 0) & (df["size_m2"] > 0)].copy()
    audit["n_after_positive"] = len(df)

    df["rent_per_m2"] = df["rent"] / df["size_m2"]
    df["log_rent"] = np.log1p(df["rent"])
    df["log_size"] = np.log1p(df["size_m2"])

    if "floor" not in df.columns:
        df["floor"] = np.nan
    if "building_floors" not in df.columns:
        df["building_floors"] = np.nan

    # Exact duplicates (full row) and duplicate listing URLs
    before = len(df)
    df = df.drop_duplicates()
    if "listing_id" in df.columns:
        df = df.drop_duplicates(subset=["listing_id"], keep="first")
    audit["dropped_exact_duplicates"] = before - len(df)

    # Near-duplicate group key: district + size + rooms + coarse geo (floor when present)
    # Coarse lat/lon (~110m) separates different buildings when floor is missing.
    lat_bin = (df["lat"].fillna(0) * 1000).round(0).astype(int).astype(str) if "lat" in df.columns else "0"
    lon_bin = (df["lon"].fillna(0) * 1000).round(0).astype(int).astype(str) if "lon" in df.columns else "0"
    df["group_key"] = (
        df["district_num"].fillna(-1).astype(int).astype(str)
        + "_"
        + df["size_m2"].round(0).astype(int).astype(str)
        + "_"
        + df["rooms_total"].fillna(-1).round(1).astype(str)
        + "_"
        + df["floor"].fillna(-99).astype(float).astype(int).astype(str)
        + "_"
        + lat_bin
        + "_"
        + lon_bin
    )

    # Collapse near-duplicates: keep the row closest to the group's median rent
    before = len(df)
    keep_indices: list[Any] = []
    for _, g in df.groupby("group_key", sort=False):
        med = g["rent"].median()
        keep_indices.append((g["rent"] - med).abs().idxmin())
    df = df.loc[keep_indices].copy().reset_index(drop=True)
    audit["dropped_near_duplicates"] = before - len(df)
    audit["n_after_near_dedup"] = len(df)

    # Location flags
    df["is_buda"] = df["district_num"].isin(BUDA_DISTRICTS).astype(float)
    df["is_centre"] = df["district_num"].isin(CENTRE_DISTRICTS).astype(float)
    df.loc[df["district_num"].isna(), ["is_buda", "is_centre"]] = np.nan

    # Domain error outliers (clear data errors)
    error_mask = (
        (df["size_m2"] < 10)
        | (df["size_m2"] > 400)
        | (df["rent"] < 20000)
        | (df["rent"] > 5_000_000)
        | (df["rent_per_m2"] < RENT_PER_M2_MIN)
        | (df["rent_per_m2"] > RENT_PER_M2_MAX)
    )
    if "rooms_total" in df.columns:
        error_mask = error_mask | (df["rooms_total"] > 15) | (df["rooms_total"] < 0)
    audit["n_domain_errors"] = int(error_mask.sum())
    if drop_error_outliers:
        df = df.loc[~error_mask].copy()

    # Categorical text normalize
    for col in ("condition", "heating", "furnished", "elevator", "balcony", "view", "orientation"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": np.nan, "none": np.nan, "": np.nan, "n/a": np.nan})
            )

    df["district_num"] = df["district_num"].astype("float")
    audit["n_clean"] = len(df)
    audit["random_state"] = RANDOM_STATE
    return df.reset_index(drop=True), audit


def validate_schema(df: pd.DataFrame) -> None:
    """Raise AssertionError if cleaned frame violates expected ranges."""
    required = ["rent", "size_m2", "log_rent", "group_key"]
    for c in required:
        assert c in df.columns, f"missing column {c}"
    assert df["rent"].gt(0).all()
    assert df["size_m2"].gt(0).all()
    assert df["rent"].notna().all()
    assert len(df) > 50, "too few rows after cleaning"


def save_processed(df: pd.DataFrame, name: str = "listings_clean.csv") -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / name
    df.to_csv(path, index=False)
    return path

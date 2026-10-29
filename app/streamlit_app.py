"""Streamlit rent estimator — uses the same fitted pipeline as training."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import EVERY custom class by name before unpickling (required on Streamlit Cloud)
from src.features import (  # noqa: E402
    AmenityFlags,
    ColumnSelector,
    DistrictFlags,
    DistrictRentEncoder,
    FloorFeatures,
    GeoDistanceFeatures,
    InteractionFeatures,
    SizeFeatures,
    build_preprocessor,
    haversine_km,
)
from src.config import BUDA_DISTRICTS, CENTRE_DISTRICTS  # noqa: E402

# Keep references so pickle can resolve them
_CUSTOM_CLASSES = (
    AmenityFlags,
    ColumnSelector,
    DistrictFlags,
    DistrictRentEncoder,
    FloorFeatures,
    GeoDistanceFeatures,
    InteractionFeatures,
    SizeFeatures,
    build_preprocessor,
    haversine_km,
)

MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Budapest Rent Predictor", page_icon="🏠", layout="centered")
st.title("Budapest Rent Price Predictor")
st.caption(
    "Classical ML demo on a historical scraped snapshot. "
    "Not a current-market estimator — rents have changed since the scrape."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model file missing at {MODEL_PATH}"
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


model, load_error = load_model()
if model is None:
    st.error("Could not load the trained model.")
    if load_error:
        st.code(load_error)
    st.info(
        "On Streamlit Cloud, reboot the app after updating requirements "
        "(scikit-learn==1.6.1). Locally run: `python -m src.train --model best`"
    )
    st.stop()

with st.form("listing"):
    c1, c2 = st.columns(2)
    with c1:
        size_m2 = st.number_input("Size (m²)", min_value=15.0, max_value=300.0, value=50.0)
        rooms = st.number_input("Rooms", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        district = st.selectbox("District (kerület)", list(range(1, 24)), index=6)
        property_type = st.selectbox("Building type", ["tégla", "panel", "egyéb", "missing"], index=0)
        animals = st.selectbox("Animals allowed", ["nem", "igen", "missing"], index=0)
    with c2:
        condition = st.selectbox(
            "Condition",
            ["új építésű", "újszerű", "felújított", "jó állapotú", "felújítandó", "missing"],
            index=2,
        )
        heating = st.selectbox(
            "Heating",
            ["gáz-cirkó", "gáz konvektor", "házközponti", "távhő", "elektromos", "missing"],
            index=0,
        )
        furnished = st.selectbox(
            "Furnished", ["igen", "nem", "megegyezés szerint", "missing"], index=1
        )
        balcony = st.selectbox(
            "Garden / balcony proxy", ["van", "nincs", "igen", "nem", "missing"], index=0
        )
        elevator = st.selectbox("Elevator", ["van", "nincs", "missing"], index=0)

    submitted = st.form_submit_button("Estimate monthly rent (HUF)")

if submitted:
    row = {
        "size_m2": size_m2,
        "rooms_total": rooms,
        "floor": np.nan,
        "building_floors": np.nan,
        "district_num": float(district),
        "is_buda": float(district in BUDA_DISTRICTS),
        "is_centre": float(district in CENTRE_DISTRICTS),
        "condition": condition,
        "heating": heating,
        "furnished": furnished,
        "view": "missing",
        "orientation": "missing",
        "balcony": balcony,
        "elevator": elevator,
        "description": f"{condition} {balcony} {elevator}",
        "title": "user input",
        "lat": np.nan,
        "lon": np.nan,
        "property_type": property_type,
        "animals": animals,
        "rent": np.nan,
    }
    X = pd.DataFrame([row])
    try:
        pred = float(model.predict(X)[0])
        st.success(f"Estimated rent: **{pred:,.0f} HUF / month**")
        st.caption(
            "Prediction is back-transformed from log-rent (approx. conditional median)."
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {type(e).__name__}: {e}")

st.markdown("---")
st.markdown(
    "Pipeline: cleaning → feature transformers → log-target regressor. "
    "See the README for metrics, limitations, and how to retrain."
)

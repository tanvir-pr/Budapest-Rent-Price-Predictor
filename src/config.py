"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_FIG = PROJECT_ROOT / "reports" / "figures"
REPORTS_RESULTS = PROJECT_ROOT / "reports" / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

# Deák Ferenc tér (approx city centre)
CENTRE_LAT = 47.4979
CENTRE_LON = 19.0546

# Approximate district centroids (Budapest I–XXIII) for distance features
# when listing coordinates are unavailable.
DISTRICT_CENTROIDS: dict[int, tuple[float, float]] = {
    1: (47.4979, 19.0395),
    2: (47.5410, 19.0180),
    3: (47.5670, 19.0360),
    4: (47.5610, 19.0900),
    5: (47.4970, 19.0510),
    6: (47.5060, 19.0650),
    7: (47.5000, 19.0700),
    8: (47.4890, 19.0800),
    9: (47.4750, 19.0700),
    10: (47.4830, 19.1400),
    11: (47.4750, 19.0300),
    12: (47.5000, 19.0000),
    13: (47.5300, 19.0700),
    14: (47.5150, 19.1000),
    15: (47.5600, 19.1200),
    16: (47.5200, 19.1700),
    17: (47.4800, 19.2000),
    18: (47.4400, 19.1800),
    19: (47.4500, 19.1400),
    20: (47.4350, 19.1000),
    21: (47.4250, 19.0700),
    22: (47.4200, 19.0200),
    23: (47.4000, 19.1200),
}

BUDA_DISTRICTS = {1, 2, 3, 11, 12, 22}
CENTRE_DISTRICTS = {1, 5, 6, 7, 8, 9, 13}

# Plausible rent-per-m² bounds (HUF) for outlier flagging (historical snapshot)
RENT_PER_M2_MIN = 500
RENT_PER_M2_MAX = 15000
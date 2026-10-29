"""Schema validation tests for cleaned listings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import clean_listings, parse_district, parse_floor, validate_schema, _parse_number


def test_parse_number_hungarian_formats():
    assert _parse_number("250 000 Ft") == 250000
    assert _parse_number("85 m2") == 85
    assert _parse_number("1.250.000") == 1250000
    assert np.isnan(_parse_number("N/A"))


def test_parse_district_variants():
    s = pd.Series(["VII. kerület", "Budapest 13.", "XIII", "xi. ker", "99"])
    out = parse_district(s)
    assert out.iloc[0] == 7
    assert out.iloc[2] == 13
    assert out.iloc[3] == 11


def test_parse_floor():
    s = pd.Series(["földszint", "3", "pince", "FSZ"])
    out = parse_floor(s)
    assert out.iloc[0] == 0
    assert out.iloc[1] == 3
    assert out.iloc[2] == -1


def test_validate_schema_rejects_bad():
    bad = pd.DataFrame({"rent": [1.0], "size_m2": [1.0], "log_rent": [0.0], "group_key": ["a"]})
    with pytest.raises(AssertionError):
        validate_schema(bad)


@pytest.mark.integration
def test_clean_real_data_if_present():
    from pathlib import Path
    from data.download import RAW_DIR

    if not list(Path(RAW_DIR).glob("*.csv")):
        pytest.skip("raw CSV not downloaded yet")
    df, audit = clean_listings()
    validate_schema(df)
    assert "column_mapping" in audit
    assert "rent" in df.columns
    assert df["rent"].min() > 0

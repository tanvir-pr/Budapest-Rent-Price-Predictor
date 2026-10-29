# Budapest Rent Price Predictor

Classical ML portfolio project (Project 1 of 7) — predict **monthly rent (HUF)** for Budapest flats from size, rooms, district, condition, amenities and location, and explain which factors drive price.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![sklearn](https://img.shields.io/badge/stack-classical%20ML-orange)](.)

> **Learning demo, not a live market estimator.** The data is a scraped snapshot from **May 2018** (jofogas.hu / Kaggle). Rents have changed since.

## Demo

```bash
python -m src.train --model best --tune
streamlit run app/streamlit_app.py
```

The Streamlit app loads `models/best_model.joblib` — the same leakage-safe pipeline used in training.

## Problem & metric

| Item | Choice |
|------|--------|
| Target | Monthly rent (HUF) |
| Train transform | `log1p(rent)` via `TransformedTargetRegressor` |
| **Primary metric** | **MAE (HUF)** after inverse transform |
| Supporting | RMSE, R², MAPE / MdAPE |
| Baseline | Dummy (median) |
| CV | 5-fold `GroupKFold` on near-duplicate groups |
| Split | Seeded `GroupShuffleSplit`, test used once |

**Why MAE?** Rent is right-skewed; RMSE is dominated by a few luxury flats.

**Retransformation bias:** back-transformed log predictions estimate the conditional **median**, not the mean.

## Dataset

| Source | Role |
|--------|------|
| [Flats for Rent at Budapest](https://www.kaggle.com/datasets/gaborvecsei/flats-to-rent-at-budapest) (G. Vecsei) | Primary tabular CSV (`flats.csv`) |
| UCI Real Estate Valuation | Unit-test only for NumPy vs sklearn (never reported as a result) |
| BKK GTFS | Optional; this build uses listing coordinates + district centroids |

**Verified schema (Week 1):** 2,844 rows × 22 columns. License on Kaggle: **Unknown** — state this when sharing. Scrape date column: `2018/05/06` (and nearby).

| Canonical | Source column |
|-----------|----------------|
| rent | `price` |
| size_m2 | `size` |
| rooms | `rooms` |
| district | `location_str` |
| lat / lon | parsed from `location` `"(lat, lon)"` |
| condition, heating, furnished, elevator, view | Hungarian categoricals |
| property_type | `type` (tégla / panel / …) |

**Note:** this scrape has **no floor** column — floor features are skipped; geo + district carry location.

After cleaning / near-dedup: ~2,291 modelling rows.

```bash
python -m data.download   # uses kagglehub
```

## EDA highlights

See `notebooks/01_eda.ipynb` and `reports/results/eda_findings.txt`.

1. Rent is right-skewed (skew ≈ 3.8); `log1p` cuts skew to ≈ 1.1.
2. Spearman(size, log-rent) ≈ **0.80** — size is the primary driver.
3. District **V** has the highest median rent; centre districts dominate the top tier.
4. Size and rooms correlate (~0.76) → multicollinearity → Ridge / ElasticNet.
5. Condition shows a clear quality premium (új / felújított vs felújítandó).

Figures: `reports/figures/01_*.png` … `10_*.png`.

## Feature catalog (summary)

| Feature | Why | Leakage risk |
|---------|-----|--------------|
| size, log_size, size bands, size/room | Non-linear size–rent link | low |
| rooms_total | Room count effect | low |
| district OHE + is_buda / is_centre | Strong location effect | low |
| dist_centre_km (haversine) | Distance decay | low |
| district_median_rpm (smoothed) | Local price level | **medium — fold-only** |
| condition / heating / furniture / amenities | Quality & comfort | low |
| property_type, animals | Building type & pet policy | low |
| interactions (size×centre, …) | EDA slope differences | low |

**Never used as input:** `rent_per_m2` (target-derived).

Ablation: `reports/results/feature_ablation.csv`, figure `11_feature_ablation.png`.

## Results

Primary metric = MAE (HUF). Numbers from a seeded run (`reports/results/model_comparison.csv`):

| Model | CV MAE | Test MAE | Test R² | Notes |
|-------|--------|----------|---------|-------|
| Dummy (median) | ~86.1k | ~80.8k | <0 | Naive baseline |
| Scratch GD | — | — | — | Matches sklearn OLS (±1e-3) in tests |
| Linear | ~45.4k | ~37.9k | ~0.79 | Strong baseline |
| Polynomial (size, rooms) | ~48.5k | ~41.9k | ~0.76 | Limited feature set |
| Ridge | ~45.2k | ~37.9k | ~0.78 | Stable regularised baseline |
| Lasso | ~45.1k | ~38.5k | ~0.78 | Needs small α on log scale |
| **ElasticNet** | **~45.0k** | **~38.2k** | **~0.78** | Best CV MAE in latest run |
| BayesianRidge | ~45.2k | ~38.1k | ~0.78 | Close second |
| Huber | ~45.8k | ~38.5k | ~0.77 | Robust to luxury tails |
| Optuna-tuned | — | ~37.9k | ~0.78 | Usually Ridge / ElasticNet |

Exact numbers vary slightly by environment; re-run `python -m src.train --tune` to refresh.

### Why this model won

Ridge on the full leakage-safe feature stack (log target + location + amenities) regularises correlated size/rooms/district signals without the over-shrinkage of a too-large Lasso α. Optuna confirmed Ridge with moderate α.

### What I learned

- Justify every feature from EDA, not from habit.
- Pipelines + group-aware splits are how you stop near-duplicate leakage.
- MAE + log target is a better story for skewed rents than RMSE alone.
- Scaling matters for GD and for Ridge/Lasso; closed-form OLS predictions are invariant to scaling but coefficients are not.

## Algorithms

**From scratch (NumPy):** normal equation, batch / mini-batch / SGD, LR schedules, L2, polynomial expansion on 1–2 features (`scratch/linreg.py`).

**sklearn:** Dummy, LinearRegression, PolynomialFeatures+Linear, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor.

**Analysis:** learning/validation curves, residual plots, IQR outliers, Huber vs OLS, Optuna.

## How to run

```bash
pip install -r requirements.txt
python -m data.download
python scripts/run_eda.py
python scripts/run_modeling.py --trials 50
pytest -q
streamlit run app/streamlit_app.py
```

Makefile targets: `make install download test train app`.

## Repository structure

```
budapest-rent-predictor/
  data/download.py          # kagglehub → data/raw/flats.csv
  notebooks/01_eda.ipynb … 04_error_analysis.ipynb
  scratch/linreg.py
  src/data.py features.py models.py train.py evaluate.py config.py
  tests/                    # schema, leakage, metrics, scratch vs sklearn
  reports/figures/ results/
  app/streamlit_app.py
  scripts/run_eda.py run_modeling.py
```

## Limitations

- 2018 snapshot; not current market prices.
- Scraped listings; missingness and reposts are real.
- No floor column in this CSV.
- License listed as Unknown on Kaggle — verify before commercial use.
- Log retransformation → median-oriented predictions.

## Interview talking points

- Why log-transform the target and what back-transformation implies.
- EDA → cleaning/feature decisions (district V, size skew, VIF).
- Leakage controls: Pipeline, fold-only district encoder, GroupKFold, near-dups.
- Normal equation vs GD; when GD wins.
- Scaling for GD / Ridge / Lasso vs OLS closed form.
- Lasso vs Ridge geometry; ElasticNet when features are correlated.
- Huber vs OLS on luxury tails.
- Bias–variance curve on polynomial degree (size & rooms only).

## Future work

- Folium prediction map; quantile regression intervals; Streamlit Community Cloud deploy.
- Optional BKK GTFS metro-distance features if terms allow.

## License

MIT for the code. Dataset rights remain with the original publishers / Kaggle listing terms.

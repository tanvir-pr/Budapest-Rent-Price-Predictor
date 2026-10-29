# Outlier study

- IQR (1.5×) on `log1p(rent)` flags ~5% of cleaned rows (`reports/results/outlier_table.csv`).
- Domain rules removed clear data errors (impossible size/rent/m²) before modelling.
- Luxury listings were **kept** for the main comparison so HuberRegressor can be contrasted with OLS / Ridge.
- Huber MAE is close to Ridge; largest absolute errors remain high-rent centre flats and atypical size–amenity combos (`worst_predictions.csv`).
- Cook's distance / influence diagnostics can be expanded in `04_error_analysis.ipynb` using statsmodels OLS on a numeric subset if needed for interviews.

# LinkedIn draft (do not publish until you review)

**Budapest Rent Price Predictor — Classical ML portfolio project**

I built a regression project that predicts monthly Budapest flat rents (HUF) from a messy 2018 scraped Kaggle dataset — with the focus on *process*, not just `.fit()`.

What I shipped:
- Deep EDA with a data dictionary, quality log, and justified cleaning decisions
- Leakage-safe sklearn pipelines (group-aware splits for near-duplicate listings)
- From-scratch NumPy linear regression (normal equation + GD variants) verified against sklearn
- Ridge / Lasso / ElasticNet / BayesianRidge / Huber, Optuna tuning, MAE in HUF as the primary metric
- Streamlit estimator using the same trained pipeline

Biggest takeaway: on right-skewed rents, log-target + MAE tells a clearer story than chasing RMSE on luxury outliers.

Repo: _add link_

#MachineLearning #DataScience #Portfolio #Python #scikitLearn

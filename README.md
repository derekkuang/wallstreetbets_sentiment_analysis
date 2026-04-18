# Predicting Next-Day Stock Volatility from r/WallStreetBets

Predicts whether a stock will experience a large next-day price move (>5%) using Reddit r/WallStreetBets discussion data combined with historical price features.

## Data

- **Source:** r/WallStreetBets posts/comments + Yahoo Finance daily prices
- **Period:** June 2023 -- March 2025
- **Universe:** 78 tickers (NVDA, TSLA, AMD, SPY, AAPL, etc.)
- **Scale:** 16,645 ticker-day observations, ~10.9% positive (big-move) base rate

## Features

| Category | Features |
|----------|----------|
| Price/Volume | Daily return, log volume, 1/3/5-day lagged returns, 5/10-day rolling volatility, volume anomaly ratio |
| WSB Engagement | Mention count, score sum, unique authors, post fraction |
| Text | TF-IDF (up to 10K features, unigrams + bigrams) on aggregated daily WSB text per ticker |
| Sentiment | VADER compound score (mean per ticker-day) |

## Model

Logistic regression with L2 regularization, trained via `GridSearchCV` with `TimeSeriesSplit` (5 folds) to respect chronological ordering. TF-IDF text features and scaled numeric features are combined through a `ColumnTransformer` pipeline. Class weighting is balanced to handle the 10.9% positive rate.

### Results (Test Set)

| Model | AUC | F1 | Precision | Recall |
|-------|-----|-----|-----------|--------|
| Always predict 0 | -- | 0.00 | 0.00 | 0.00 |
| Price-only logistic | 0.582 | 0.249 | 0.162 | 0.537 |
| WSB-numeric-only logistic | 0.577 | 0.229 | 0.222 | 0.236 |
| Price + WSB numeric | 0.598 | 0.279 | 0.207 | 0.430 |
| Text-only logistic | 0.776 | 0.419 | 0.311 | 0.643 |
| **Final (text + all numeric, tuned)** | **0.806** | **0.439** | **0.324** | **0.680** |

## Validation

Documented in `updated.ipynb`:

- **No data leakage** -- all features use information available at market close; label is next-day return. Verified via correlation analysis and manual return recomputation.
- **5% threshold justified** -- corresponds to ~90th percentile of daily absolute returns. Balances signal rarity with sufficient training samples.
- **Feature importance** -- top numeric drivers are rolling volatility and mention count. Top TF-IDF terms are ticker names and "earnings", confirming the model learns sensible patterns.
- **Robust to TF-IDF dimensionality** -- AUC stable at 0.79--0.81 across 500 to 10,000 features.
- **Calibration** -- model overestimates big-move probability (ECE = 0.22). Platt scaling recommended if using raw probabilities for decisions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn nltk yfinance matplotlib plotly tqdm
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Project Structure

```
main.ipynb          # Full pipeline: data collection, cleaning, EDA, modeling, evaluation
updated.ipynb       # Methodology validation & improvements
data/
  wsb.csv           # Raw WSB data
  wsb_cleaned.csv   # Cleaned WSB data
  wsb_exploded.csv  # One row per (post, ticker)
  prices_daily.csv  # Yahoo Finance daily prices
  modeling_df.csv   # Final merged modeling dataframe
plots/              # Visualizations from main.ipynb
plots/updated/      # Validation plots from updated.ipynb
```

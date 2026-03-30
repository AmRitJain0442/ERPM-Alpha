# GDELT India: USD/INR Exchange Rate Research

This repository contains a multi-model research project on USD/INR behavior using:

- GDELT political and financial news signals
- Macroeconomic and trade indicators
- Time-series and ML models (GARCH, XGBoost, ensemble methods)
- LLM-assisted simulation (Gemini persona-based forecasting)
- Monte Carlo scenario forecasting

The project objective is to evaluate whether news-derived sentiment and event intensity can help explain or predict exchange-rate volatility and short-horizon movement risk.

## Core Ideas

- **News as signal**: Use GDELT tone and Goldstein indicators, plus thematic filtering.
- **Volatility-first framing**: Focus on predicting risk/volatility regimes, not only point prices.
- **Model diversity**: Compare statistical, machine learning, hybrid, and simulation-based methods.
- **Reproducible artifacts**: Save intermediate datasets, reports, and plots for each phase.

## Repository Layout

- `data/gold_standard/`: Curated economic/trade source data and collection notes
- `scripts/data_collection/`: Data ingestion scripts (FRED, US Census, India commerce)
- `Phase-B/`: Thematic news filtering and feature engineering pipeline
- `GARCH/`: EGARCH/hybrid volatility modeling
- `ensemble_model/`: Ensemble forecasting workflows
- `gemini_preds/`: Gemini multi-persona market simulations (v2-v5)
- `monte_carlo_simulation/`: Probabilistic forecasting and risk bands
- `sample_datasets/`: Preview-sized datasets and metadata
- Root-level scripts: End-to-end analysis, merging, plotting, diagnostics, and utilities

## Data Inputs Used Across Workflows

- `usd_inr_exchange_rates_1year.csv`: Daily USD/INR time series
- `india_news_gz_combined_sorted.csv`: Large combined India-related GDELT dataset
- `india_financial_political_news_filtered.csv`: Filtered subset for economic/political signal extraction
- `data/gold_standard/fred/*.csv`: FRED macro indicators
- `data/gold_standard/india_commerce/*.csv`: India commerce files

## Environment Setup

Use Python 3.10+ (3.11 recommended).

```bash
# from repository root
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# optional: upgrade tooling
python -m pip install --upgrade pip setuptools wheel

# install common dependencies used across modules
pip install pandas numpy scikit-learn xgboost scipy statsmodels matplotlib seaborn arch requests tqdm yfinance pandas-datareader google-generativeai
```

Notes:

- This repo has multiple subproject requirement files (for example in `GARCH/`, `gemini_preds/`, `ensemble_model/`, and `latest/`).
- If you are only running one module, prefer installing from that module's `requirements.txt`.

## Quick Start Workflows

### 1) Fetch exchange rates

```bash
python fetch_exchange_rates.py
```

Outputs:

- `usd_inr_exchange_rates_1year.csv`
- `usd_inr_exchange_rates_1year.json`

### 2) Filter financial and political GDELT news

```bash
python filter_financial_political_news_optimized.py
```

Output:

- `india_financial_political_news_filtered.csv`

### 3) Build enhanced political-news model (root workflow)

```bash
python advanced_model_with_political_news.py
```

This workflow merges exchange rates, aggregated news features, and macro/trade variables, then compares baseline vs political-news-augmented model behavior.

### 4) Run Phase-B thematic feature engineering

```bash
cd Phase-B
python thematic_filter.py
python correlation_analysis.py
```

### 5) Run GARCH models

```bash
cd GARCH
pip install -r requirements.txt
python main.py
```

### 6) Run Gemini simulations

```bash
cd gemini_preds
pip install -r requirements.txt
python run_simulation_v4.py
```

If using Gemini APIs, configure credentials before running these scripts.

### 7) Run Monte Carlo forecasts

```bash
cd monte_carlo_simulation
python monte_carlo_exchange_rate.py
```

## Outputs You Should Expect

Typical generated artifacts include:

- Cleaned/merged CSV datasets
- Correlation tables and lag-analysis outputs
- Model reports and comparative metrics
- Volatility forecasts and confidence intervals
- Visualizations (`.png`) for diagnostics and presentation

## Dataset Packaging

Utilities for publishing and sharing are included:

- `create_sample_datasets.py`: creates preview subsets under `sample_datasets/`
- `upload_to_huggingface.py`: organizes and uploads datasets to Hugging Face Hub

Referenced dataset page:

- `https://huggingface.co/datasets/AmritJain/gdelt-india-research-datasets`

## Existing Documentation

For deeper detail, see:

- `RESEARCH_PRESENTATION.md`
- `MODEL_PRESENTATION.md`
- `Phase-B/README.md`
- `monte_carlo_simulation/README.md`
- `data/gold_standard/README.md`

## Notes and Caveats

- Several scripts assume specific local files already exist in root or module folders.
- Some experiments are versioned (`v2`, `v3`, `v4`, `v5`) and may require slightly different inputs.
- API-key-dependent pipelines (for example Gemini and some data collection workflows) need environment configuration.

## Suggested Execution Order (Practical)

1. Prepare Python environment and install dependencies.
2. Fetch exchange rates (`fetch_exchange_rates.py`).
3. Ensure GDELT combined files are present.
4. Run filtered-news extraction.
5. Run either:
   - Root enhanced model workflow, or
   - Module-specific workflows (`Phase-B`, `GARCH`, `gemini_preds`, Monte Carlo).
6. Review generated CSV metrics and plots.

## Author

Amrit

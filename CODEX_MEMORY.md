# Codex Memory: `gdelt_india`

This file is a working memory document for future coding sessions in this repo.
It is meant to answer:

- What this repository is actually doing
- Which parts are current vs legacy
- Which datasets/scripts matter for which workflow
- What caveats are easy to forget

Last refreshed: 2026-04-18

## 1. Repo Identity

This is a research-heavy USD/INR forecasting repository built around the idea that news sentiment and event intensity from GDELT may help explain or predict exchange-rate movement, volatility, or risk regimes.

The repo is not a single application. It is a layered research workspace with:

- a newest memory-first hybrid forecasting stack in `final phase/`
- phase-based experiments (`phase-a`, `Phase-B`, `Phase-C`)
- more formal modeling modules (`GARCH`, `ensemble_model`, `monte_carlo_simulation`, `india_usa_trade`)
- historical Gemini persona systems (`gemini_preds`)
- a newer refactored modular stack in `latest/`
- many root-level one-off scripts, plots, reports, and packaging utilities

The most current engineering-oriented code path is now `final phase/`.
`latest/` is still important and was the previous refactored mainline.
The most complete research narrative is spread across `README.md`, `STUDY_NOTES.md`, `MODEL_PRESENTATION.md`, `RESEARCH_PRESENTATION.md`, and `research_paper.tex`.

## 2. Current Repository Snapshot

- Git status was not clean when this memory was refreshed.
- The main untracked work was `CODEX_MEMORY.md` itself plus the new `final phase/` module and its generated outputs.
- HEAD commit: `34f1854` (`Added Readme`)
- Recent evolution from `git log`:
  - `d7756a5` Initial commit
  - `bcc6235` added phase b
  - `32f4044` Phase-C
  - `ca3b176` results of phase c
  - `29ef91e` added master data files
  - `5f7a77e` added data for import and export
  - `1d9976b` GARCH and ensemble
  - `e1db56c` added gemini-gen strat
  - `83731d5` added v3
  - `a245f0a` V4 model complete: added dynamic weights
  - `e13b75b` refactor code structure
  - `34f1854` added README

Interpretation:

- early repo = phase-based sentiment/volatility experiments
- middle repo = GARCH, ensemble, Gemini persona research
- latest repo state = refactor into a modular CLI system under `latest/`

## 3. High-Level Mental Model

The full repo is best understood as several connected research tracks:

1. Signal decomposition:
   separate exchange-rate series into trend/cycle/noise so news targets the noisy component.

2. News feature engineering:
   convert raw GDELT into daily sentiment, Goldstein, theme, and volume features.

3. Volatility/risk prediction:
   classify "danger days" or model volatility rather than only exact prices.

4. Econometric/statistical modeling:
   EGARCH, Ridge, XGBoost, ARIMA/VAR-like workflows.

5. LLM-assisted forecasting:
   Gemini persona systems and, in the refactor, 4 Gemini analyst tasks plus optional 30 local Ollama agents.

6. Ensemble/scenario workflows:
   combine many weak/strong models, and use Monte Carlo for distributional forecasting.

7. Trade/macro enrichment:
   add FRED, Census, and India commerce data as macro/trade context.

Important practical conclusion:

- If a future task is about "the current runnable system", start in `final phase/`.
- If the task is about the earlier refactor or Gemini/Ollama workflow, then start in `latest/`.
- If a future task is about reproducing charts/results from the paper/presentations, expect to work across root scripts plus `Phase-B`, `Phase-C`, `GARCH`, `gemini_preds`, `ensemble_model`, and `monte_carlo_simulation`.

## 4. Core Data Inventory

These are the important files the code actually uses.

### Main modeling datasets

| File | Rows | Date range | Notes |
| --- | ---: | --- | --- |
| `Super_Master_Dataset.csv` | 1,825 | 2019-01-02 to 2026-01-15 | Main tabular dataset for `latest/`; columns include INR, OIL, GOLD, US10Y, DXY, India/US tone/stability/mentions/panic features |
| `usd_inr_exchange_rates_1year.csv` | 255 | 2024-12-30 to 2025-12-29 | One-year USD/INR series used by many older scripts |
| `combined_goldstein_exchange_rates.csv` | 253 | 2025-01-02 to 2025-12-29 | Goldstein summaries plus exchange-rate columns |
| `political_news_exchange_merged.csv` | 252 | 2025-01-02 to 2025-12-24 | Aggregated political-news features merged with USD/INR |
| `Phase-B/merged_training_data.csv` | 254 | 2025-01-02 to 2025-12-26 | Thematic news features plus `IMF_3`; Phase-B/Phase-C/GARCH input |

### Raw / semi-raw GDELT datasets

| File | Rows | Date range | Approx size | Notes |
| --- | ---: | --- | ---: | --- |
| `india_news_gz_combined_sorted.csv` | 2,546,999 | 2025-01-01 to 2025-12-28 | 1.012 GB | Main large India GDELT file mentioned in docs |
| `india_news_combined_sorted.csv` | 432,509 | 2015-02-20 to 2025-12-07 | smaller | Used by `latest/` and some older scripts |
| `usa_news_combined_sorted.csv` | 14,377,323 | 2025-01-01 to 2025-12-30 | 5.739 GB | Very large USA news file; `latest/` chunk-loads this |
| `india_financial_political_news_filtered.csv` | 1,511,637 | 2025-01-01 to 2025-12-28 | 0.612 GB | Keyword-filtered India political/financial subset |

### Gold-standard macro/trade data

Present under `data/gold_standard/`:

- `fred/fred_combined_20251230_021942.csv`
- `fred/fred_wide_format_20251230_021943.csv`
- `india_commerce/TradeStat-Eidb-Export-Commodity-wise.csv`
- `india_commerce/TradeStat-Eidb-Import-Commodity-wise.csv`

`data/gold_standard/us_census/` exists as a directory but was empty when this memory doc was created.

### Sample / packaging datasets

`sample_datasets/` contains preview-sized copies for:

- exchange rates
- GDELT news
- master dataset
- correlation analysis
- trade data
- Monte Carlo forecasts
- GARCH outputs

These are generated by `create_sample_datasets.py` and used for sharing / Hugging Face packaging.

## 5. Documentation Reality Check

The docs are useful, but they are not perfectly synchronized.

What is reliable:

- `README.md` gives the broad repo layout and intended workflows.
- `STUDY_NOTES.md` is the best high-level synthesis of findings and module purpose.
- `research_paper.tex` is the most paper-like artifact and already contains real content.

What is inconsistent:

- some docs mention 3-4 day lead times while others mention 9-12 days
- some docs describe India-only datasets where scripts currently point at USA data or vice versa
- `report.md` is still a blank paper template, while `research_paper.tex` is substantially written
- `STUDY_NOTES.md` says it reflects commit `a245f0a`, which is not current HEAD anymore

Rule for future work:

- trust code paths and actual file schemas first
- use docs as historical interpretation, not as the final source of truth

## 6. Newest System: `final phase/`

This is the newest runnable stack added after the `latest/` refactor.

### Purpose

`final phase/` is a market-memory-first USD/INR prediction system that combines:

- macro and financial signals already present in `Super_Master_Dataset.csv`
- GDELT-derived structured signals from the repo's merged datasets
- statistical and ML regressors
- shared market-memory snapshots
- rule-based or LLM-based multi-persona forecasting
- explicit ablations and detailed result logging

### Important files

- `final phase/run.py`
  - CLI entry point
  - supports repeated `--llm-model` flags
  - blocks Claude Opus style specs by default unless `--allow-premium-llm-models` is passed

- `final phase/run_llm_matrix.ps1`
  - reproducible launcher for multi-model OpenRouter ablations
  - current default models:
    - `openrouter:google/gemini-2.5-flash`
    - `openrouter:anthropic/claude-sonnet-4`
    - `openrouter:openai/gpt-5-chat`

- `final phase/final_phase/config.py`
  - run config
  - experiment definitions
  - data paths into root datasets

- `final phase/final_phase/data.py`
  - integrates:
    - `Super_Master_Dataset.csv`
    - `combined_goldstein_exchange_rates.csv`
    - `Phase-B/merged_training_data.csv`
    - `political_news_exchange_merged.csv`

- `final phase/final_phase/features.py`
  - builds grouped features:
    - `technical`
    - `macro`
    - `master_sentiment`
    - `goldstein`
    - `thematic`
    - `political`
    - `memory`

- `final phase/final_phase/memory.py`
  - builds the shared market-memory snapshot
  - tracks persona-memory calibration over the backtest

- `final phase/final_phase/personas.py`
  - deterministic persona engine
  - LLM persona engine
  - currently the main multi-persona implementation to extend

- `final phase/final_phase/llm_clients.py`
  - OpenRouter and Ollama chat clients
  - local disk response cache
  - JSON extraction hardened for mixed content shapes

- `final phase/final_phase/modeling.py`
  - Ridge
  - RandomForestRegressor
  - XGBoost if available, otherwise GradientBoostingRegressor
  - logistic direction model
  - EGARCH volatility estimate if `arch` is installed

- `final phase/final_phase/evaluation.py`
  - walk-forward backtesting
  - temporal leakage controls
  - per-experiment output generation

### Bias and leakage controls

- raw immediate `INR` level is not a direct feature
- the target is forward return, not copied spot
- default info gap is `forecast_horizon_days + embargo_days = 2` trading days
- walk-forward fitting uses strict historical slices only
- persona calibration updates are delayed until the target date has matured

### Current default experiments

- `stat_ml_full`
- `stat_ml_no_gdelt`
- `stat_ml_no_macro`
- `memory_only_no_personas`
- `rule_personas`
- plus one experiment per provided `--llm-model`

### Current output locations worth remembering

- `final phase/outputs/backtest_100d/`
- `final phase/outputs/llm_matrix_v2_30d/`
- `final phase/outputs/llm_gemini_100d/`

### Current result snapshot as of 2026-04-18

Best 100-day non-LLM baseline:

- `memory_only_no_personas`
  - MAE price: `0.24209546`
  - RMSE price: `0.32898914`
  - directional accuracy: `0.61`

Best corrected 30-day LLM matrix result:

- `llm_openrouter_google_gemini_2_5_flash`
  - MAE price: `0.44329590`
  - RMSE price: `0.56962724`
  - directional accuracy: `0.53333333`

100-day LLM reference:

- `llm_openrouter_google_gemini_2_5_flash`
  - MAE price: `0.27261683`
  - RMSE price: `0.38986340`
  - directional accuracy: `0.58`

Practical interpretation:

- LLM personas improved materially over `rule_personas` and `stat_ml_full` on the tested 30-day window.
- The best pure structured baselines still beat the LLM hybrids on both 30-day and 100-day runs.
- `memory_only_no_personas` is surprisingly strong and should always remain in the ablation set.

### Model caveats already discovered

- avoid Claude Opus in this stack unless there is an explicit reason and cost approval
- `openrouter:openai/gpt-5-mini` can return reasoning-only responses with `content = null` for this workload
- `openrouter:google/gemma-4-26b-a4b-it:free` was unreliable enough to abstain or rate-limit in longer matrix runs
- the stable tested trio for this workload is:
  - `openrouter:google/gemini-2.5-flash`
  - `openrouter:anthropic/claude-sonnet-4`
  - `openrouter:openai/gpt-5-chat`

## 7. The Previous Modular System: `latest/`

This is the most important directory for future engineering work.

### Purpose

`latest/` is a refactored "LLM-as-Analyst USD/INR Prediction System" with:

- a CLI (`run.py`)
- feature loading and regime detection
- optional Gemini analyst tasks
- optional local Ollama multi-agent market simulation
- regime-conditional statistical prediction
- walk-forward backtesting
- persistent disk cache and meta-learner state

### Important files

- `latest/run.py`
  - CLI entry point
  - commands: `backtest`, `predict`, `range`

- `latest/config.py`
  - data paths
  - regime list and thresholds
  - feature names
  - model config
  - Gemini and Ollama settings

- `latest/data_loader.py`
  - loads `Super_Master_Dataset.csv`
  - adds technicals (`MA_5`, `MA_20`, `MA_momentum`, `RSI`, z-score, volatility)
  - builds a context packet for the LLM/agents
  - includes URL-slug headline extraction for GDELT `SOURCEURL`

- `latest/prompts.py`
  - defines the 4 Gemini analyst tasks:
    - regime classifier
    - event impact scorer
    - causal chain extractor
    - risk signal detector

- `latest/llm_tasks.py`
  - Gemini client init
  - retry handling
  - JSON extraction
  - disk cache use
  - numeric feature encoding from Gemini outputs

- `latest/market_agents.py`
  - 30 local Ollama personas
  - grouped into 6 archetypes:
    - technical
    - fundamental
    - carry
    - sentiment
    - flow
    - quant
  - produces consensus direction, entropy, confidence, archetype signals

- `latest/stat_engine.py`
  - regime-conditional predictor
  - `CALM_CARRY` -> Ridge
  - `TRENDING_*` -> XGBoost
  - `HIGH_VOLATILITY` / `CRISIS_STRESS` -> XGBoost plus EGARCH uncertainty

- `latest/pipeline.py`
  - orchestrates daily prediction
  - builds context from day `t-1`
  - optionally calls Gemini tasks and/or market agents
  - chooses regime
  - adds LLM/simulation features to numeric feature row
  - asks stat engine for prediction and confidence interval

- `latest/meta_learner.py`
  - stores per-regime feature weight state
  - starts LLM/agent influence at 5 percent
  - saves state to `latest/cache/meta_learner_state.json`

- `latest/backtest.py`
  - expanding-window walk-forward backtest
  - compares against:
    - `MA_Momentum` baseline
    - naive random walk baseline

- `latest/cache.py`
  - disk-backed JSON cache for Gemini task outputs

### Actual runtime status

Verified during this repo pass:

- `python latest/run.py --help` works
- `python latest/run.py predict 2025-01-10 --no-llm` worked and returned:
  - prediction: `85.8676`
  - actual: `86.188`
  - regime: `HIGH_VOLATILITY`

This means the stat-only path in `latest/` is currently runnable in this checkout.

### Data dependencies

`latest/config.py` points to these root-level files, and they were present:

- `Super_Master_Dataset.csv`
- `india_news_combined_sorted.csv`
- `usa_news_combined_sorted.csv`

### Important implementation details

- `latest/` uses `Super_Master_Dataset.csv`, not the one-year exchange-rate CSV, as its main model table.
- The target is next-day `INR` via `df["target"] = df["INR"].shift(-1)`.
- Context packets use the previous day plus recent price/macro/sentiment history.
- GDELT headlines are extracted from URL slugs, not from article bodies.
- GDELT files are chunk-read in `pipeline.py` to avoid loading all 14M+ USA rows into RAM.
- LLM tasks are cached on disk by task, date, and prompt hash.
- The meta-learner state persists across runs, so repeated runs are not perfectly stateless.

### Important caveats in `latest/`

- `latest/config.py` defaults Gemini to `gemini-2.0-flash`, while many older scripts use `gemini-2.5-flash`.
- `run.py --quick-agents` help text says "Use 10 agents only", but the actual code selects one persona per archetype, which currently means 6 agents, not 10.
- `MetaLearner.record_baseline()` exists but is not used by the current pipeline/backtest flow, so weight updates are not based on a clean with-vs-without baseline comparison.
- In `pipeline.py`, agent features currently reuse the same regime weight returned by the meta-learner for LLM features.
- Result payloads can still report an `llm_weight` even when `--no-llm` is used, because that field is just the current regime weight from the meta-learner.
- Ollama concurrency is limited by how Ollama serves a single local model; HTTP calls are threaded, but actual inference may still serialize.

## 8. Historical Gemini Persona Systems: `gemini_preds/`

This folder contains the older, research-oriented Gemini forecasting track.

### What it is

A series of versioned market-simulation systems:

- `gemini_market_simulation.py`
- `gemini_market_simulation_v2.py`
- `gemini_market_simulation_v3.py`
- `gemini_market_simulation_v4.py`
- `gemini_market_simulation_v5.py`

with matching runners, analyzers, plots, persona JSON files, and stored results.

### Key conceptual difference vs `latest/`

`gemini_preds/`:

- is the original 9-persona Gemini forecasting family
- tightly couples prompting, statistics, persona weighting, and simulation loops in large scripts
- is aimed at research/backtests and saved result analysis

`latest/`:

- is a refactor into modular components
- uses 4 Gemini analyst tasks plus optional 30 local Ollama agents
- is engineered as a reusable CLI/backtest/predict pipeline

### Important files

- `gemini_market_simulation_v4.py`
  - most important historical Gemini version
  - debiased prompting
  - robust aggregation
  - adaptive stat/LLM weighting
  - news digest integration

- `gemini_market_simulation_v5.py`
  - simplified / baseline-oriented follow-up

- `news_digest.py`
  - extracts and summarizes GDELT headlines for persona grounding

- `prepare_jan2026_data.py`
  - creates Jan 2026 datasets for live-style tests

- `run_simulation_v4.py`, `run_simulation_v4_jan2026.py`, `run_simulation_v5.py`
  - wrappers around the large simulation scripts

- `analyze_results_v*.py`, `plot_v*.py`
  - post-hoc analysis and charting

### Dependencies

- expects `GEMINI_API_KEY` in environment
- uses `google.generativeai`
- depends on root datasets like `Super_Master_Dataset.csv`, `india_news_combined_sorted.csv`, `usa_news_combined_sorted.csv`

### Stored outputs

This folder already contains:

- large `simulation_results_v*.json` files
- summary CSVs
- dashboards/plots
- Jan 2026 evaluation artifacts

This means many Gemini experiments have already been run and their outputs are versioned into the repo.

## 9. Phase-Based Pipeline

### `phase-a/`

Purpose:

- perform VMD decomposition on exchange rates
- isolate `IMF_3` noise component

Reality:

- `phase-a/P1.py` downloads `INR=X` from yfinance directly
- it uses 2 years of daily data as a placeholder/source
- outputs `../IMF_3.csv`
- saves `decomposition.png`

Important note:

- this script is standalone and not integrated with `latest/`
- it reflects an earlier research step rather than production-style data flow

### `Phase-B/`

Purpose:

- thematic filtering of GDELT data into daily features
- correlation of those features against `IMF_3`

Files:

- `thematic_filter.py`
- `correlation_analysis.py`
- `quick_correlation_check.py`
- `README.md`

What it does:

- keyword-matches articles into themes:
  - economy
  - conflict
  - policy
  - corporate
- computes daily metrics:
  - tone by theme
  - weighted Goldstein
  - counts
  - volume spikes
- merges with `IMF_3`
- saves:
  - `correlation_results.csv`
  - `merged_training_data.csv`
  - plots

Important caveat:

- docs describe the India workflow, but the `__main__` section of `thematic_filter.py` currently points to:
  - input: `usa_news_combined_sorted.csv`
  - output: `Usa_news_thematic_features.csv`

So the class itself is generic, but the script entrypoint is mismatched relative to the README narrative.

### `Phase-C/`

Purpose:

- use lagged Phase-B features to predict danger/high-volatility days or the IMF_3 component

Files:

- `danger_signal_classifier.py`
- `hybrid.py`

What it does:

- hardcodes lag features:
  - `Tone_Economy` lag 3
  - `Goldstein_Weighted` lag 4
  - `Volume_Spike` lag 1
- danger classifier:
  - top 20 percent of `IMF_3` as danger target
  - XGBoost classifier
  - threshold tuning for F1
- hybrid regression:
  - XGBoost regressor on `IMF_3`

Important caveats:

- these are script-style files, not reusable modules
- they use absolute Windows paths into this repo
- they assume `Phase-B/merged_training_data.csv` already exists

## 10. Formal Volatility Module: `GARCH/`

This is the more structured version of the volatility work.

### Purpose

- fit EGARCH / GARCH / GJR-GARCH
- compare models
- build an EGARCH + XGBoost hybrid volatility pipeline
- produce formal outputs under `GARCH/output/`

### Important files

- `config.py`
  - data paths
  - EGARCH parameters
  - XGBoost config
  - feature lists

- `egarch_model.py`
  - EGARCH model implementation and comparison helpers

- `hybrid_model.py`
  - hybrid EGARCH plus XGBoost pipeline

- `main.py`
  - main orchestration script

- `visualizations.py`
  - dashboard and plot helpers

### Data dependencies

- `Phase-B/merged_training_data.csv`
- `combined_goldstein_exchange_rates.csv`
- `data/gold_standard/fred/fred_wide_format_20251230_021943.csv`
- `usd_inr_exchange_rates_1year.csv`

### Outputs

Examples already present:

- `GARCH/output/garch_comparison.csv`
- `GARCH/output/predictions_price.csv`
- `GARCH/output/feature_importance.csv`
- several diagnostic PNGs

### Practical interpretation

If future work is about volatility modeling with a cleaner code layout than Phase-C, this is the right place to start.

## 11. Ensemble Work: `ensemble_model/`

This folder contains several generations of ensemble forecasting experiments.

### Key files

- `ensemble_exchange_rate_model.py`
  - large earlier ensemble system

- `enhanced_ensemble.py`
- `refined_ensemble.py`
- `ultimate_ensemble.py`
  - biggest and most ambitious all-in-one script
  - combines classical TS, ML, deep learning, decomposition, GDELT, Monte Carlo, and Gemini summary analysis

- `run_ensemble.py`
- `gemini_daily_predictor.py`

### Important practical note

`ultimate_ensemble.py` is valuable as research history, but it is monolithic and not the best place for incremental engineering changes unless the task is specifically about reproducing old ensemble behavior.

### Security note

Older ensemble Gemini code uses a hardcoded Gemini API key.

Affected files:

- `ensemble_model/ultimate_ensemble.py`
- `ensemble_model/gemini_daily_predictor.py`

Treat these as legacy/sensitive until cleaned up.

## 12. Monte Carlo Module: `monte_carlo_simulation/`

Purpose:

- probabilistic forecasting and uncertainty bands for USD/INR

Important files:

- `monte_carlo_exchange_rate.py`
  - multiple simulation styles

- `weekly_rolling_forecast.py`
  - runs rolling weekly simulations

- `README.md`
- `WEEKLY_FORECAST_GUIDE.md`

Current behavior:

- reads root exchange-rate and merged political-news datasets
- extracts regime parameters from historical volatility and event counts
- simulates multiple weeks forward
- writes summary CSVs and detailed result CSVs
- produces large PNGs already stored in repo

This is scenario/risk tooling, not the primary next-day prediction engine.

## 13. Trade and Data Collection Modules

### `india_usa_trade/`

Purpose:

- fetch and analyze India-USA bilateral trade data

Important files:

- `main.py`
  - orchestration for fetching and analysis

- `config.py`
  - country codes, HS codes, output config

- `census_api_fetcher.py`
- `comtrade_fetcher.py`
- `commodity_shift_analysis.py`
- `generate_report.py`

Capabilities:

- multi-year trade summary
- trade balance analysis
- commodity shift analysis
- seasonality analysis

Outputs:

- stored under `india_usa_trade/output/`
- many are already present and also sampled into `sample_datasets/trade_data/`

### `scripts/data_collection/`

Purpose:

- collect "gold standard" macro/trade data

Important files:

- `collect_all_data.py`
- `fetch_fred.py`
- `fetch_us_census.py`
- `fetch_india_commerce.py`
- `setup.py`
- `QUICKSTART.md`

Practical notes:

- FRED and Census are designed for environment variable API keys
- India commerce is still described as a manual-download workflow
- this area is more data-ops than modeling

Security caveat:

- `fetch_fred.py` and `collect_all_data.py` both contain a fallback hardcoded FRED API key string if the environment variable is missing

## 14. Important Root-Level Scripts

These scripts are not one coherent package; they are research utilities and analyses.

### Data ingest / filtering / merging

- `fetch_exchange_rates.py`
  - pulls USD/INR from Frankfurter API
  - writes `usd_inr_exchange_rates_1year.csv` and JSON

- `filter_financial_political_news.py`
- `filter_financial_political_news_optimized.py`
  - chunk-process GDELT and retain only financial/political India news

- `combine_csv.py`
- `combine_usa_news.py`
- `combine_and_analyze_goldstein.py`
- `calculate_daily_goldstein_avg.py`

### Modeling / analysis

- `advanced_exchange_rate_model.py`
  - merges exchange, GDELT, FRED, commerce data
  - runs classical econometric and ML models

- `advanced_model_with_political_news.py`
  - extends the above with dedicated political-news features

- `bilateral_sentiment_model.py`
  - compares India vs USA sentiment signals

- `meaningful_patterns_analysis.py`
  - looks for direction/volatility/regime patterns instead of exact prediction

- `model_accuracy_analysis.py`
- `mixtureofexperts.py`
- `hiddenstaes.py`

### Plotting / presentation

- many `plot_*.py` files
- `MODEL_PRESENTATION.md`
- `RESEARCH_PRESENTATION.md`
- `research_paper.tex`
- `research_paper.pdf`

### Packaging / sharing

- `create_sample_datasets.py`
- `upload_to_huggingface.py`

## 15. Research Paper State

There are two different paper-stage artifacts:

- `research_paper.tex`
  - already contains substantial paper text
  - title: `FXSage: Foreign Exchange Sentiment-Augmented Generalized Engine`
  - includes authors, abstract, methodology sections, and related work

- `report.md`
  - still an instructional template
  - not yet filled out as a finished paper

Practical takeaway:

- if asked to continue paper writing, use `research_paper.tex` as the real source
- `report.md` is not where the actual paper content lives right now

## 16. Known Path / Data Mismatches

These are easy traps.

### Bilateral USA news path mismatch

`bilateral_sentiment_model.py` expects:

- `usa/usa_news_combined_sorted.csv`

But that file was not present.
The actual combined USA file present is:

- `usa_news_combined_sorted.csv` at repo root

The `usa/` folder currently contains:

- `usa_daily_goldstein_averages.csv`
- compressed raw chunk files

So that script likely needs path correction before use.

### Phase-B India/USA mismatch

- docs frame Phase-B around India thematic features
- `thematic_filter.py` entrypoint currently runs on USA news

### Data-source inconsistency

- `phase-a/P1.py` uses yfinance (`INR=X`)
- `fetch_exchange_rates.py` uses Frankfurter API
- many downstream scripts assume root CSVs already exist

### Mixed time horizons

- `Super_Master_Dataset.csv` spans 2019-2026
- many older root scripts only use 2025 or 1-year windows

So metrics across modules are not always directly comparable.

## 17. Credentials and Secret Handling

This repo mixes good and bad patterns.

### Environment-variable based

- `latest/` Gemini path uses `GEMINI_API_KEY`
- `gemini_preds/` mostly uses `GEMINI_API_KEY`
- `india_usa_trade/` uses `CENSUS_API_KEY`
- data collectors use `FRED_API_KEY` / `CENSUS_API_KEY`

### Hardcoded secrets / risky defaults

Hardcoded Gemini key string found in:

- `ensemble_model/ultimate_ensemble.py`
- `ensemble_model/gemini_daily_predictor.py`
- `test_gemini.py`
- `test_gemini_parsing.py`

Hardcoded fallback FRED key string found in:

- `scripts/data_collection/fetch_fred.py`
- `scripts/data_collection/collect_all_data.py`

If the repo is ever shared publicly beyond current usage, these should be cleaned up first.

## 18. Testing Reality

This repo does not look like a unit-tested application.

What exists instead:

- manual smoke scripts:
  - `test_gemini.py`
  - `test_gemini_parsing.py`
  - `gemini_preds/test_gemini.py`
- backtesting scripts
- analysis scripts that write plots/tables

So "testing" in this repo usually means:

- run a workflow end to end
- inspect generated metrics/plots
- compare against stored outputs

## 19. Generated Artifact Locations

Useful output directories/files:

- `GARCH/output/`
- `ensemble_model/output/`
- `gemini_preds/` JSON summaries and PNGs
- `monte_carlo_simulation/` reports, summaries, detailed weekly results
- root PNG files used in presentations/reports
- `run_results/`

This repo stores many generated outputs in version control.

## 20. What To Use For Common Future Tasks

### If the task is "improve the current prediction engine"

Start in:

- `final phase/`

Especially:

- `run.py`
- `run_llm_matrix.ps1`
- `final_phase/config.py`
- `final_phase/memory.py`
- `final_phase/personas.py`
- `final_phase/evaluation.py`
- `final_phase/modeling.py`

### If the task is "improve the earlier modular prediction engine"

Start in:

- `latest/`

Especially:

- `config.py`
- `data_loader.py`
- `pipeline.py`
- `stat_engine.py`
- `llm_tasks.py`
- `market_agents.py`
- `backtest.py`

### If the task is "reproduce paper figures/results"

Expect to use:

- root plotting scripts
- `Phase-B/`
- `Phase-C/`
- `GARCH/`
- `gemini_preds/`
- `monte_carlo_simulation/`
- `research_paper.tex`

### If the task is "news feature engineering"

Start in:

- `Phase-B/`
- root filters/combines
- `data_loader.py` if the target is the newer modular stack

### If the task is "Gemini persona forecasting"

Start in:

- `gemini_preds/gemini_market_simulation_v4.py`
- `gemini_preds/run_simulation_v4.py`

### If the task is "local LLM agent simulation"

Start in:

- `latest/market_agents.py`

### If the task is "macro or trade data refresh"

Start in:

- `scripts/data_collection/`
- `india_usa_trade/`
- `data/gold_standard/`

## 21. Default Assumptions For Future Sessions

Unless a future task makes it clear otherwise, assume:

- `final phase/` is the newest system to modify for hybrid backtests
- `latest/` is the previous refactored mainline
- `Super_Master_Dataset.csv` is the main tabular input for that system
- Gemini work in `gemini_preds/` is historical but still important for research reproduction
- many root scripts are one-off analyses and may need path cleanup before rerunning
- outputs and charts checked into the repo are part of the research record, not disposable clutter

## 22. Short Summary I Should Remember

- This repo is a USD/INR research lab, not a single product.
- `final phase/` is the newest memory-first hybrid prediction system.
- `latest/` is the previous modular prediction system.
- `gemini_preds/` is the older persona-based Gemini research line.
- `Phase-B` and `Phase-C` are the original thematic-sentiment and danger-day pipelines.
- `GARCH/` is the formal volatility track.
- `ensemble_model/` and `monte_carlo_simulation/` are broader experiment families.
- The repo contains large real datasets, saved results, and paper assets.
- There are several path mismatches and a few hardcoded credentials in older code.
- For future engineering changes, start with `final phase/` unless the user explicitly wants an older research workflow.

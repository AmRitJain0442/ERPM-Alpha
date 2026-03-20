# Research Study Notes: USD/INR Exchange Rate Prediction via News Sentiment & LLM Agents

**Lead Researcher Notes** | Last Updated: March 2026
**Course**: BITS F471 (LLM Evaluation Research)
**Core Question**: Can news sentiment (GDELT) + LLM agents predict USD/INR exchange rate movements?

---

## 1. THE BIG PICTURE (What This Project Actually Does)

We built a **multi-layered prediction system** for USD/INR exchange rates that combines:
1. **Signal decomposition** (VMD) to separate trend/cycle/noise in the exchange rate
2. **News feature engineering** (GDELT) to extract market-relevant sentiment signals
3. **Statistical volatility models** (EGARCH) to capture volatility clustering + asymmetric shocks
4. **LLM-based multi-persona simulation** (Gemini) where 9 AI "analysts" provide independent views
5. **Ensemble methods** combining 15+ models with optimized weights

**The punchline finding**: Simpler models (MA_Momentum) beat complex ones (LSTM, GRU, Attention all had negative R²). The optimal ensemble converged to 95.5% MA_Momentum + 4.5% GDELT_Ridge. LLMs add value only for short-term (1-3 day) predictions.

---

## 2. DATA SOURCES & PIPELINE

### 2.1 Primary Data Sources
| Source | What It Contains | Volume | Time Range |
|--------|-----------------|--------|------------|
| **GDELT** (BigQuery) | Global news events: Goldstein scale, AvgTone, actors, URLs | 7.5M+ articles (2.5M India, 5M USA) | Jan 2025 - Dec 2025 |
| **Yahoo Finance** (yfinance) | USD/INR daily close prices, DXY | ~500 trading days | Jan 2023 - Jan 2026 |
| **FRED** | US10Y, Fed Funds Rate, DXY, Oil (WTI), CPI, M2, GDP | Daily/Monthly | 2023-2025 |
| **India Commerce** | TradeStat import/export commodity data | Annual aggregates | 2024-2025 |
| **UN Comtrade / US Census** | India-USA bilateral trade flows | Monthly/Annual | 2010-2025 |

### 2.2 Key Data Files in Repo
- `Super_Master_Dataset.csv` / `Super_Master_Dataset_Jan2026.csv` - merged dataset with all features + prices
- `india_news_combined_sorted.csv` / `india_news_gz_combined_sorted.csv` - raw GDELT India
- `usa_news_combined_sorted.csv` - raw GDELT USA
- `IMF_3.csv` - extracted noise component from VMD decomposition
- `combined_goldstein_exchange_rates.csv` - merged goldstein + exchange rates
- `Phase-B/merged_training_data.csv` - news features merged with IMF_3 for training
- `usd_inr_exchange_rates_1year.csv` - clean exchange rate series

### 2.3 Data Collection Scripts
- `scripts/data_collection/` - FRED, India Commerce, US Census API fetchers
- `fetch_exchange_rates.py` - Yahoo Finance pull
- `combine_csv.py`, `combine_usa_news.py` - data merging utilities
- `upload_to_huggingface.py` - dataset publication
- `create_sample_datasets.py` - curated sample data in `sample_datasets/`

---

## 3. RESEARCH PHASES (How The Work Evolved)

### Phase A: Signal Decomposition (`phase-a/`)
**Goal**: Separate the exchange rate signal into predictable components.

**Method**: Variational Mode Decomposition (VMD) with K=3 modes:
- **IMF 1 (Trend)**: Smooth macro curve driven by interest rates, inflation differentials
- **IMF 2 (Cycle)**: Medium-term oscillations (business cycle, seasonal)
- **IMF 3 (Noise)**: High-frequency residuals -- THIS is what news can predict

**Key Params**: alpha=2000 (bandwidth), tau=0 (strict fidelity), K=3 modes

**Critical Insight**: "Only try to predict the Red Line (IMF 3) with news. Don't try to predict the Blue Line (trend) with headlines."

**File**: `phase-a/P1.py` (outputs `IMF_3.csv`)

---

### Phase B: Thematic Feature Engineering (`Phase-B/`)
**Goal**: Transform raw GDELT into features that correlate with IMF 3 noise.

**Method**: Keyword-based thematic filtering of 2.5M articles into 4 categories:
| Theme | Keywords | Article Count |
|-------|----------|---------------|
| Economy | RBI, inflation, tax, GDP, forex, Fed, Sensex... | 139,953 |
| Conflict | protest, strike, war, Kashmir, Pakistan, China... | 560,938 |
| Policy | regulation, reform, parliament, Modi, cabinet... | 594,924 |
| Corporate | Adani, Reliance, Tata, merger, IPO, scandal... | 53,779 |

**Engineered Features** (16 total per day):
- Tone_Economy, Tone_Conflict, Tone_Policy, Tone_Corporate, Tone_Overall
- Goldstein_Weighted (Goldstein × NumMentions -- captures impact magnitude)
- Goldstein_Avg
- Count per theme, Count_Total
- Volume_Spike, Volume_Spike_Economy, Volume_Spike_Conflict

**Key Design Decision**: Goldstein_Weighted is `score × mentions`. A Goldstein of -10 in 5 mentions = noise. A -10 in 5000 mentions = market crash signal.

**Correlation Analysis**: `correlation_analysis.py` merges features with IMF_3 and runs:
- Pearson correlations with p-values
- Lag analysis (0-7 day lags)
- Heatmap visualization

**Files**: `thematic_filter.py`, `correlation_analysis.py`
**Outputs**: `india_news_thematic_features.csv`, `correlation_results.csv`, `merged_training_data.csv`

---

### Phase C: Volatility Classification & Prediction (`Phase-C/`)
**Goal**: Can lagged news features predict high-volatility "danger days"?

#### Danger Signal Classifier (`danger_signal_classifier.py`)
- **Target**: Binary -- is IMF_3 in top 20% (high volatility)?
- **Features with discovered lags**:
  - `Tone_Economy` lagged 3 days
  - `Goldstein_Weighted` lagged 4 days
  - `Volume_Spike` lagged 1 day
- **Model**: XGBoost classifier with `scale_pos_weight` for class imbalance
- **Evaluation**: Classification report, ROC-AUC, F1-optimized threshold tuning
- **Key Finding**: Political sentiment **leads** exchange rate by 3-4 days

#### Hybrid Predictor (`hybrid.py`)
- XGBoost regression on IMF_3 using the same lagged features
- Walk-forward validation (80/20 time split, no shuffling)
- Outputs RMSE/MAE on test set

**Important**: Both models use **strict temporal ordering** -- no data leakage.

---

## 4. MODELS BUILT (The Full Arsenal)

### 4.1 EGARCH Volatility Model (`GARCH/`)
**Why EGARCH over GARCH?**
1. Models log(variance) -- never predicts negative volatility
2. Captures **leverage effect**: bad news increases vol MORE than good news of same magnitude
3. Better suited for FX where panic causes instant spikes but relief decays gradually

**Math**: log(σ²_t) = ω + α|z_{t-1}| + γz_{t-1} + βlog(σ²_{t-1})

**Key Result**: γ = 0.20 (positive in EGARCH = leverage effect confirmed). EGARCH beat GARCH and GJR-GARCH on AIC.

**Architecture** (`config.py`, `egarch_model.py`, `hybrid_model.py`, `main.py`):
```
[Returns] → [EGARCH] → [Conditional Volatility] → [XGBoost + GDELT features] → [Adjusted Prediction]
```
- Stage 1: EGARCH extracts baseline volatility "physics"
- Stage 2: XGBoost corrects using GDELT news, FRED macro, derived features

**Derived Features Created**:
- Panic_Index = -Tone_Overall + |Tone_Conflict| + Volume_Spike*0.1
- Diff_Stability = rolling 5-day std of Tone_Overall
- News_Shock = Goldstein_Avg deviation from 10-day MA
- Is_Panic / Is_Relief binary indicators
- GARCH_Vol_Lag1, GARCH_Vol_Change

**Top Feature Importances** (XGBoost hybrid):
1. Realized_Vol_20d: 27.5%
2. India_Avg_Goldstein: 15.8%
3. USD Trade Index: 9.6%
4. Goldstein_Avg: 8.2%
5. Oil Price: 6.2%

---

### 4.2 Gemini Multi-Persona LLM System (`gemini_preds/`)
**The Core Innovation**: Simulates a trading floor with 9 specialized AI analysts.

#### Version History:
| Version | Key Feature | Stat/LLM Weight |
|---------|-------------|-----------------|
| V1 | Basic Gemini queries with personas | N/A |
| V2 | Added statistical baseline | Fixed 50/50 |
| V3 | Ridge regression + Gemini hybrid | Fixed 80/20 |
| V4 | **Debiased design** + adaptive weights | Dynamic 55-85% stat |
| V5 | Naive baseline (last close) for comparison | N/A |

#### V4 Architecture (THE KEY MODEL):

**9 Personas** with assigned weights:
| Persona | Weight | Focus |
|---------|--------|-------|
| Macro Analyst | 15% | Rate diffs, growth gaps, capital flows |
| Rates Strategist | 13% | Yield curve, carry trade, real rates |
| Flow Trader | 12% | Positioning, client flows, contrarian |
| RBI Watcher | 12% | RBI policy, reserves, intervention |
| Commodity Analyst | 12% | Oil/Gold impact on India's CAD |
| Technical Analyst | 10% | Price action, momentum, mean reversion |
| Sentiment Analyst | 10% | GDELT tone, volume, sentiment momentum |
| Risk Manager | 8% | Tail risks, vol regime, skepticism |
| Quant Researcher | 8% | Model trust, regime shifts, distribution |

**Debiasing Techniques** (V4's key contribution):
1. **Symmetric bull/bear case presentation** in prompts
2. **Randomized direction order** (50% show bull first) to prevent anchoring
3. **Historical accuracy feedback** ("Your accuracy: 52% - BELOW average")
4. **Explicit uncertainty bands** (95% CI shown to each persona)
5. **Pips-based adjustment** (0-30 pips scale, not raw price)
6. **Historical bias correction** ("WARNING: +0.05% bullish bias recently")

**Aggregation Pipeline**:
1. Each persona returns JSON: {direction, adjustment_pips, confidence, primary_reason}
2. Convert pips to % adjustment (10 pips ≈ 0.10%)
3. Trimmed Weighted Mean: remove top/bottom 10% outliers, weight by persona weights
4. Entropy calculation: H = -Σp·log₂(p), normalize to [0,1]
5. Consensus detection: >65% one direction = consensus

**Adaptive Weight Calculation**:
- Base: 70% stat, 30% LLM
- Adjustments based on: entropy, R², volatility regime, recent LLM value-add
- Final: clamp to stat_weight ∈ [55%, 85%]
- Formula: final = stat_pred × (1 + llm_adj% × llm_weight)

**News Digest System** (`news_digest.py`):
- Extracts headlines from GDELT SOURCEURL by parsing URL slugs
- Uses Gemini to create balanced daily digests (bull/bear factors)
- Feeds digest into each persona's prompt for grounding

#### V4 Performance:
- Average prediction error: ~0.3%
- Direction accuracy: 54-58% (V3 was 52-55%)
- Statistical R²: 0.76-0.82
- January 2026 live test: captured INR depreciation ~90→~92 with <0.5% avg error

---

### 4.3 Ensemble Models (`ensemble_model/`)

**Evolution**:
1. **Basic Ensemble** (`ensemble_exchange_rate_model.py`): VMD+SARIMA+LSTM+GARCH+GDELT+MonteCarlo. RMSE=0.479, R²=0.599
2. **Enhanced** (`enhanced_ensemble.py`): Simplified. Monte Carlo dominated (100% weight).
3. **Refined** (`refined_ensemble.py`): Monte Carlo 50% + MA_Trend 50%. R²=0.477
4. **Ultimate** (`ultimate_ensemble.py`): 15+ models including deep learning, decomposition, Gemini AI.

**Ultimate Ensemble Result** (the key finding):
- **MA_Momentum won with 95.5% weight** (RMSE=0.495, R²=0.571)
- GDELT_Ridge got 4.5% weight
- ALL deep learning models had negative R² (worse than random):
  - LSTM: R²=-1.484
  - GRU: R²=-1.347
  - Attention: R²=-11.493
  - Gemini_Adjusted: R²=-74.494 (!)

**Interpretation**: For month-ahead FX prediction with this data, simple momentum + mean reversion beats everything.

---

### 4.4 Monte Carlo Simulation (`monte_carlo_simulation/`)
Four simulation approaches, all using 10,000 paths over 30 days:
1. **Geometric Brownian Motion** (GBM): Standard baseline
2. **Regime-Switching**: High/low vol states from political news patterns
3. **Jump Diffusion** (Merton): GBM + random jumps for sudden events
4. **Sentiment-Augmented**: GBM with Goldstein-adjusted drift

Also includes a **weekly rolling forecast** system for operationalization.

---

### 4.5 Advanced Statistical Models (`advanced_exchange_rate_model.py`)
Full econometric framework:
- Multiple Linear Regression (OLS) with diagnostic tests (Durbin-Watson, Breusch-Pagan)
- Vector Autoregression (VAR) with optimal lag selection
- Granger Causality tests (Goldstein → Exchange Rate)
- Random Forest + XGBoost with 80+ engineered features
- Stationarity tests (ADF)

---

### 4.6 Pattern Discovery (`meaningful_patterns_analysis.py`)
Instead of exact prediction, discovers:
1. **Directional prediction** (UP/DOWN/STABLE) using RF classifier
2. **Volatility prediction** (high/low vol regime classification)
3. **Sentiment threshold effects** (extreme positive vs negative sentiment impact)
4. **Lead-lag relationships** (optimal lag for Goldstein/Tone → Price)
5. **Market regime detection** (4 regimes: Turbulent_Weakening, Turbulent_Strengthening, Calm_Weak, Calm_Strong)
6. **Event density impact** (high event days → higher volatility)

---

## 5. KEY FINDINGS (What We Actually Proved)

### 5.1 News Sentiment DOES Lead Exchange Rates
- Political sentiment leads exchange rate movements by **3-4 days** (lag analysis)
- Goldstein_Weighted and Tone_Economy are the strongest GDELT predictors
- Volume spikes (article count changes) correlate with volatility

### 5.2 Volatility is the #1 Predictor
- Realized_Vol_20d = 27.5% feature importance in hybrid model
- EGARCH confirms asymmetric volatility (leverage effect γ=0.20)
- Bad news increases vol more than good news of same magnitude

### 5.3 Simple Models Beat Complex Ones
- MA_Momentum: R²=0.571, RMSE=0.495
- LSTM: R²=-1.484 (worse than predicting mean!)
- Attention: R²=-11.493
- Deep learning overfits on small financial datasets

### 5.4 LLMs Add Value Only Short-Term
- Gemini V4 best at 1-3 day predictions (~0.3% error)
- For 2-4 week horizon, statistical models dominate
- Optimal LLM contribution is 15-30% weight (never >45%)
- Debiasing prompts is CRITICAL (V4 >> V3)

### 5.5 Recommended Model by Time Horizon
| Horizon | Best Model | Expected Error |
|---------|-----------|---------------|
| 1-3 Days | Gemini V4 Daily | 0.04-0.25% |
| 1 Week | Gemini V4 Hybrid | 0.3-0.5% |
| 2-4 Weeks | MA_Momentum Ensemble | 0.5-0.8% |
| 1+ Month | Monte Carlo + GARCH | Wide CIs |

---

## 6. TECHNICAL ARCHITECTURE & DEPENDENCIES

### Tech Stack
- **Python 3.10/3.12** (has both venvs -- some pycache shows 3.10, some 3.12)
- **Google Gemini API** (gemini-2.5-flash) -- via `google-generativeai`
- **ML**: scikit-learn, xgboost, tensorflow/keras
- **Stats**: arch (GARCH), statsmodels (ARIMA, VAR)
- **Data**: pandas, numpy, scipy
- **Viz**: matplotlib, seaborn
- **Signal Processing**: vmdpy (VMD)
- **APIs**: yfinance, FRED API, UN Comtrade, US Census

### File Structure Summary
```
gdelt_india/
├── phase-a/              # VMD signal decomposition
├── Phase-B/              # GDELT thematic filtering + correlation analysis
├── Phase-C/              # Danger signal classifier + hybrid predictor
├── GARCH/                # EGARCH + XGBoost hybrid volatility model
├── gemini_preds/         # V1-V5 Gemini multi-persona simulation
├── ensemble_model/       # Basic → Enhanced → Refined → Ultimate ensemble
├── monte_carlo_simulation/  # MC simulation with regime switching
├── india_usa_trade/      # Trade data analysis (Comtrade/Census)
├── scripts/data_collection/  # FRED, India Commerce, US Census fetchers
├── sample_datasets/      # Curated sample CSVs for reproducibility
├── data/gold_standard/   # FRED + India Commerce raw data
├── *.py                  # Root-level analysis scripts
├── research_paper.tex    # LaTeX paper with TikZ diagrams
├── MODEL_PRESENTATION.md # Model architecture documentation
├── RESEARCH_PRESENTATION.md  # Full research narrative
└── report.md             # Paper template (ACL/ARR format)
```

---

## 7. KNOWN WEAKNESSES & IMPROVEMENT OPPORTUNITIES

### Data Issues
- GDELT keyword matching is crude (substring match, not NLP-based). Could use embeddings.
- Commerce data is annual but used as constant for daily model. Limited value.
- No intraday data -- daily close only. Misses within-day sentiment reactions.

### Model Issues
- Deep learning models (LSTM/GRU) overfit badly. Need more data or regularization.
- EGARCH + XGBoost hybrid uses forward-fill for missing FRED data -- creates lookahead risk.
- Gemini API rate limiting causes missing persona responses on some days.
- V4 entropy/confidence calculation needs more robust handling of edge cases.
- No proper cross-validation for the ensemble weight optimization (single train/test split).

### Methodological Gaps
- No proper **ablation study** showing contribution of each component
- No **statistical significance tests** (bootstrap, paired t-test) on model comparisons
- Granger causality tested but not fully integrated into the prediction pipeline
- Missing **out-of-sample** walk-forward validation for ensemble (only split-based)
- `report.md` is still the blank ACL template -- paper not written yet

### Potential Improvements
1. **Better NLP**: Use sentence transformers for thematic classification instead of keyword matching
2. **Regime-aware ensemble**: Switch model weights based on detected volatility regime
3. **Real-time pipeline**: Stream GDELT data + auto-trigger predictions
4. **More debiasing**: Add calibration layer that adjusts LLM outputs based on historical bias
5. **Multi-currency**: Extend to EUR/INR, GBP/INR, JPY/INR for diversification analysis
6. **Alternative LLMs**: Test Claude, GPT-4o alongside Gemini for persona robustness

---

## 8. CRITICAL NUMBERS TO REMEMBER

| Metric | Value | Context |
|--------|-------|---------|
| Total news articles processed | 7.5M+ | India + USA GDELT |
| Sentiment lead time | 3-4 days | Goldstein → Price lag |
| EGARCH asymmetry (γ) | 0.20 | Bad news > good news for vol |
| V4 avg prediction error | ~0.3% | Full year backtest |
| V4 direction accuracy | 54-58% | Random = 50% |
| V4 stat weight range | 55-85% | Adaptive |
| Optimal ensemble | 95.5% MA + 4.5% GDELT | After 15+ model comparison |
| LSTM R² | -1.484 | Worse than random |
| Best 1-day error | 0.04% | Gemini daily on Dec 22, 2025 |
| Monte Carlo sims | 10,000 paths | 30-day horizon |
| XGBoost #1 feature | Realized_Vol_20d (27.5%) | Hybrid model |
| XGBoost #2 feature | India_Avg_Goldstein (15.8%) | Hybrid model |

---

## 9. FOR THE PAPER (What To Emphasize)

### Strong Claims We Can Make:
1. Political sentiment **leads** exchange rate movements by 3-4 days (with Granger causality support)
2. EGARCH confirms **asymmetric volatility** in INR (γ=0.20, leverage effect)
3. Debiased LLM prompting (V4) significantly improves prediction vs naive prompting (V3)
4. Simple models (MA) beat deep learning (LSTM, GRU) for FX prediction on this scale
5. Multi-persona LLM ensemble adds value for short-term but not long-term horizons

### Claims We Should Be Careful About:
- Direction accuracy improvement (54-58% vs 50% random) -- small, needs significance test
- "7.5M articles" processed -- most are filtered out; actual signal comes from ~100K economy-tagged articles
- V4 Jan 2026 "live" test -- only 22 trading days, too small for robust conclusions
- R²=0.571 for best ensemble -- still explains only 57% of variance

### Paper Structure (for `research_paper.tex` / `report.md`):
The LaTeX paper exists and has TikZ diagrams for the pipeline. The `report.md` is still the blank template. Priority: fill in the experimental results section with proper tables.

---

*These notes represent my understanding of the complete codebase as of the latest commit (a245f0a). Refer back to this document when planning improvements, writing the paper, or onboarding new team members.*

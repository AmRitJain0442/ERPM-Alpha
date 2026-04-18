# Final Phase Architecture

## 1. Design Goals

This module is designed around five requirements:

1. Use real repo data, not synthetic placeholders.
2. Combine financial, statistical, ML, and multi-persona reasoning.
3. Add a shared memory structure so personas and models see the same market state.
4. Reduce bias from immediate price anchoring.
5. Produce reproducible, fully logged ablations.

## 2. Data Layers

The system integrates four main layers:

### Base Macro / Cross-Asset Layer

Source: `Super_Master_Dataset.csv`

Fields:

- `INR`
- `OIL`
- `GOLD`
- `US10Y`
- `DXY`
- India and US tone/stability/mentions/panic features

### Bilateral Goldstein Layer

Source: `combined_goldstein_exchange_rates.csv`

Fields:

- India and USA Goldstein/event summaries
- combined/bilateral sentiment differential features

### Thematic News Layer

Source: `Phase-B/merged_training_data.csv`

Fields:

- `Tone_Economy`
- `Tone_Conflict`
- `Tone_Policy`
- `Tone_Corporate`
- `Goldstein_Weighted`
- `Volume_Spike`
- `IMF_3`

### Political-News Aggregation Layer

Source: `political_news_exchange_merged.csv`

Fields:

- daily Goldstein mean/std
- event counts
- tone mean/std
- mention/article counts

## 3. Shared Market Memory

Each prediction date gets a structured market memory snapshot.

The memory has four roles:

1. compress recent history into stable state variables
2. give every persona the same market state
3. provide derived memory features to the numeric ensemble
4. store persona calibration history across the backtest

### Memory Components

#### Macro Memory

- oil pressure
- dollar pressure
- rate pressure
- gold/risk signal
- carry pressure

#### Sentiment Memory

- India stress
- US stress
- bilateral divergence
- thematic sentiment pressure
- event heat

#### Volatility Memory

- realized volatility
- volatility ratio
- trend pressure
- reversal pressure
- EGARCH or fallback volatility estimate

#### Persona Memory

- directional hit rate
- signed bias
- average confidence
- recent magnitude calibration

## 4. Model Stack

The numeric side uses a model suite rather than one model.

### Regressors

- Ridge regression
- Random forest regressor
- Gradient boosting / XGBoost regressor

These predict forward return, not raw price.

### Direction Model

- multinomial logistic regression

This predicts:

- up
- flat
- down

### Volatility Model

- EGARCH if `arch` is installed and data is sufficient
- fallback to realized volatility if not

### Ensemble Logic

Base-model weights are learned from a validation slice inside the training window.

Higher validation error -> lower ensemble weight.

The direction model is used as a consistency check on the regression blend.

## 5. Persona Stack

The persona stack has two interchangeable modes:

### Rule Personas

Deterministic heuristics with no API dependency.

They provide:

- a strong non-LLM baseline
- cheap 100-day ablations
- transparent reasoning

### LLM Personas

Personas receive the market memory snapshot instead of raw immediate price.

They output JSON:

- direction
- magnitude
- confidence
- thesis
- risk flags

Supported backends:

- OpenRouter
- Ollama

## 6. Bias Controls

This module explicitly avoids a trivial persistence setup.

### Controls

- raw `INR` is not a direct feature
- raw immediate previous close is not fed to personas
- target is shifted by `forecast_horizon_days + embargo_days`
- walk-forward evaluation uses strict temporal ordering
- model weights are learned only from historical validation slices

## 7. Ablation Plan

Default ablations:

- `stat_ml_full`
- `stat_ml_no_gdelt`
- `stat_ml_no_macro`
- `memory_only_no_personas`
- `rule_personas`

LLM ablations are added dynamically when `--llm-model ...` is provided.

Each LLM model gets its own experiment.

## 8. Evaluation Protocol

Default:

- minimum train size: 120 rows
- validation window inside training: 30 rows
- test window: 100 rows
- monthly-style refit cadence: every 10 rows

Metrics:

- MAE on future price
- RMSE on future price
- MAE on forward return
- directional accuracy
- sign F1-like harmonic score
- bias

## 9. Logging Philosophy

This system records more than headline metrics.

For every prediction date it saves:

- ensemble component outputs
- model weights
- direction probabilities
- persona votes
- persona reasoning
- market memory snapshot
- actual outcome

The idea is that any surprising result should be auditable after the run.

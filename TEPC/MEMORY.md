# TEPC Memory

This file is a local memory note for the INR/USD TEPC stack.

Last refreshed: 2026-04-18

## Purpose

`TEPC/` is the chaos-topology forecasting branch for INR/USD. It is separate
from `final phase/` and `latest/` because the mathematical framing is
different:

- dynamic macro network instead of flat feature-only modeling
- rolling graph topology from cross-asset shocks
- coupled Lorenz oscillators for synchronization features
- explicit ablations between macro, topology, chaos, and combined stacks

## Main Entry Points

- `TEPC/pull_data.py`
  - pulls and stores fresh market data under `TEPC/data/`
  - tries public GDELT timeline endpoints
  - if GDELT throttles, falls back to a local daily panel derived from
    `Phase-B/merged_training_data.csv`,
    `combined_goldstein_exchange_rates.csv`, and
    `political_news_exchange_merged.csv`

- `TEPC/run.py`
  - executes the walk-forward TEPC backtest
  - writes full artifacts under `TEPC/outputs/`

- `TEPC/plot_results.py`
  - generates comparison charts for one or more saved TEPC runs

## Current Stored Feed Layout

The canonical stored feed files are:

- `TEPC/data/market_nodes_daily.csv`
- `TEPC/data/gdelt_daily.csv`
- `TEPC/data/combined_feed.csv`
- `TEPC/data/metadata.json`

## Important Data Behavior

The loader merges the repo's historical datasets with the stored pulled feed.
The merge policy prefers the curated historical columns when both exist, but
the pulled market feed is allowed to extend the time axis beyond the historical
master dataset.

Current practical result:

- base historical master coverage remains intact
- live market columns extend the panel into 2026-04
- sparse optional nodes are only kept when they do not destroy the minimum
  history needed for the requested backtest

## Current Node Policy

Core nodes:

- `INRUSD`
- `DXY`
- `BRENT`
- `GOLD`
- `US10Y`
- `INDIA_SENTIMENT`
- `US_SENTIMENT`
- `GOLDSTEIN_SPREAD`
- `GEO_RISK`
- `THEME_PRESSURE`

Optional nodes:

- `EURUSD`
- `GBPINR`
- `USDCNH`
- `CNHUSD`
- `LIVE_INDIA_FX_NEWS`
- `LIVE_USD_MACRO_NEWS`
- `LIVE_GEO_RISK`

Selection rule:

- keep optional nodes only if they have enough tail coverage
- if optional nodes reduce the available history below the required walk-forward
  budget, drop the most restrictive optional nodes first
- normalize composite geopolitical nodes to neutral zero outside their observed
  coverage instead of allowing them to collapse the whole panel

## Latest Pulled Data Snapshot

From `TEPC/data/metadata.json` on 2026-04-18:

- start date: `2024-12-01`
- end date: `2026-04-18`
- market rows: `504`
- gdelt rows: `255`

Current warning:

- public GDELT API rate-limited the live timeline pull, so the local fallback
  GDELT panel was used

## Raw GDELT Reality Check

The exact raw files requested for TEPC strict runs were verified on 2026-04-19:

- `usa_news_combined_sorted.csv` covers `2025-01-01` to `2025-12-30`
- `india_news_gz_combined_sorted.csv` covers `2025-01-01` to `2025-12-28`

So those two files do not currently contain 2023 rows. A pre-2025 strict
cross-country raw TEPC run would require an older USA raw file.

## Strict Lag Configuration

The stricter TEPC setup now uses:

- `response_lag_days = 2`
- `forecast_horizon_days = 1`
- direct flat INR/USD lag, z-score, range-position, and momentum features
  removed from the macro feature stack

Interpretation:

- a row observed on date `t` predicts the first labeled market response at
  `t+2`
- this avoids exposing the model to the target day's `T-1` price
- topology and chaos still include INR/USD as the target node in the network,
  but not as direct tabular lag features in the macro baseline

## Latest Strict Raw Run

Run directory:

- `TEPC/outputs/strict_raw_2025_80d/`

Inputs:

- market pull from `2025-01-01` to `2025-12-31`
- raw GDELT aggregation from the local India and USA files
- `gdelt_source = local_raw`

Dataset summary:

- feature rows: `214`
- feature range: `2025-02-13` to `2025-12-24`
- node count: `15`
- live raw-news nodes retained:
  - `LIVE_INDIA_FX_NEWS`
  - `LIVE_USD_MACRO_NEWS`
  - `LIVE_GEO_RISK`

Metric ranking:

1. `tepc_full`
   - breakout accuracy: `0.775`
   - macro F1: `0.33701`
   - MAE return: `0.0027458`
2. `topology_chaos`
   - breakout accuracy: `0.6875`
   - macro F1: `0.30303`
   - MAE return: `0.0029907`
3. `macro_baseline`
   - breakout accuracy: `0.7625`
   - macro F1: `0.28842`
   - MAE return: `0.00278195`
4. `chaos_only`
   - breakout accuracy: `0.7375`
   - macro F1: `0.28297`
   - MAE return: `0.00307979`
5. `topology_only`
   - breakout accuracy: `0.6875`
   - macro F1: `0.27569`
   - MAE return: `0.00335159`

Interpretation to remember:

- after enforcing the 2-day lag and using raw local GDELT, the full TEPC blend
  becomes the best experiment
- the stricter setup lowers the easy direction score ceiling versus the looser
  run, but it still produces a usable 80-day backtest
- the raw daily GDELT panel is dense enough to keep all three live news nodes
  in the final graph

## Latest 100-Day Backtest

Run directory:

- `TEPC/outputs/livepull_100d/`

Dataset summary:

- feature rows: `321`
- feature range: `2025-01-14` to `2026-04-10`
- node count: `12`
- included optional FX nodes: `EURUSD`, `GBPINR`

Metric ranking:

1. `chaos_only`
   - breakout accuracy: `0.80`
   - macro F1: `0.40398`
   - MAE return: `0.003454`
2. `topology_chaos`
   - breakout accuracy: `0.71`
   - macro F1: `0.37155`
   - MAE return: `0.003448`
3. `topology_only`
   - breakout accuracy: `0.68`
   - macro F1: `0.37144`
   - MAE return: `0.003490`
4. `tepc_full`
   - breakout accuracy: `0.75`
   - macro F1: `0.35645`
   - MAE return: `0.003430`
5. `macro_baseline`
   - breakout accuracy: `0.72`
   - macro F1: `0.28235`
   - MAE return: `0.003627`

Interpretation to remember:

- chaos features are currently the strongest classifier in this branch
- the full stack has the best MAE among the richer blended models, but not the
  best directional F1
- adding every available feature does not automatically dominate a cleaner
  chaos-driven specification

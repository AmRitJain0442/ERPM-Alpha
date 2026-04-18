# TEPC

This folder contains a new INR/USD-focused implementation of a
Topology-Enabled Predictions from Chaos (TEPC) pipeline.

The design goal is to stop treating USD/INR as an isolated time series and
instead model it as one node inside a global macro network shaped by:

- dollar strength
- crude oil
- gold
- rates
- GDELT-derived geopolitical pressure
- India and US sentiment differentials

The current implementation is local-data-first so it can run immediately on
this repo, while keeping the architecture open for live API ingestion later.
It now also supports a stricter lagged setup where observations at date `t`
predict the first market response at `t+2`, so the model does not see the
target day's `T-1` price.

## What The Module Does

The TEPC pipeline:

1. loads the repo's INR/USD, macro, FRED, Goldstein, thematic, and political datasets
2. constructs a daily node panel for INR/USD and its macro/geopolitical drivers
3. transforms node series into comparable shock series
4. computes a rolling correlation-driven network topology
5. builds a weighted adjacency matrix with an exponential distance kernel
6. extracts persistent Laplacian style graph features across a filtration
7. injects deterministic Lorenz oscillators into each node
8. couples the oscillators through the graph Laplacian and records synchronization behavior
9. trains ML ensembles to predict:
   - INR/USD breakout direction
   - future return
   - near-term realized volatility
10. runs walk-forward ablations and records every prediction

## Local Node Universe

The runnable local version uses nodes that are already present in this repo:

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

This is enough to prototype the TEPC architecture on real repo data right now.

When raw local GDELT files are present, the pipeline also adds:

- `LIVE_INDIA_FX_NEWS`
- `LIVE_USD_MACRO_NEWS`
- `LIVE_GEO_RISK`

## Important Modeling Choice

The prompt asked for a persistent Laplacian coupling stage. The current code
implements a pragmatic graph-Laplacian-first version:

- rolling correlation -> distance -> weighted adjacency
- weighted graph Laplacian for the coupled Lorenz engine
- filtration over edge-strength quantiles
- persistent topological summaries from the Laplacian spectrum

This is not yet a full higher-order simplicial persistent Laplacian stack.
It is the correct engineering compromise for a fast, inspectable first version.

## Default Experiments

- `macro_baseline`
- `topology_only`
- `chaos_only`
- `topology_chaos`
- `tepc_full`

These are meant to answer:

- does plain macro still dominate?
- does topology add signal?
- does chaos synchronization add signal?
- does the full TEPC blend beat simpler baselines?

## Outputs

Each run writes to `TEPC/outputs/<timestamp or explicit output dir>/`:

- `config.json`
- `dataset_summary.json`
- `feature_groups.json`
- `node_panel.csv`
- `node_transforms.csv`
- `feature_frame.csv`
- `ablation_summary.csv`
- `metrics.json`
- `run_summary.md`
- plus per-experiment `daily_predictions.csv`
- optional plot outputs under `plots/` when generated with `plot_results.py`

## Running

From the repository root:

```powershell
python "TEPC\pull_data.py" --start-date 2025-01-01 --end-date 2025-12-31
python "TEPC\run.py" --help
python "TEPC\run.py"
python "TEPC\run.py" --test-days 60 --experiment tepc_full
python "TEPC\plot_results.py" --run-dir "TEPC\outputs\livepull_100d"
```

`pull_data.py` stores fresh market and GDELT-derived feeds under `TEPC/data/`:

- `market_nodes_daily.csv`
- `gdelt_daily.csv`
- `combined_feed.csv`
- `metadata.json`

When those files are present, `run.py` automatically uses them to extend the
TEPC node graph with:

- `EURUSD`
- `GBPINR`
- `USDCNH`
- `CNHUSD`
- live GDELT-derived news pressure nodes

The current priority order for GDELT is:

1. local raw aggregation from:
   - `india_news_gz_combined_sorted.csv`
   - `usa_news_combined_sorted.csv`
2. public GDELT timeline API
3. local fallback panel built from the repo's thematic, political, and
   Goldstein datasets

Important date reality:

- `usa_news_combined_sorted.csv` currently covers `2025-01-01` to `2025-12-30`
- `india_news_gz_combined_sorted.csv` currently covers `2025-01-01` to `2025-12-28`

So with those exact files, strict raw GDELT backtests are effectively 2025
experiments unless additional earlier USA raw files are added.

## Strict Timing

The stricter configuration uses:

- `response_lag_days = 2`
- no direct flat-feature use of INR/USD lag or momentum terms
- target response beginning two trading days after the observation date

That means a row dated `t` predicts the first labeled response at `t+2`, which
prevents the model from depending on the target day's immediately prior price.

## Plotting

`plot_results.py` generates comparison charts for an existing TEPC run:

- metric bar panels
- accuracy-vs-error scatter
- actual-vs-predicted overlay for the best experiments
- rolling absolute error comparison

Example:

```powershell
python "TEPC\plot_results.py" `
  --run-dir "TEPC\outputs\livepull_100d"
```

## Default Interpretation

- `predicted_label = up` means INR weakens and USD/INR rises
- `predicted_label = down` means INR strengthens and USD/INR falls
- volatility is modeled as forward realized absolute return intensity

## Where Native Acceleration Fits Later

The heavy loop is the coupled Lorenz RK4 integration. The current version is
vectorized NumPy for quick iteration. If this becomes the bottleneck, the
natural next step is a native RK4 kernel in C++ or Rust with Python bindings.

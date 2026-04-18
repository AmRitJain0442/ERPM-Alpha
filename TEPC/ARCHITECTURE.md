# TEPC Architecture For INR/USD

## 1. Objective

Build an INR/USD forecasting system that treats the market as a coupled
macroeconomic network rather than a single price series.

The pipeline combines:

- macro and cross-asset node construction
- rolling network topology
- chaotic oscillator coupling
- synchronization feature extraction
- ML forecasting

## 2. Data Layers

The current implementation uses repo-local data first.

### Core Market Layer

Source: `Super_Master_Dataset.csv`

Provides:

- INR/USD proxy level
- Brent proxy
- Gold
- US 10Y
- DXY
- India/US sentiment differential fields

### Goldstein Layer

Source: `combined_goldstein_exchange_rates.csv`

Provides:

- bilateral Goldstein summaries
- weighted combined Goldstein pressure
- USA/India sentiment differential

### Thematic Layer

Source: `Phase-B/merged_training_data.csv`

Provides:

- thematic tone blocks
- volume spike fields
- IMF-linked stress signal

### Political Layer

Source: `political_news_exchange_merged.csv`

Provides:

- event count
- tone dispersion
- Goldstein mean/std
- mention/article intensity

### FRED Macro Layer

Source: `data/gold_standard/fred/fred_wide_format_20251230_021943.csv`

Used mostly as fallback or enrichment for:

- DGS10
- DFF
- DTWEXBGS
- DCOILWTICO
- DEXINUS

## 3. Node Construction

Nodes are either direct market series or engineered macro/geopolitical pressure nodes.

### Direct Nodes

- `INRUSD`
- `DXY`
- `BRENT`
- `GOLD`
- `US10Y`

### Composite Nodes

- `INDIA_SENTIMENT`
- `US_SENTIMENT`
- `GOLDSTEIN_SPREAD`
- `GEO_RISK`
- `THEME_PRESSURE`

These composite nodes let the network ingest alternative data without forcing
everything into the same semantics as a traded asset.

## 4. Shock Transformation

For strictly positive market series, the code uses log returns.

For signed alternative data series such as tone or Goldstein-derived pressure,
the code uses signed log-differences:

- `sign(diff(x)) * log(1 + abs(diff(x)))`

This preserves directionality while keeping magnitudes comparable.

## 5. Dynamic Topology

For each decision date:

1. take a rolling window of node shock history
2. compute the node correlation matrix
3. convert correlations into distances
4. apply the exponential kernel

The resulting adjacency matrix is:

- dense
- weighted
- time-varying

This is the daily shape of the macro system.

## 6. Persistent Laplacian Layer

The implementation uses a graph-Laplacian filtration:

1. build the weighted Laplacian from the adjacency matrix
2. threshold the graph across edge-strength quantiles
3. recompute the Laplacian at each filtration level
4. extract spectral summaries

Recorded topological features include:

- target-node degree
- network density
- Fiedler value
- Laplacian trace
- spectral entropy
- connected-component count across thresholds
- filtered Fiedler values across thresholds
- INR/USD adjacency weights to each other node

## 7. Chaotic Engine

Every node receives a Lorenz oscillator:

- `dx/dt = sigma(y - x) + coupling`
- `dy/dt = x(rho - z) - y + coupling`
- `dz/dt = xy - beta z + coupling`

Initial conditions are deterministic functions of:

- recent node volatility
- recent node drift
- range position

The coupling term is diffusive and based on the graph Laplacian, so highly
connected nodes can physically pull on the INR/USD oscillator.

## 8. Filtration Over Coupling Strength

The system increases coupling strength over a schedule of epsilon values.

For each epsilon stage it records:

- synchronization intensity
- target-node synchronization
- target-node terminal state
- phase response

It also estimates a finite-time Lyapunov style divergence score using a
perturbed shadow system.

## 9. Forecasting Layer

The forecasting task is split into three outputs:

- breakout direction classification
- forward return regression
- forward volatility regression

The ensemble is made from standard ML models because they are stable and easy
to run inside walk-forward backtests:

- logistic regression
- random forest
- gradient boosting
- optional XGBoost when available

## 10. Ablation Philosophy

The ablations are designed to isolate where the signal is coming from:

- `macro_baseline`: macro plus alternative-data baseline without topology or chaos
- `topology_only`: only graph-topology features
- `chaos_only`: only chaos synchronization features
- `topology_chaos`: coupled physics without the broader macro baseline
- `tepc_full`: macro + alt + topology + chaos

## 11. Native Optimization Path

The obvious optimization target is the RK4 loop in the coupled Lorenz engine.

If runtime becomes limiting:

1. move the RK4 kernel to C++ or Rust
2. expose it through Python bindings
3. keep data prep, experiment logic, and ML in Python

That split preserves fast experimentation while accelerating the true hot path.

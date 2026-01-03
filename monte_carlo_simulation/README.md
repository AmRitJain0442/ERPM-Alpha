# Monte Carlo Simulation for USD/INR Exchange Rate

## Overview

This folder contains Monte Carlo simulations for USD/INR exchange rate forecasting, incorporating meaningful patterns discovered from political news sentiment analysis.

## What is Monte Carlo Simulation?

Monte Carlo simulation is a probabilistic forecasting method that:
- Runs thousands of different scenarios (10,000 in this case)
- Uses random sampling to account for uncertainty
- Provides probability distributions instead of single-point predictions
- Gives confidence intervals and risk metrics

**Better than point predictions because:**
- Exchange rates are inherently uncertain
- Provides range of possible outcomes with probabilities
- Useful for risk management and scenario planning

## Models Implemented

### 1. **Standard GBM (Geometric Brownian Motion)**
- Classic financial model: dS = μS dt + σS dW
- Uses historical mean return and volatility
- Assumes constant drift and volatility

### 2. **Regime-Switching Model** ⭐ RECOMMENDED
- Switches between high/low volatility states
- Uses political news patterns to define regimes
- Parameters:
  - High volatility: std=0.0035 (from turbulent periods)
  - Low volatility: std=0.0020 (from calm periods)
  - Transition probabilities based on observed patterns

### 3. **Jump Diffusion Model**
- Adds sudden "jumps" to standard GBM
- Models extreme events (political shocks, policy announcements)
- Jump probability: 25% (based on high event days)
- Jump mean: -0.005 (slight negative bias)

### 4. **Sentiment-Augmented Model**
- Adjusts drift based on recent political sentiment
- Recent 7-day average Goldstein score used
- Sentiment impact coefficient: 0.0005

## Key Results (30-Day Forecast)

**Current Rate:** 89.90 INR

### Regime-Switching Model (Recommended):

| Scenario | Rate | Change |
|----------|------|--------|
| **Best Case (95th percentile)** | 92.62 INR | +3.02% |
| **Expected (Median)** | 90.37 INR | +0.52% |
| **Worst Case (5th percentile)** | 88.23 INR | -1.85% |

### Probabilities:
- **Increase (rate > 89.90):** 63.94%
- **Decrease (rate < 89.90):** 36.06%
- **Large move (>1 INR):** 46.65%

### Value at Risk (VaR):
- **95% confidence:** -1.85% (won't lose more than this in 95% of scenarios)
- **99% confidence:** -2.90% (won't lose more than this in 99% of scenarios)

## Patterns Incorporated

The simulation uses meaningful patterns discovered from political news analysis:

1. **Volatility Regimes**
   - High volatility periods (std=0.35%) during political turbulence
   - Low volatility periods (std=0.20%) during calm times
   - Transition probabilities: 15% high→low, 10% low→high

2. **Sentiment Effects**
   - Positive sentiment → +0.0507% average drift
   - Negative sentiment → -0.0141% average drift

3. **Event-Driven Jumps**
   - 25% probability of jump on any given day
   - Based on observed high political event density

## Files Generated

1. **monte_carlo_results.png**
   - 9 comprehensive visualizations
   - Sample paths, distributions, confidence intervals
   - Fan charts, scenarios, probability timelines

2. **monte_carlo_statistics.csv**
   - Summary statistics for all 4 models
   - Mean, median, percentiles, min/max

3. **monte_carlo_forecast.csv**
   - Day-by-day forecast for 30 days
   - Mean, median, and confidence intervals
   - Can be used for daily tracking

4. **monte_carlo_report.txt**
   - Detailed interpretation
   - Parameters used
   - Recommendations

## How to Use These Results

### For Risk Management:
```
Use the 5th and 95th percentiles as your risk range.
Plan for worst case: 88.23 INR (-1.85%)
Hope for best case: 92.62 INR (+3.02%)
```

### For Decision Making:
```
Expected outcome: Rate will likely increase to ~90.37 INR
Confidence: 64% probability of increase
Risk: 36% probability of decrease
```

### For Hedging:
```
Value at Risk (95%): -1.85%
This means: In 95% of scenarios, losses won't exceed 1.85%
Use this for sizing hedging positions
```

## Model Comparison

| Model | Mean (30d) | Std | Best | Worst |
|-------|-----------|-----|------|-------|
| **Regime-Switching** ⭐ | 90.38 | 1.34 | 92.62 | 88.23 |
| Standard GBM | 90.45 | 1.39 | 92.74 | 88.17 |
| Sentiment-Augmented | 91.25 | 1.40 | 93.53 | 88.95 |
| Jump Diffusion | 87.22 | 3.94 | 93.61 | 80.71 |

**Recommendation:** Use Regime-Switching model as it:
- Captures volatility clustering observed in data
- Incorporates political news patterns
- More realistic than constant volatility GBM
- Less extreme than jump diffusion

## Running the Simulation Again

```bash
cd monte_carlo_simulation
python monte_carlo_exchange_rate.py
```

**Customization options in the code:**
- `T = 30`: Change forecast horizon (days)
- `n_simulations = 10000`: Increase for more precision
- `sentiment_impact = 0.0005`: Adjust sentiment effect
- `jump_probability`: Modify based on event expectations

## Interpretation Guidelines

### What the Results Tell You:

✅ **DO USE FOR:**
- Understanding possible range of outcomes
- Risk assessment and scenario planning
- Setting hedging strategies
- Identifying tail risks
- Confidence intervals for decisions

❌ **DON'T USE FOR:**
- Exact price predictions
- Guaranteed outcomes
- Short-term day-trading signals
- Ignoring fundamental factors

### Limitations:

1. Based on 1 year of historical data
2. Assumes patterns continue into future
3. Cannot predict black swan events
4. Political landscape may change
5. Model parameters estimated from limited sample

### Confidence Levels:

- **50% Confidence Interval:** 89.51 to 91.26 INR (rate likely in this range)
- **90% Confidence Interval:** 88.23 to 92.62 INR (very likely in this range)
- **Full Range:** 85.27 to 96.74 INR (observed in simulations)

## Advanced Usage

### Updating Parameters:

If you have new information, update these in the code:

```python
# Recent sentiment (last 7 days average)
recent_sentiment = political_df['GoldsteinScale_mean'].iloc[-7:].mean()

# Regime transition probabilities
p_high_to_low = 0.15  # Adjust based on current regime stability
p_low_to_high = 0.10

# Jump parameters (if expecting high political activity)
jump_probability = 0.25
```

### Extending Forecast Horizon:

```python
T = 60  # For 60-day forecast instead of 30
```

Note: Uncertainty increases with longer horizons.

## Contact & Further Analysis

For questions or custom scenarios, modify the simulation parameters or run additional analyses on the generated CSV files.

---

**Last Updated:** 2026-01-03
**Model Version:** 1.0
**Data Period:** 2025 (1 year)

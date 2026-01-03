# Weekly Rolling Forecast Guide

## 🎯 What This Does

**Run ONCE, Get forecasts for EVERY WEEK:**
- Week 1 forecast (next 7 days)
- Week 2 forecast (days 8-14)
- Week 3 forecast (days 15-21)
- Week 4 forecast (days 22-28)

Each week's forecast builds on the previous week, showing how uncertainty grows over time.

---

## 📊 Current Forecast Results

### **Current State**
- **Date:** 2025-12-29
- **Exchange Rate:** 89.90 INR

### **Weekly Forecasts**

| Week | Dates | Expected | Best Case | Worst Case | Prob↑ |
|------|-------|----------|-----------|------------|-------|
| **1** | Dec 30 - Jan 5 | 90.01 (+0.1%) | 91.16 | 88.92 | **57%** |
| **2** | Jan 6 - Jan 12 | 90.12 (+0.2%) | 91.73 | 88.58 | **59%** |
| **3** | Jan 13 - Jan 19 | 90.24 (+0.4%) | 92.20 | 88.34 | **61%** |
| **4** | Jan 20 - Jan 26 | 90.36 (+0.5%) | 92.61 | 88.13 | **63%** |

---

## 📈 Key Insights

### Week 1 (Most Reliable)
- **Expected:** 90.01 INR
- **90% Confidence Range:** 88.92 to 91.16 INR
- **Probability of increase:** 56.8%
- **Value at Risk (95%):** -1.09% maximum likely loss
- **Uncertainty:** ±1.12 INR

**Use for:** Immediate decisions, short-term planning, hedging strategies

### Week 2 (Moderate Uncertainty)
- **Expected:** 90.12 INR
- **90% Confidence Range:** 88.58 to 91.73 INR
- **Probability of increase:** 58.8%
- **Probability of large move (>1 INR):** 30.4%
- **Uncertainty:** ±1.57 INR (40% wider than Week 1)

**Use for:** Medium-term budgeting, contract pricing

### Week 3-4 (Scenario Planning)
- **Uncertainty grows significantly**
- Week 4 range: 88.13 to 92.61 (4.48 INR range!)
- Still useful for scenario planning
- Don't rely on point estimates

**Use for:** Long-term scenario analysis, risk assessment

---

## 🎲 How It Works

### Sequential Simulation Approach

```
Week 1:
├─ Start: Current price (89.90)
├─ Simulate: 10,000 scenarios × 7 days
└─ Result: 10,000 different ending prices

Week 2:
├─ Start: Use Week 1's 10,000 ending prices
├─ Simulate: 7 more days from each
└─ Result: New distribution (wider uncertainty)

Week 3:
├─ Start: Use Week 2's distribution
├─ Simulate: 7 more days
└─ Result: Even wider distribution

Week 4:
└─ Pattern continues...
```

**Why this matters:**
- Uncertainty compounds over time
- More realistic than independent forecasts
- Shows actual probability evolution

---

## 📁 Files Generated

### 1. **weekly_rolling_forecast.png**
Comprehensive visualization with:
- All weekly paths overlaid (color-coded)
- Weekly distribution boxplots
- Confidence intervals over 28 days
- Weekly expected values with error bars
- Uncertainty growth chart
- Probability of increase by week
- Individual week distributions (Weeks 1-3)

### 2. **weekly_forecast_summary.csv**
Summary statistics for all weeks:
- Mean, Median, Std
- Percentiles (5th, 25th, 75th, 95th)
- Expected change %
- Uncertainty ranges

### 3. **week_1_detailed_results.csv** (through week_4)
10,000 simulation results for each week:
- Simulation number
- Final price
- Return %

**Use these for:**
- Custom analysis
- Building your own charts
- Risk calculations

### 4. **weekly_forecast_report.txt**
Actionable summary report with:
- Weekly scenarios
- Probabilities
- Value at Risk
- Usage guidelines

---

## 💼 Practical Applications

### For Weekly Planning:

**Monday Morning Routine:**
```
1. Run: python weekly_rolling_forecast.py
2. Check Week 1 forecast
3. Review probability of increase
4. Set hedging strategy if VaR exceeds threshold
5. Update stakeholders with scenarios
```

### For Risk Management:

**Week 1:**
- VaR 95%: -1.09%
- **Action:** If you have $1M exposure, max likely loss = $10,900
- **Hedge:** Size positions accordingly

**Week 2:**
- VaR 95%: -1.47%
- **Action:** Risk increasing, review hedges

**Week 4:**
- VaR 95%: -1.97%
- **Action:** Don't make long-term commitments without hedges

### For Decision Making:

**Scenario: Should I lock in rate for next week?**
```
Current rate: 89.90
Week 1 median: 90.01
Probability of better rate: 43.2% (100% - 56.8%)
Probability of worse rate: 56.8%

Decision: 57% chance rate increases → Consider locking in
```

**Scenario: Planning 3-week forward contract**
```
Week 3 forecast: 90.24 (median)
Week 3 range: 88.34 to 92.20
Uncertainty: ±1.93 INR

Decision: Wide range! Use worst case (92.20) for planning
```

---

## 🔄 How Often to Run

### Recommended Schedule:

**Daily (if active trading):**
- Quick check of Week 1 forecast
- Monitor if actual rate diverges from forecast
- Update risk positions

**Weekly (for planning):**
- Full run every Monday
- Review all 4 weeks
- Update budgets and hedges

**Ad-hoc (for major events):**
- After major political announcements
- During high volatility periods
- Before large transactions

---

## 🎨 Customization Options

Edit `weekly_rolling_forecast.py`:

### Change Forecast Horizon:
```python
N_WEEKS = 4  # Change to 6 for 6-week forecast
```

### Increase Precision:
```python
N_SIMULATIONS = 10000  # Increase to 50000 for more precision
# Note: Takes longer to run
```

### Adjust Confidence Levels:
```python
CONFIDENCE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
# Add 0.01 and 0.99 for 99% confidence
```

---

## 📊 Understanding Uncertainty Growth

**Why does uncertainty grow?**

Week 1 Uncertainty: ±1.12 INR (Range: 2.24)
Week 2 Uncertainty: ±1.57 INR (Range: 3.15) ← +40% wider
Week 3 Uncertainty: ±1.93 INR (Range: 3.85) ← +22% wider
Week 4 Uncertainty: ±2.24 INR (Range: 4.48) ← +16% wider

**This is normal!**
- Each day adds randomness
- Random effects compound
- Long-term predictions = wider ranges

**Rule of thumb:**
- Week 1: Fairly reliable
- Week 2-3: Use with caution
- Week 4+: Scenario planning only

---

## 🎯 Quick Decision Framework

### For Week 1 Forecast:

**If probability of increase > 60%:**
→ Rate likely to go up, consider hedging/locking in

**If probability of increase < 40%:**
→ Rate likely to go down, wait if possible

**If probability 40-60%:**
→ Uncertain, use median as planning value

### For Value at Risk:

**VaR < 1%:**
→ Low risk, minimal hedging needed

**VaR 1-2%:**
→ Moderate risk, consider partial hedge

**VaR > 2%:**
→ High risk, hedge strongly or reduce exposure

---

## 🔍 Comparing Weekly Forecasts

### Week-over-Week Changes:

| Metric | Week 1 | Week 2 | Week 3 | Week 4 |
|--------|--------|--------|--------|--------|
| Expected | 90.01 | 90.12 | 90.24 | 90.36 |
| Uncertainty | ±1.12 | ±1.57 | ±1.93 | ±2.24 |
| Prob(increase) | 57% | 59% | 61% | 63% |
| VaR 95% | -1.09% | -1.47% | -1.73% | -1.97% |

**Pattern:**
- Expected value slowly increasing
- Uncertainty growing faster than expected change
- Probability of increase rising (bullish trend)
- Risk (VaR) increasing with time

---

## ⚠️ Common Mistakes to Avoid

### ❌ DON'T:
1. **Use Week 4 median as exact prediction**
   - It's just the middle of a wide range!

2. **Ignore uncertainty ranges**
   - The range is as important as the median

3. **Make Week 4 decisions without hedging**
   - Too uncertain, need risk management

4. **Expect actual price to match forecast exactly**
   - Real world has surprises!

### ✅ DO:
1. **Use Week 1 for immediate decisions**
   - Most reliable forecast

2. **Consider full range for planning**
   - Best case, expected, worst case

3. **Monitor probabilities**
   - 60%+ is moderately confident signal

4. **Update weekly**
   - Fresh data = better forecasts

---

## 📞 Interpreting the Results

### Example Interpretation:

**Week 1 Forecast:**
```
Expected: 90.01 INR (+0.12%)
Range: 88.92 to 91.16
Probability of increase: 57%
```

**What this means:**
- "Most likely outcome is small increase to ~90.01"
- "90% confident rate will be between 88.92 and 91.16"
- "Slightly more likely to go up (57%) than down (43%)"
- "For planning, use 91.16 as worst case (conservative)"

### Example for Stakeholders:

**Email Template:**
```
Subject: Weekly Exchange Rate Forecast

Based on Monte Carlo simulation (10,000 scenarios):

Week 1 (Dec 30 - Jan 5):
• Expected: 90.01 INR (slight increase)
• Range: 88.92 to 91.16 (90% confidence)
• Trend: 57% probability of increase

Recommendation:
• Budget conservatively at 91.16
• Monitor daily for significant deviations
• Consider locking in if rate drops below 89.50

Risk Level: LOW
Max likely loss (VaR 95%): 1.09%
```

---

## 🚀 Running the Forecast

### Command:
```bash
cd monte_carlo_simulation
python weekly_rolling_forecast.py
```

### Expected Runtime:
- ~30-60 seconds for 4 weeks, 10,000 simulations
- Increase with more weeks or simulations

### Output:
✓ weekly_rolling_forecast.png
✓ weekly_forecast_summary.csv
✓ week_1_detailed_results.csv
✓ week_2_detailed_results.csv
✓ week_3_detailed_results.csv
✓ week_4_detailed_results.csv
✓ weekly_forecast_report.txt

---

## 🎓 Advanced Usage

### Conditional Forecasts:

**"What if political sentiment gets worse?"**

Modify in code:
```python
# Instead of using recent_sentiment
recent_sentiment = political_df['GoldsteinScale_mean'].iloc[-7:].mean()

# Use a pessimistic value
recent_sentiment = -1.0  # Assume negative sentiment
```

**"What if we enter high volatility regime?"**
```python
# Force high volatility
high_vol_std = high_vol_std * 1.5  # 50% higher volatility
```

### Comparing Multiple Scenarios:

Run multiple times with different parameters, save results with different names:
```bash
# Baseline
python weekly_rolling_forecast.py
mv weekly_forecast_summary.csv baseline_forecast.csv

# Pessimistic scenario (edit code first)
python weekly_rolling_forecast.py
mv weekly_forecast_summary.csv pessimistic_forecast.csv

# Compare in Excel/Python
```

---

## 📚 Further Reading

- **Monte Carlo Methods:** Understanding probabilistic forecasting
- **Value at Risk (VaR):** Risk management metric
- **Regime-Switching Models:** Capturing different market states
- **Exchange Rate Forecasting:** Why it's so difficult!

---

**Last Updated:** 2026-01-03
**Model Version:** Rolling Weekly v1.0
**Data Source:** Political news + historical exchange rates (2025)

---

## 🆘 Troubleshooting

**Q: Week 1 forecast seems wrong**
A: Check if current_price is up to date. Update exchange rate CSV.

**Q: Uncertainty too wide?**
A: Normal for exchange rates. Consider shorter horizon (2-3 weeks).

**Q: Probabilities always near 50%?**
A: Indicates weak signal. Market is uncertain, use wider hedges.

**Q: How accurate are these forecasts?**
A: Week 1 is fairly reliable. Week 4 is scenario planning only.

**Q: Can I trust the median forecast?**
A: It's the most likely outcome, but always consider the range!

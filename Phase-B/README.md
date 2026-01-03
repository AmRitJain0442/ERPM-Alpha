# Phase B: GDELT Thematic Filtering & Feature Engineering

This phase implements **theme-specific filtering** to align GDELT news data with the IMF 3 (Noise) component of the INR/USD exchange rate.

## Overview

Markets don't react to news "averages" — they react to **spikes** and **specific themes**. This pipeline filters your GDELT data for themes that actually impact currency exchange rates.

## What This Does

### 1. Thematic Filtering
Filters 2.5M+ GDELT articles into 4 key themes:

- **Economy**: RBI policy, inflation, taxation, Fed rates, GDP, forex, trade
- **Conflict**: Protests, strikes, geopolitical tensions, border disputes
- **Policy**: Government regulations, bills, cabinet decisions
- **Corporate**: Adani, Reliance, Tata, major corporate events

### 2. Feature Engineering
Creates daily features that correlate with exchange rate noise:

| Feature | Description |
|---------|-------------|
| `Tone_Economy` | Average sentiment of economy-related articles |
| `Tone_Conflict` | Average sentiment of conflict-related articles |
| `Tone_Policy` | Average sentiment of policy-related articles |
| `Tone_Corporate` | Average sentiment of corporate-related articles |
| `Goldstein_Weighted` | Goldstein score weighted by article mentions (high-impact events) |
| `Volume_Spike` | % change in article count vs previous day |
| `Volume_Spike_Economy` | % change in economy article count |
| `Volume_Spike_Conflict` | % change in conflict article count |

**Key Insight**: A Goldstein score of -10 in an article mentioned 5 times is noise. A score of -10 mentioned 5000 times signals a market crash.

## Files

### Scripts
- `thematic_filter.py` - Main filtering and feature engineering pipeline
- `correlation_analysis.py` - Correlation analysis with IMF 3 noise component

### Generated Data
- `india_news_thematic_features.csv` - Daily aggregated features (362 days)
- `correlation_results.csv` - Correlation coefficients with IMF 3
- `correlation_heatmap.png` - Visual correlation analysis

## Usage

### Step 1: Run Thematic Filtering
```bash
cd Phase-B
python thematic_filter.py
```

**Output**: `india_news_thematic_features.csv` with 16 engineered features

**Processing Stats**:
- Input: 2,546,999 GDELT records
- Economy articles: 139,953
- Conflict articles: 560,938
- Policy articles: 594,924
- Corporate articles: 53,779
- Output: 362 daily aggregated records

### Step 2: Correlation Analysis
```bash
python correlation_analysis.py
```

**Requirements**: You need an `IMF_3.csv` file with columns:
- `Date` (datetime)
- `IMF_3` (the noise component from CEEMDAN decomposition)

**Outputs**:
- Console: Correlation coefficients and p-values
- `correlation_results.csv`: Full correlation table
- `correlation_heatmap.png`: Visual heatmap
- Time series plots for top 3 features
- Lag analysis (0-7 days)

### Step 3: Interpret Results

Look for:
1. **Highest absolute correlations** (positive or negative)
2. **Statistical significance** (p-value < 0.05)
3. **Lag effects** (news impact may occur 1-3 days later)
4. **Volume spikes** (often stronger signal than sentiment)

## Understanding the Features

### Sentiment (Tone) Features
- **Range**: -100 (extremely negative) to +100 (extremely positive)
- **Market Impact**: Negative tone in economy news → Currency depreciation
- **Example**: RBI rate hike with negative tone = bearish INR

### Goldstein Scale
- **Range**: -10 (conflictual) to +10 (cooperative)
- **Weighted**: Multiplied by NumMentions for impact
- **Example**: Border conflict (-8.0) mentioned 10,000 times = -80,000 weighted score

### Volume Spikes
- **Range**: % change from previous day
- **Signal**: High volatility regardless of sentiment
- **Example**: 500% spike in economy articles = major event

## Customization

### Add More Keywords
Edit `thematic_filter.py` lines 17-47 to add domain-specific keywords:

```python
self.ECONOMY_KEYWORDS = [
    'economy', 'inflation', 'rbi',
    'your_custom_keyword',  # Add here
]
```

### Change Aggregation Period
Modify `aggregate_daily_features()` to aggregate by week/month instead of day.

### Add New Themes
Add new keyword lists and filtering logic:

```python
self.ENERGY_KEYWORDS = ['oil', 'petrol', 'crude', 'opec']

self.df['IsEnergy'] = self.df['CombinedText'].apply(
    lambda x: self.contains_keywords(x, self.ENERGY_KEYWORDS)
)
```

## Expected Results

### Strong Positive Correlations
- **Volume_Spike_Economy**: Market volatility follows news volume
- **Count_Economy**: More economic news = more market movement

### Strong Negative Correlations
- **Tone_Conflict**: Negative geopolitical news → INR depreciation
- **Goldstein_Weighted**: Conflictual events → Market uncertainty

### Lag Effects
News impact typically shows:
- **0-1 day lag**: Breaking news, central bank announcements
- **2-3 day lag**: Policy changes, corporate scandals
- **No correlation**: Irrelevant themes (water shortages, local protests)

## Next Steps

1. **Identify top 3-5 features** with highest correlation to IMF 3
2. **Build regression model** using these features to predict noise
3. **Test causality**: Does news predict exchange rate or vice versa?
4. **Engineer interaction terms**: `Tone_Economy * Volume_Spike_Economy`
5. **Add sentiment intensity**: Not just average, but variance and extremes

## Troubleshooting

### No IMF_3.csv file
You need to first extract IMF 3 from your exchange rate data using CEEMDAN decomposition.

### Low correlations
- Try different lag periods (news impact may be delayed)
- Check if keywords match your domain
- Ensure date ranges overlap between news and exchange rate data

### Memory issues
For large datasets, process in chunks:
```python
chunks = pd.read_csv(csv_path, chunksize=100000)
for chunk in chunks:
    # Process chunk
```

## References

- GDELT documentation: https://www.gdeltproject.org/data.html
- Goldstein Scale: Conflict-cooperation measurement (-10 to +10)
- IMF decomposition: Intrinsic Mode Functions from CEEMDAN

---

**Author**: Phase B Pipeline
**Date**: 2025-01-03
**Data Coverage**: 2025-01-01 to 2025-12-28 (362 days)

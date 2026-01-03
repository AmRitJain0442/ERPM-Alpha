# Gold Standard Data Sources

This directory contains data collected from the most authoritative sources for US-India trade analysis.

## Overview

These are the primary government sources providing the most accurate trade and economic data:

| Source | What it offers | Best for |
|--------|---------------|----------|
| **US Census Bureau** | Detailed monthly export/import data between US and India | High granularity - see what's being traded (oil, tech, etc.) which impacts currency demand differently |
| **India Ministry of Commerce** | India's official export/import data bank | The Indian perspective - may differ slightly from US data due to reporting lags |
| **FRED** | Aggregated time-series data for trade balance and exchange rates | Quick setup - easiest to plug directly into Python models |

## Directory Structure

```
data/gold_standard/
├── fred/                      # FRED economic data
│   ├── fred_combined_*.csv   # All series combined
│   └── fred_wide_format_*.csv # Wide format (date x series)
│
├── us_census/                 # US Census Bureau trade data
│   ├── us_imports_from_india.csv
│   └── us_exports_to_india.csv
│
└── india_commerce/            # India Ministry of Commerce data
    ├── README.md              # Manual download instructions
    └── manual_download_*.csv  # Manually downloaded files
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

#### FRED API Key (Required)
1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Request an API key
4. Set environment variable:
   ```bash
   # Windows
   set FRED_API_KEY=your_key_here

   # Linux/Mac
   export FRED_API_KEY=your_key_here
   ```

#### US Census Bureau API Key (Optional but Recommended)
1. Visit: https://api.census.gov/data/key_signup.html
2. Request a free API key
3. Set environment variable:
   ```bash
   # Windows
   set CENSUS_API_KEY=your_key_here

   # Linux/Mac
   export CENSUS_API_KEY=your_key_here
   ```

### 3. Run Data Collection

```bash
# Collect all data sources
python scripts/data_collection/collect_all_data.py

# Or run individual collectors
python scripts/data_collection/fetch_fred.py
python scripts/data_collection/fetch_us_census.py
python scripts/data_collection/fetch_india_commerce.py
```

## Data Sources in Detail

### 1. FRED (Federal Reserve Economic Data)

**What it provides:**
- India/US exchange rate (DEXINUS)
- US trade balance (goods and services)
- US exports/imports aggregates
- Economic indicators (GDP, CPI, unemployment)
- Interest rates and money supply
- Oil prices (affects trade)

**Advantages:**
- Easy API access
- Clean, standardized data
- High reliability
- Wide format ready for time series analysis

**Update Frequency:** Varies by series (daily to quarterly)

**Collection Status:** ✓ Automated

### 2. US Census Bureau (International Trade)

**What it provides:**
- Detailed commodity-level trade data
- Monthly export/import values
- HS codes and product descriptions
- Country-specific trade flows

**Advantages:**
- Most granular US trade data
- Official government source
- Commodity-level detail

**Update Frequency:** Monthly (45-60 day lag)

**Collection Status:** ✓ Automated (API-based)

**Notes:**
- India country code: 4100
- Data available from 2010 onwards
- Large dataset - commodity-level detail

### 3. India Ministry of Commerce

**What it provides:**
- India's official export/import statistics
- Country-wise trade data
- Product category breakdowns

**Advantages:**
- Indian government's official data
- May capture different reporting timing
- Useful for cross-validation

**Update Frequency:** Monthly

**Collection Status:** ⚠ Manual download required

**Instructions:**
See `india_commerce/README.md` for detailed download instructions

## Data Usage

### Loading FRED Data (Wide Format)

```python
import pandas as pd

# Load wide format (easiest for time series)
df = pd.read_csv('data/gold_standard/fred/fred_wide_format_*.csv',
                 index_col=0, parse_dates=True)

# Access specific series
exchange_rate = df['DEXINUS']  # India/US exchange rate
trade_balance = df['BOPGSTB']  # US trade balance

# Plot
import matplotlib.pyplot as plt
exchange_rate.plot(title='India/US Exchange Rate')
plt.show()
```

### Loading US Census Data

```python
import pandas as pd

# Load imports
imports = pd.read_csv('data/gold_standard/us_census/us_imports_from_india.csv')

# Load exports
exports = pd.read_csv('data/gold_standard/us_census/us_exports_to_india.csv')

# Analyze by commodity
top_imports = imports.groupby('I_COMMODITY_LDESC')['ALL_VAL_YR'].sum().sort_values(ascending=False)
print(top_imports.head(10))
```

## Data Quality Notes

### Differences Between Sources

1. **Reporting Timing:**
   - US and India may report the same transaction in different months
   - FRED aggregates may use different cutoff dates

2. **Currency Conversion:**
   - US reports in USD
   - India may report in INR then convert
   - Exchange rate timing affects values

3. **Classification:**
   - Different commodity classification systems
   - HS codes may be aggregated differently

4. **Lags:**
   - US Census: 45-60 days
   - India Commerce: Variable
   - FRED: Depends on source series

### Best Practices

1. **Use multiple sources** for validation
2. **Check date alignment** when comparing
3. **Document which source** you use for each analysis
4. **Note the lag** in latest data availability

## Automation

### Scheduled Updates

To keep data current, set up scheduled runs:

```bash
# Linux/Mac (crontab)
# Run monthly on the 15th at 2 AM
0 2 15 * * cd /path/to/gdelt_india && python scripts/data_collection/collect_all_data.py

# Windows (Task Scheduler)
# Create a task that runs collect_all_data.py monthly
```

### Error Handling

The collectors are designed to:
- Continue on individual failures
- Log errors clearly
- Save partial results
- Provide detailed status reports

## Troubleshooting

### "API Key Required" Error

Make sure environment variables are set:
```bash
echo %FRED_API_KEY%        # Windows
echo $FRED_API_KEY         # Linux/Mac
```

### "No Data Collected" Error

Check:
1. Internet connection
2. API key validity
3. API rate limits (wait and retry)
4. Date range (some series may not go back to 2010)

### India Commerce Manual Download Issues

The India trade portal structure may change. If automated collection fails:
1. Visit the portal manually
2. Follow README instructions
3. Download CSV files
4. Place in `india_commerce/` folder
5. Re-run the collection script

## Next Steps

After collecting this data, you can:

1. **Merge with GDELT data** for event-driven analysis
2. **Build time series models** using FRED data
3. **Analyze commodity-specific effects** using Census data
4. **Cross-validate** using Indian perspective
5. **Create features** for machine learning models

## References

- FRED API Docs: https://fred.stlouisfed.org/docs/api/fred/
- Census Trade API: https://www.census.gov/data/developers/data-sets/international-trade.html
- India Trade Stats: https://tradestat.commerce.gov.in/

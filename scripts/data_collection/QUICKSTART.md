# Quick Start Guide - Gold Standard Data Collection

## 5-Minute Setup

### Step 1: Install Dependencies (1 minute)

```bash
cd scripts/data_collection
pip install -r requirements.txt
```

### Step 2: Get API Keys (2-3 minutes)

**FRED API Key** (Required)
1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request API Key"
3. Fill the form (takes 30 seconds)
4. Copy your key

**Census API Key** (Optional but recommended)
1. Go to: https://api.census.gov/data/key_signup.html
2. Enter your email
3. Check your email for the key

### Step 3: Set Environment Variables (30 seconds)

**Windows:**
```cmd
set FRED_API_KEY=your_fred_key_here
set CENSUS_API_KEY=your_census_key_here
```

**Linux/Mac:**
```bash
export FRED_API_KEY=your_fred_key_here
export CENSUS_API_KEY=your_census_key_here
```

**Or use .env file (recommended):**
```bash
# Copy template
cp .env.template .env

# Edit .env and add your keys
notepad .env  # Windows
nano .env     # Linux/Mac
```

### Step 4: Run Setup Verification (30 seconds)

```bash
python setup.py
```

This checks:
- Python version
- Installed packages
- Directory structure
- API keys
- API connections

### Step 5: Collect Data (5-10 minutes)

```bash
python collect_all_data.py
```

This will:
- Fetch FRED economic data (~2-3 minutes)
- Fetch US Census trade data (~3-5 minutes)
- Check for India Commerce manual downloads

## What You Get

After running the collector, you'll have:

```
data/gold_standard/
├── fred/
│   ├── fred_combined_TIMESTAMP.csv       # All FRED series
│   └── fred_wide_format_TIMESTAMP.csv    # Ready for analysis
│
├── us_census/
│   ├── us_imports_from_india.csv         # Detailed import data
│   └── us_exports_to_india.csv           # Detailed export data
│
└── india_commerce/
    └── README.md                          # Manual download instructions
```

## Quick Analysis

Try the example script:

```bash
python example_usage.py
```

This generates:
- Exchange rate trend analysis
- Trade balance analysis
- Top imports from India
- Correlation plots

## Data Overview

### FRED Data (~15 series)
- India/US exchange rate (DEXINUS)
- US trade balance
- Economic indicators (GDP, CPI, unemployment)
- Interest rates
- Oil prices

**Best for:** Time series modeling, macroeconomic analysis

### US Census Data (Commodity-level)
- Monthly export/import values
- HS code classifications
- Product descriptions

**Best for:** Commodity-specific analysis, granular trade patterns

### India Commerce Data (Manual)
- Indian government perspective
- Cross-validation
- Different reporting timing

**Best for:** Validation, alternative perspective

## Common Issues

### "API Key Required"

```bash
# Check if keys are set
echo %FRED_API_KEY%        # Windows
echo $FRED_API_KEY         # Linux/Mac

# If empty, set them again
```

### "No module named 'requests'"

```bash
pip install -r requirements.txt
```

### "No data collected"

1. Check internet connection
2. Verify API keys are correct
3. Check FRED/Census websites are accessible
4. Try running individual collectors:
   ```bash
   python fetch_fred.py
   python fetch_us_census.py
   ```

## Next Steps

1. **Explore the data:**
   ```python
   import pandas as pd

   # Load FRED data
   df = pd.read_csv('../../data/gold_standard/fred/fred_wide_format_*.csv',
                    index_col=0, parse_dates=True)

   # Check available series
   print(df.columns)

   # Plot exchange rate
   df['DEXINUS'].plot(title='India/US Exchange Rate')
   ```

2. **Combine with GDELT data** for event-driven analysis

3. **Build models:**
   - Time series forecasting
   - Correlation analysis
   - Feature engineering

## Automation

Run monthly to keep data current:

**Windows Task Scheduler:**
- Create task
- Trigger: Monthly on day 15
- Action: Run `python collect_all_data.py`

**Linux/Mac Cron:**
```bash
# Add to crontab
0 2 15 * * cd /path/to/gdelt_india/scripts/data_collection && python collect_all_data.py
```

## Help

- Full documentation: `../../data/gold_standard/README.md`
- India Commerce manual download: `../../data/gold_standard/india_commerce/README.md`
- API issues: Check API provider websites
- Script issues: Check error messages, verify API keys

## Data Usage Tips

1. **Date alignment:** FRED and Census data may have different release schedules
2. **Missing values:** Some series may have gaps - handle appropriately
3. **Units:** Check units for each series (some are indexed, some are absolute values)
4. **Frequency:** Mix of daily, monthly, and quarterly data
5. **Lags:** Census data has ~45-60 day reporting lag

## Resources

- FRED Documentation: https://fred.stlouisfed.org/docs/api/fred/
- Census Trade API: https://www.census.gov/data/developers/data-sets/international-trade.html
- India Trade Portal: https://tradestat.commerce.gov.in/

---

**Time to first data:** ~5 minutes
**Data coverage:** 2010 - present
**Update frequency:** Run monthly for updates

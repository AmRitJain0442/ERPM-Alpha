import pandas as pd
import yfinance as yf
from google.cloud import bigquery
import os
from curl_cffi import requests as curl_requests

# --- FIX SSL/TLS ISSUES for yfinance with curl_cffi ---
# Create a custom session with SSL verification disabled
session = curl_requests.Session(impersonate="chrome", verify=False)

# --- 1. SETUP GOOGLE AUTH ---
# Make sure you have your BigQuery JSON key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
client = bigquery.Client()

# --- 2. FETCH GDELT (INDIA & USA) ---
print("Querying GDELT for India and US data...")

query = """
    SELECT
      PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS Date,
      ActionGeo_CountryCode AS Country,
      AVG(AvgTone) AS Avg_Tone,
      AVG(GoldsteinScale) AS Avg_Stability,
      SUM(NumMentions) AS Total_Mentions,
      COUNTIF(GoldsteinScale < -7.0) / COUNT(*) AS Panic_Index
    FROM `gdelt-bq.gdeltv2.events`
    WHERE ActionGeo_CountryCode IN ('US', 'IN')
      AND NumMentions > 10
      AND SQLDATE > 20190101
    GROUP BY Date, Country
    ORDER BY Date
"""

# Run query
news_raw = client.query(query).to_dataframe()
news_raw.set_index('Date', inplace=True)

# --- 3. PIVOT DATA (The "Differential" Strategy) ---
print("Reshaping News Data...")

# Separate India and US data
in_news = news_raw[news_raw['Country'] == 'IN'].add_prefix('IN_')
us_news = news_raw[news_raw['Country'] == 'US'].add_prefix('US_')

# Join them on Date
# Now you have: IN_Avg_Stability, US_Avg_Stability in one row
news_df = in_news.join(us_news, how='inner')

# Drop the redundant 'Country' text columns
news_df.drop(columns=['IN_Country', 'US_Country'], inplace=True)

# Create the "Differential" Features (Alpha Signals)
# The Market moves on the DIFFERENCE between US and India
news_df['Diff_Stability'] = news_df['US_Avg_Stability'] - news_df['IN_Avg_Stability']
news_df['Diff_Tone'] = news_df['US_Avg_Tone'] - news_df['IN_Avg_Tone']

# --- 4. FETCH MACRO FACTORS (The "Other Things") ---
print("Fetching Critical Macro Factors...")
# These are the 3 things that ACTUALLY move the needle besides news:
# 1. Brent Crude Oil (India imports 85% of oil -> Price UP = Rupee DOWN)
# 2. Gold (India imports massive gold -> Price UP = Rupee DOWN)
# 3. US 10Y Treasury (Yield UP = Money leaves India = Rupee DOWN)

tickers = {
    'INR': 'INR=X',      # The Target
    'OIL': 'BZ=F',       # Brent Crude Oil (More relevant for India than WTI)
    'GOLD': 'GC=F',      # Gold Futures
    'US10Y': '^TNX',     # US 10-Year Bond Yield
    'DXY': 'DX-Y.NYB'    # Dollar Index (Global USD Strength)
}

# Download each ticker individually with retry logic
print("Downloading macro data (this may take a moment)...")
macro_data = {}
max_retries = 3

for name, ticker in tickers.items():
    for attempt in range(max_retries):
        try:
            print(f"  Fetching {name} ({ticker})... (attempt {attempt + 1})")
            data = yf.download(ticker, start='2019-01-01', progress=False, timeout=30, session=session)
            if not data.empty:
                # Handle multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    close_col = data['Close']
                    if isinstance(close_col, pd.DataFrame):
                        close_col = close_col.iloc[:, 0]
                else:
                    close_col = data['Close']
                macro_data[name] = close_col
                print(f"    ✓ {name}: {len(data)} rows")
                break
            else:
                print(f"    ✗ {name}: No data (attempt {attempt + 1})")
        except Exception as e:
            print(f"    ✗ {name}: {str(e)[:60]} (attempt {attempt + 1})")
        
        if attempt == max_retries - 1:
            print(f"    ✗ {name}: Failed after {max_retries} attempts")

if macro_data:
    macro_df = pd.DataFrame(macro_data)
    # Ensure index is proper datetime
    macro_df.index = pd.to_datetime(macro_df.index).normalize()
    macro_df.index.name = 'Date'
    print(f"\nMacro data date range: {macro_df.index.min()} to {macro_df.index.max()}")
else:
    print("ERROR: Could not fetch any macro data!")
    macro_df = pd.DataFrame()

# --- 5. THE FINAL MERGE ---
print("\nCreating Master Dataset...")

# Ensure news_df index is also proper datetime
news_df.index = pd.to_datetime(news_df.index).normalize()
news_df.index.name = 'Date'
print(f"News data date range: {news_df.index.min()} to {news_df.index.max()}")

# Join Macro + News
master_df = macro_df.join(news_df, how='inner')

# Clean up (Fill small gaps in macro data with previous day's value)
master_df.ffill(inplace=True)
master_df.dropna(inplace=True)

print(f"SUCCESS. Generated Master Dataset with {len(master_df)} rows.")
print("Features Available:", master_df.columns.tolist())

# Save
master_df.to_csv('Super_Master_Dataset.csv')
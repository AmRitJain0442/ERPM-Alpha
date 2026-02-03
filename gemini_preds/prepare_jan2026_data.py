"""
Prepare January 2026 data for V4 simulation testing.

This script:
1. Fetches financial data (INR, OIL, GOLD, US10Y, DXY) for Jan 2026
2. Processes jan2026.csv GDELT data for India/US news aggregates
3. Merges everything into Super_Master_Dataset format
4. Creates a dataset ready for V4 simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import yfinance with SSL fix
try:
    from curl_cffi import requests as curl_requests
    session = curl_requests.Session(impersonate="chrome", verify=False)
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False
    session = None

import yfinance as yf


def fetch_financial_data(start_date: str = "2026-01-01", end_date: str = "2026-02-01"):
    """Fetch financial data from Yahoo Finance for the given date range."""
    print("=" * 60)
    print("FETCHING FINANCIAL DATA FOR JANUARY 2026")
    print("=" * 60)
    
    tickers = {
        'INR': 'INR=X',      # USD/INR Exchange Rate
        'OIL': 'BZ=F',       # Brent Crude Oil
        'GOLD': 'GC=F',      # Gold Futures
        'US10Y': '^TNX',     # US 10-Year Bond Yield
        'DXY': 'DX-Y.NYB'    # Dollar Index
    }
    
    macro_data = {}
    max_retries = 3
    
    for name, ticker in tickers.items():
        for attempt in range(max_retries):
            try:
                print(f"  Fetching {name} ({ticker})... (attempt {attempt + 1})")
                if HAS_CURL_CFFI:
                    data = yf.download(ticker, start=start_date, end=end_date, 
                                       progress=False, timeout=30, session=session)
                else:
                    data = yf.download(ticker, start=start_date, end=end_date, 
                                       progress=False, timeout=30)
                
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
        macro_df.index = pd.to_datetime(macro_df.index).normalize()
        macro_df.index.name = 'Date'
        print(f"\nFinancial data: {macro_df.index.min()} to {macro_df.index.max()}")
        print(f"Total trading days: {len(macro_df)}")
        return macro_df
    else:
        print("ERROR: Could not fetch financial data!")
        return None


def process_gdelt_news(gdelt_file: str):
    """
    Process the jan2026.csv GDELT file to extract India/US news aggregates.
    
    Updated columns in jan2026.csv (new format):
    Date, GlobalEventID, Event_Location, NumMentions, NumSources, 
    AvgTone, GoldsteinScale, Actor1Name, Actor2Name, SourceURL, V2Themes
    """
    print("\n" + "=" * 60)
    print("PROCESSING GDELT NEWS DATA")
    print("=" * 60)
    
    print(f"Reading {gdelt_file}...")
    
    # Read the file - it's now smaller so no need for chunks
    df = pd.read_csv(gdelt_file)
    print(f"Total rows loaded: {len(df)}")
    
    # Parse date - format is YYYYMMDD (integer)
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')
    
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique dates: {df['Date'].nunique()}")
    
    # Use Event_Location for country classification (new format)
    # IN = India, US = USA
    df['Country'] = df['Event_Location'].apply(lambda x: 
        'IN' if x == 'IN' else ('US' if x == 'US' else 'OTHER'))
    
    country_counts = df['Country'].value_counts()
    print("\nNews distribution by country:")
    print(country_counts)
    
    # Aggregate by date and country
    print("\nAggregating news metrics by date...")
    
    def calculate_panic_index(goldstein_series):
        """Calculate panic index: fraction of news with GoldsteinScale < -7"""
        if len(goldstein_series) == 0:
            return 0.0
        return (goldstein_series < -7).sum() / len(goldstein_series)
    
    news_agg = df.groupby(['Date', 'Country']).agg({
        'AvgTone': 'mean',
        'GoldsteinScale': ['mean', calculate_panic_index],
        'NumMentions': 'sum'
    }).reset_index()
    
    # Flatten column names
    news_agg.columns = ['Date', 'Country', 'Avg_Tone', 'Avg_Stability', 'Panic_Index', 'Total_Mentions']
    
    # Pivot to get India and US side by side
    in_news = news_agg[news_agg['Country'] == 'IN'].copy()
    us_news = news_agg[news_agg['Country'] == 'US'].copy()
    
    in_news = in_news.rename(columns={
        'Avg_Tone': 'IN_Avg_Tone',
        'Avg_Stability': 'IN_Avg_Stability', 
        'Total_Mentions': 'IN_Total_Mentions',
        'Panic_Index': 'IN_Panic_Index'
    }).drop(columns=['Country'])
    
    us_news = us_news.rename(columns={
        'Avg_Tone': 'US_Avg_Tone',
        'Avg_Stability': 'US_Avg_Stability',
        'Total_Mentions': 'US_Total_Mentions',
        'Panic_Index': 'US_Panic_Index'
    }).drop(columns=['Country'])
    
    # Merge on date
    news_df = pd.merge(in_news, us_news, on='Date', how='outer')
    news_df = news_df.set_index('Date').sort_index()
    
    # Fill any missing with defaults
    news_df = news_df.fillna({
        'IN_Avg_Tone': -2.0,
        'IN_Avg_Stability': 0.0,
        'IN_Total_Mentions': 0,
        'IN_Panic_Index': 0.1,
        'US_Avg_Tone': -2.0,
        'US_Avg_Stability': 0.0,
        'US_Total_Mentions': 0,
        'US_Panic_Index': 0.1
    })
    
    # Add differential features
    news_df['Diff_Stability'] = news_df['US_Avg_Stability'] - news_df['IN_Avg_Stability']
    news_df['Diff_Tone'] = news_df['US_Avg_Tone'] - news_df['IN_Avg_Tone']
    
    print(f"\nNews data aggregated for {len(news_df)} trading days")
    print(f"Date range: {news_df.index.min()} to {news_df.index.max()}")
    
    return news_df, df  # Return both aggregated and raw for news headlines


def create_master_dataset(macro_df: pd.DataFrame, news_df: pd.DataFrame,
                         existing_master: str = "../Super_Master_Dataset.csv"):
    """
    Merge financial and news data, then combine with existing master dataset.
    """
    print("\n" + "=" * 60)
    print("CREATING MASTER DATASET FOR JANUARY 2026")
    print("=" * 60)
    
    # Join financial and news data
    jan2026_df = macro_df.join(news_df, how='inner')
    
    # Forward fill small gaps
    jan2026_df = jan2026_df.ffill().dropna()
    
    print(f"January 2026 data: {len(jan2026_df)} trading days")
    
    # Load existing master dataset
    if os.path.exists(existing_master):
        print(f"\nLoading existing master dataset: {existing_master}")
        master_df = pd.read_csv(existing_master, parse_dates=['Date'])
        master_df = master_df.set_index('Date')
        print(f"Existing data: {len(master_df)} rows, {master_df.index.min()} to {master_df.index.max()}")
        
        # Combine - update existing rows and add new ones
        # Remove any overlapping dates from master
        jan_start = jan2026_df.index.min()
        master_df = master_df[master_df.index < jan_start]
        
        # Concatenate
        combined_df = pd.concat([master_df, jan2026_df])
        combined_df = combined_df.sort_index()
        
        print(f"\nCombined dataset: {len(combined_df)} rows")
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    else:
        combined_df = jan2026_df
    
    # Reset index for saving
    combined_df = combined_df.reset_index()
    
    return combined_df


def extract_news_for_simulation(raw_gdelt: pd.DataFrame, output_india: str, output_usa: str):
    """
    Extract and save India/USA news for the simulation's news digest feature.
    Format compatible with news_digest.py expectations.
    
    The news_digest.py expects columns:
    - SQLDATE (integer format YYYYMMDD) OR Date (parseable)
    - SOURCEURL
    - AvgTone
    - GoldsteinScale
    - NumMentions
    - Actor1Name, Actor2Name (optional)
    - ActionGeo_FullName (optional)
    - V2Themes (bonus: for enhanced theme analysis)
    """
    print("\n" + "=" * 60)
    print("EXTRACTING NEWS FILES FOR SIMULATION")
    print("=" * 60)
    
    # Format date as SQLDATE integer (YYYYMMDD) for compatibility with news_digest.py
    raw_gdelt['SQLDATE'] = raw_gdelt['Date'].dt.strftime('%Y%m%d').astype(int)
    
    # Also keep standard Date format
    raw_gdelt['DateStr'] = raw_gdelt['Date'].dt.strftime('%Y-%m-%d')
    
    # Use NumSources as a proxy for NumArticles if not available
    if 'NumArticles' not in raw_gdelt.columns:
        raw_gdelt['NumArticles'] = raw_gdelt.get('NumSources', 1)
    
    # India news (Event_Location == 'IN')
    india_news = raw_gdelt[raw_gdelt['Country'] == 'IN'].copy()
    india_out = india_news[['SQLDATE', 'DateStr', 'GlobalEventID', 'GoldsteinScale', 'AvgTone', 
                            'NumMentions', 'Actor1Name', 'Actor2Name', 
                            'SourceURL', 'V2Themes']].copy()
    india_out.columns = ['SQLDATE', 'Date', 'GlobalEventID', 'GoldsteinScale', 'AvgTone', 
                         'NumMentions', 'Actor1Name', 'Actor2Name',
                         'SOURCEURL', 'V2Themes']
    india_out = india_out.sort_values('SQLDATE')
    india_out.to_csv(output_india, index=False)
    print(f"Saved India news: {len(india_out)} rows -> {output_india}")
    
    # USA news (Event_Location == 'US')
    usa_news = raw_gdelt[raw_gdelt['Country'] == 'US'].copy()
    usa_out = usa_news[['SQLDATE', 'DateStr', 'GlobalEventID', 'GoldsteinScale', 'AvgTone',
                        'NumMentions', 'Actor1Name', 'Actor2Name',
                        'SourceURL', 'V2Themes']].copy()
    usa_out.columns = ['SQLDATE', 'Date', 'GlobalEventID', 'GoldsteinScale', 'AvgTone',
                       'NumMentions', 'Actor1Name', 'Actor2Name',
                       'SOURCEURL', 'V2Themes']
    usa_out = usa_out.sort_values('SQLDATE')
    usa_out.to_csv(output_usa, index=False)
    print(f"Saved USA news: {len(usa_out)} rows -> {output_usa}")


def main():
    print("\n" + "=" * 70)
    print("JANUARY 2026 DATA PREPARATION FOR V4 SIMULATION")
    print("=" * 70)
    print()
    
    # Paths
    gdelt_file = "../jan2026.csv"
    existing_master = "../Super_Master_Dataset.csv"
    output_master = "../Super_Master_Dataset_Jan2026.csv"
    output_india = "india_news_jan2026.csv"
    output_usa = "usa_news_jan2026.csv"
    
    # Step 1: Fetch financial data
    macro_df = fetch_financial_data(start_date="2025-12-01", end_date="2026-02-01")
    
    if macro_df is None or len(macro_df) == 0:
        print("\nERROR: Could not fetch financial data!")
        print("The simulation may use data up to Jan 15 which is already in the dataset.")
        macro_df = None
    
    # Step 2: Process GDELT news
    news_df, raw_gdelt = process_gdelt_news(gdelt_file)
    
    # Step 3: Create master dataset
    if macro_df is not None:
        master_df = create_master_dataset(macro_df, news_df, existing_master)
        master_df.to_csv(output_master, index=False)
        print(f"\nSaved: {output_master}")
    else:
        print("\nUsing existing master dataset (already has data to Jan 15)")
        output_master = existing_master
    
    # Step 4: Extract news files for simulation
    extract_news_for_simulation(raw_gdelt, output_india, output_usa)
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  1. Master dataset: {output_master}")
    print(f"  2. India news: {output_india}")
    print(f"  3. USA news: {output_usa}")
    print()
    print("Next step: Run the V4 simulation with:")
    print("  python run_simulation_v4_jan2026.py")
    
    return output_master, output_india, output_usa


if __name__ == "__main__":
    main()

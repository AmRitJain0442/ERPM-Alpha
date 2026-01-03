"""
FRED (Federal Reserve Economic Data) Collector
Fetches trade balance, exchange rates, and macroeconomic indicators
API Documentation: https://fred.stlouisfed.org/docs/api/fred/
"""

import requests
import pandas as pd
from datetime import datetime
import os
import time

class FREDCollector:
    def __init__(self, api_key=None):
        """
        Initialize FRED API collector

        Args:
            api_key: FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY', '1e8480395f333cd2395ca758bdc3df1e')
        self.base_url = "https://api.stlouisfed.org/fred"
        # Get workspace root (2 levels up from this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(script_dir))
        self.output_dir = os.path.join(workspace_root, "data", "gold_standard", "fred")
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.api_key:
            raise ValueError(
                "FRED API key required. Get one at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "Set as environment variable: FRED_API_KEY"
            )

    def get_series(self, series_id, start_date=None, end_date=None):
        """
        Fetch a specific FRED series

        Args:
            series_id: FRED series identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with series data
        """
        endpoint = f"{self.base_url}/series/observations"

        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }

        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()

            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['series_id'] = series_id
                return df
            else:
                print(f"No data found for series: {series_id}")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()

    def get_series_info(self, series_id):
        """Get metadata about a FRED series"""
        endpoint = f"{self.base_url}/series"

        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json'
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if 'seriess' in data and len(data['seriess']) > 0:
                return data['seriess'][0]
            else:
                return {}

        except Exception as e:
            print(f"Error fetching info for {series_id}: {e}")
            return {}

    def collect_trade_indicators(self, start_date='2010-01-01', end_date=None):
        """
        Collect key trade and economic indicators

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary of DataFrames for each indicator
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Key FRED series for US-India trade analysis
        series_to_fetch = {
            # Exchange Rates
            'DEXINUS': 'India / U.S. Foreign Exchange Rate',

            # US Trade Balance
            'BOPGSTB': 'Trade Balance: Goods and Services',
            'BOPGTB': 'Trade Balance: Goods',
            'BOPSTB': 'Trade Balance: Services',

            # US Exports/Imports
            'EXPGS': 'Exports of Goods and Services',
            'IMPGS': 'Imports of Goods and Services',

            # Economic Indicators
            'GDP': 'Gross Domestic Product',
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
            'UNRATE': 'Unemployment Rate',

            # Interest Rates
            'DFF': 'Federal Funds Effective Rate',
            'DGS10': '10-Year Treasury Constant Maturity Rate',

            # Currency/Money Supply
            'M2SL': 'M2 Money Stock',
            'DTWEXBGS': 'Trade Weighted U.S. Dollar Index: Broad, Goods and Services',

            # Oil Prices (affects trade)
            'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI)',

            # Manufacturing
            'INDPRO': 'Industrial Production Index',
        }

        print("=== Collecting FRED Data ===")
        all_data = {}

        for series_id, description in series_to_fetch.items():
            print(f"\nFetching: {series_id} - {description}")

            # Get series info
            info = self.get_series_info(series_id)
            if info:
                print(f"  Units: {info.get('units', 'N/A')}")
                print(f"  Frequency: {info.get('frequency', 'N/A')}")

            # Get series data
            df = self.get_series(series_id, start_date, end_date)

            if not df.empty:
                df['description'] = description
                df['units'] = info.get('units', 'N/A')
                df['frequency'] = info.get('frequency', 'N/A')
                all_data[series_id] = df
                print(f"  ✓ Retrieved {len(df)} observations")
            else:
                print(f"  ✗ No data retrieved")

            time.sleep(0.5)  # Rate limiting

        return all_data

    def save_data(self, data_dict, combined=True):
        """
        Save collected data to CSV files

        Args:
            data_dict: Dictionary of DataFrames
            combined: If True, save all series in one file; if False, separate files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if combined:
            # Combine all series into one file
            all_dfs = []
            for series_id, df in data_dict.items():
                all_dfs.append(df)

            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                filepath = os.path.join(self.output_dir, f'fred_combined_{timestamp}.csv')
                combined_df.to_csv(filepath, index=False)
                print(f"\nCombined data saved to: {filepath}")

        else:
            # Save each series separately
            for series_id, df in data_dict.items():
                filename = f'{series_id}_{timestamp}.csv'
                filepath = os.path.join(self.output_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"Saved: {filepath}")

    def create_wide_format(self, data_dict):
        """
        Convert time series data to wide format (date x series)

        Args:
            data_dict: Dictionary of DataFrames

        Returns:
            Wide format DataFrame
        """
        wide_data = {}

        for series_id, df in data_dict.items():
            if not df.empty and 'date' in df.columns and 'value' in df.columns:
                # Convert value to numeric
                df['value'] = pd.to_numeric(df['value'], errors='coerce')

                # Use date as index and value as column
                wide_data[series_id] = df.set_index('date')['value']

        if wide_data:
            wide_df = pd.DataFrame(wide_data)
            wide_df.index = pd.to_datetime(wide_df.index)
            wide_df = wide_df.sort_index()
            return wide_df
        else:
            return pd.DataFrame()

    def save_wide_format(self, data_dict):
        """Save data in wide format for easy analysis"""
        wide_df = self.create_wide_format(data_dict)

        if not wide_df.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f'fred_wide_format_{timestamp}.csv')
            wide_df.to_csv(filepath)
            print(f"\nWide format data saved to: {filepath}")
            print(f"Shape: {wide_df.shape}")
            print(f"Date range: {wide_df.index.min()} to {wide_df.index.max()}")

            return wide_df
        else:
            return pd.DataFrame()


def main():
    """Main execution function"""
    # Initialize collector
    # Get API key from environment variable
    api_key = os.getenv('FRED_API_KEY')

    if not api_key:
        print("=" * 60)
        print("FRED API Key Required")
        print("=" * 60)
        print("\n1. Get a free API key at:")
        print("   https://fred.stlouisfed.org/docs/api/api_key.html")
        print("\n2. Set it as an environment variable:")
        print("   Windows: set FRED_API_KEY=your_key_here")
        print("   Linux/Mac: export FRED_API_KEY=your_key_here")
        print("\n3. Or add to your .env file:")
        print("   FRED_API_KEY=your_key_here")
        print("=" * 60)
        return

    try:
        collector = FREDCollector(api_key=api_key)

        # Collect data from 2010 to present
        print("\nCollecting trade and economic indicators from 2010 to present...")
        data = collector.collect_trade_indicators(start_date='2010-01-01')

        if data:
            # Save in both formats
            print("\n--- Saving Data ---")
            collector.save_data(data, combined=True)
            wide_df = collector.save_wide_format(data)

            # Display summary
            print("\n=== Collection Summary ===")
            print(f"Series collected: {len(data)}")
            for series_id in data.keys():
                print(f"  - {series_id}: {len(data[series_id])} observations")

            if not wide_df.empty:
                print("\n=== Wide Format Preview ===")
                print(wide_df.tail(10))

        else:
            print("\nNo data collected. Check API key and network connection.")

    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()

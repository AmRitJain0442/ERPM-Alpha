"""
US Census Bureau International Trade Data Collector
Fetches detailed monthly export/import data between US and India
API Documentation: https://www.census.gov/data/developers/data-sets/international-trade.html
"""

import requests
import pandas as pd
import json
from datetime import datetime
import os

class USCensusCollector:
    def __init__(self, api_key=None):
        """
        Initialize US Census Bureau API collector

        Args:
            api_key: Census API key (get from https://api.census.gov/data/key_signup.html)
        """
        self.api_key = api_key
        self.base_url = "https://api.census.gov/data/timeseries/intltrade"
        # Get workspace root (2 levels up from this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(script_dir))
        self.output_dir = os.path.join(workspace_root, "data", "gold_standard", "us_census")
        
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_trade_data(self, start_year=2010, end_year=None, trade_type='imports'):
        """
        Fetch US-India trade data

        Args:
            start_year: Starting year for data collection
            end_year: Ending year (defaults to current year)
            trade_type: 'imports' or 'exports'

        Returns:
            DataFrame with trade data
        """
        if end_year is None:
            end_year = datetime.now().year

        endpoint = f"{self.base_url}/{trade_type}/country"

        all_data = []

        for year in range(start_year, end_year + 1):
            print(f"Fetching {trade_type} data for {year}...")

            params = {
                'get': 'CTY_CODE,CTY_NAME,I_COMMODITY,I_COMMODITY_LDESC,ALL_VAL_MO,ALL_VAL_YR',
                'time': year,
                'CTY_CODE': '4100',  # India's country code
            }

            if self.api_key:
                params['key'] = self.api_key

            try:
                response = requests.get(endpoint, params=params)
                response.raise_for_status()

                data = response.json()

                if len(data) > 1:  # First row is headers
                    headers = data[0]
                    rows = data[1:]

                    df = pd.DataFrame(rows, columns=headers)
                    df['year'] = year
                    df['trade_type'] = trade_type
                    all_data.append(df)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {year}: {e}")
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def save_data(self, df, filename=None):
        """Save data to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"us_india_trade_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath

    def collect_all_trade_data(self, start_year=2010, end_year=None):
        """Collect both imports and exports data"""
        print("=== Collecting US Census Trade Data ===")

        # Fetch imports
        print("\n1. Fetching US imports from India...")
        imports_df = self.fetch_trade_data(start_year, end_year, 'imports')
        if not imports_df.empty:
            self.save_data(imports_df, 'us_imports_from_india.csv')

        # Fetch exports
        print("\n2. Fetching US exports to India...")
        exports_df = self.fetch_trade_data(start_year, end_year, 'exports')
        if not exports_df.empty:
            self.save_data(exports_df, 'us_exports_to_india.csv')

        print("\n=== Collection Complete ===")

        return {
            'imports': imports_df,
            'exports': exports_df
        }


def main():
    """Main execution function"""
    # Initialize collector
    # To use API key, set environment variable: CENSUS_API_KEY
    api_key = os.getenv('CENSUS_API_KEY')

    if not api_key:
        print("Warning: No API key found. Request limit will be lower.")
        print("Get a free API key at: https://api.census.gov/data/key_signup.html")
        print("Set it as environment variable: CENSUS_API_KEY")

    collector = USCensusCollector(api_key=api_key)

    # Collect data from 2010 to present
    data = collector.collect_all_trade_data(start_year=2010)

    # Display summary
    print("\n=== Summary ===")
    if not data['imports'].empty:
        print(f"Imports records: {len(data['imports'])}")
    if not data['exports'].empty:
        print(f"Exports records: {len(data['exports'])}")


if __name__ == "__main__":
    main()

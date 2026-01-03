import requests
import pandas as pd
from datetime import datetime, timedelta
import json

def fetch_usd_inr_rates():
    """
    Fetch USD to INR exchange rates for the past 1 year using Frankfurter API
    """
    # Calculate date range (past 1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Format dates for API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching USD/INR exchange rates from {start_date_str} to {end_date_str}...")

    # Frankfurter API endpoint for historical data
    url = f"https://api.frankfurter.app/{start_date_str}..{end_date_str}"

    # Parameters: from USD to INR
    params = {
        'from': 'USD',
        'to': 'INR'
    }

    try:
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Parse the data
        rates_list = []
        for date, rate_info in data['rates'].items():
            rates_list.append({
                'Date': date,
                'USD_to_INR': rate_info['INR']
            })

        # Create DataFrame
        df = pd.DataFrame(rates_list)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Save to CSV
        csv_filename = 'usd_inr_exchange_rates_1year.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\nData saved to {csv_filename}")

        # Save to JSON as well
        json_filename = 'usd_inr_exchange_rates_1year.json'
        df.to_json(json_filename, orient='records', date_format='iso', indent=2)
        print(f"Data also saved to {json_filename}")

        # Display statistics
        print(f"\n--- Exchange Rate Statistics ---")
        print(f"Total data points: {len(df)}")
        print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Highest rate: {df['USD_to_INR'].max():.4f} INR per USD")
        print(f"Lowest rate: {df['USD_to_INR'].min():.4f} INR per USD")
        print(f"Average rate: {df['USD_to_INR'].mean():.4f} INR per USD")
        print(f"Latest rate: {df.iloc[-1]['USD_to_INR']:.4f} INR per USD (on {df.iloc[-1]['Date'].strftime('%Y-%m-%d')})")

        # Display first few rows
        print(f"\n--- Sample Data (first 5 rows) ---")
        print(df.head().to_string(index=False))

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

if __name__ == "__main__":
    df = fetch_usd_inr_rates()
    if df is not None:
        print(f"\n✓ Successfully fetched and saved USD/INR exchange rates for the past year!")

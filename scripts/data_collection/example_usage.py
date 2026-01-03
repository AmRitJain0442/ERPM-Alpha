"""
Example: Using Gold Standard Data for Analysis
Demonstrates how to load and analyze collected trade data
"""

import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os

def load_latest_fred_data():
    """Load the most recent FRED data file"""
    fred_dir = "data/gold_standard/fred"

    # Find the latest wide format file
    wide_files = glob(os.path.join(fred_dir, "fred_wide_format_*.csv"))

    if not wide_files:
        print("No FRED data found. Run fetch_fred.py first.")
        return None

    # Get the most recent file
    latest_file = max(wide_files, key=os.path.getctime)
    print(f"Loading: {latest_file}")

    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    return df


def load_us_census_data():
    """Load US Census trade data"""
    census_dir = "data/gold_standard/us_census"

    imports_file = os.path.join(census_dir, "us_imports_from_india.csv")
    exports_file = os.path.join(census_dir, "us_exports_to_india.csv")

    data = {}

    if os.path.exists(imports_file):
        print(f"Loading: {imports_file}")
        data['imports'] = pd.read_csv(imports_file)

    if os.path.exists(exports_file):
        print(f"Loading: {exports_file}")
        data['exports'] = pd.read_csv(exports_file)

    return data


def analyze_exchange_rate(fred_df):
    """Analyze India/US exchange rate trends"""
    if 'DEXINUS' not in fred_df.columns:
        print("Exchange rate data not available")
        return

    print("\n=== India/US Exchange Rate Analysis ===")

    exchange_rate = fred_df['DEXINUS'].dropna()

    print(f"\nPeriod: {exchange_rate.index.min()} to {exchange_rate.index.max()}")
    print(f"Latest rate: {exchange_rate.iloc[-1]:.4f} INR/USD")
    print(f"Average rate: {exchange_rate.mean():.4f} INR/USD")
    print(f"Min rate: {exchange_rate.min():.4f} INR/USD")
    print(f"Max rate: {exchange_rate.max():.4f} INR/USD")

    # Calculate yearly changes
    yearly = exchange_rate.resample('YE').last()
    yearly_change = yearly.pct_change() * 100

    print("\nYearly % Change:")
    print(yearly_change.tail(5))

    # Plot
    plt.figure(figsize=(12, 6))
    exchange_rate.plot()
    plt.title('India/US Exchange Rate (INR/USD)')
    plt.ylabel('INR per USD')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/gold_standard/exchange_rate_trend.png', dpi=300)
    print("\nPlot saved: data/gold_standard/exchange_rate_trend.png")
    plt.close()


def analyze_trade_balance(fred_df):
    """Analyze US trade balance"""
    if 'BOPGSTB' not in fred_df.columns:
        print("Trade balance data not available")
        return

    print("\n=== US Trade Balance Analysis ===")

    trade_balance = fred_df['BOPGSTB'].dropna()

    print(f"\nLatest trade balance: ${trade_balance.iloc[-1]:,.0f} million")
    print(f"Average: ${trade_balance.mean():,.0f} million")

    # Trend analysis
    recent = trade_balance.last('5Y')
    print(f"\n5-Year Average: ${recent.mean():,.0f} million")
    print(f"Trend: {'Improving' if recent.iloc[-1] > recent.mean() else 'Deteriorating'}")

    # Plot
    plt.figure(figsize=(12, 6))
    trade_balance.plot()
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('US Trade Balance: Goods and Services')
    plt.ylabel('Balance (Million USD)')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/gold_standard/trade_balance_trend.png', dpi=300)
    print("\nPlot saved: data/gold_standard/trade_balance_trend.png")
    plt.close()


def analyze_top_imports(census_data):
    """Analyze top import categories from India"""
    if 'imports' not in census_data:
        print("US Census import data not available")
        return

    print("\n=== Top US Imports from India ===")

    imports_df = census_data['imports']

    # Group by commodity and sum values
    if 'I_COMMODITY_LDESC' in imports_df.columns and 'ALL_VAL_YR' in imports_df.columns:
        # Convert to numeric
        imports_df['ALL_VAL_YR'] = pd.to_numeric(imports_df['ALL_VAL_YR'], errors='coerce')

        # Get top categories
        top_imports = imports_df.groupby('I_COMMODITY_LDESC')['ALL_VAL_YR'].sum().sort_values(ascending=False)

        print("\nTop 10 Import Categories by Value:")
        for i, (category, value) in enumerate(top_imports.head(10).items(), 1):
            print(f"{i:2d}. {category[:60]:<60} ${value:>12,.0f}")

        # Plot
        plt.figure(figsize=(12, 8))
        top_imports.head(15).plot(kind='barh')
        plt.title('Top 15 US Imports from India by Value')
        plt.xlabel('Total Value (USD)')
        plt.tight_layout()
        plt.savefig('data/gold_standard/top_imports.png', dpi=300)
        print("\nPlot saved: data/gold_standard/top_imports.png")
        plt.close()


def analyze_correlation(fred_df):
    """Analyze correlation between exchange rate and trade"""
    if 'DEXINUS' not in fred_df.columns or 'BOPGSTB' not in fred_df.columns:
        print("Insufficient data for correlation analysis")
        return

    print("\n=== Exchange Rate vs Trade Balance Correlation ===")

    # Prepare data
    analysis_df = fred_df[['DEXINUS', 'BOPGSTB']].dropna()

    if len(analysis_df) < 10:
        print("Insufficient overlapping data points")
        return

    # Calculate correlation
    correlation = analysis_df.corr().iloc[0, 1]
    print(f"\nCorrelation: {correlation:.4f}")
    print(f"Interpretation: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} {'positive' if correlation > 0 else 'negative'} relationship")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Time series
    ax1_twin = ax1.twinx()
    analysis_df['DEXINUS'].plot(ax=ax1, color='blue', label='Exchange Rate')
    analysis_df['BOPGSTB'].plot(ax=ax1_twin, color='red', label='Trade Balance')

    ax1.set_ylabel('Exchange Rate (INR/USD)', color='blue')
    ax1.set_xlabel('Date')
    ax1_twin.set_ylabel('Trade Balance (Million USD)', color='red')
    ax1.set_title('Exchange Rate vs Trade Balance Over Time')
    ax1.grid(True, alpha=0.3)

    # Scatter plot
    ax2.scatter(analysis_df['DEXINUS'], analysis_df['BOPGSTB'], alpha=0.5)
    ax2.set_xlabel('Exchange Rate (INR/USD)')
    ax2.set_ylabel('Trade Balance (Million USD)')
    ax2.set_title(f'Correlation: {correlation:.4f}')
    ax2.grid(True, alpha=0.3)

    # Add trend line
    z = pd.Series.to_numpy(analysis_df['DEXINUS'])
    p = pd.Series.to_numpy(analysis_df['BOPGSTB'])
    coeffs = pd.Series.to_numpy(pd.Series([z, p]).T.corr().iloc[0])

    plt.tight_layout()
    plt.savefig('data/gold_standard/exchange_trade_correlation.png', dpi=300)
    print("\nPlot saved: data/gold_standard/exchange_trade_correlation.png")
    plt.close()


def main():
    """Main execution"""
    print("=" * 70)
    print("  Gold Standard Data Analysis Example")
    print("=" * 70)

    # Load FRED data
    print("\n--- Loading FRED Data ---")
    fred_df = load_latest_fred_data()

    # Load US Census data
    print("\n--- Loading US Census Data ---")
    census_data = load_us_census_data()

    # Perform analyses
    if fred_df is not None:
        analyze_exchange_rate(fred_df)
        analyze_trade_balance(fred_df)
        analyze_correlation(fred_df)

    if census_data:
        analyze_top_imports(census_data)

    print("\n" + "=" * 70)
    print("  Analysis Complete")
    print("=" * 70)
    print("\nGenerated visualizations in: data/gold_standard/")


if __name__ == "__main__":
    main()

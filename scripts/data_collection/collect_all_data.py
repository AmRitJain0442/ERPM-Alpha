"""
Master Data Collection Script
Runs all "Gold Standard" data collectors for US-India trade analysis
"""

import os
import sys
from datetime import datetime

# Import individual collectors
from fetch_us_census import USCensusCollector
from fetch_india_commerce import IndiaCommerceCollector
from fetch_fred import FREDCollector


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_api_keys():
    """Check if required API keys are present"""
    keys_status = {
        'FRED_API_KEY': os.getenv('FRED_API_KEY', '1e8480395f333cd2395ca758bdc3df1e'),
        'CENSUS_API_KEY': os.getenv('CENSUS_API_KEY')
    }

    print_banner("API Keys Status")

    all_present = True
    for key_name, key_value in keys_status.items():
        status = "[OK] Present" if key_value else "[X] Missing"
        print(f"{key_name}: {status}")

        if not key_value:
            all_present = False

    if not all_present:
        print("\n[!] Warning: Some API keys are missing.")
        print("  Collection may be limited or fail for certain sources.\n")

        print("How to set API keys:")
        print("  1. FRED API Key:")
        print("     https://fred.stlouisfed.org/docs/api/api_key.html")
        print("\n  2. Census API Key:")
        print("     https://api.census.gov/data/key_signup.html")
        print("\n  3. Set as environment variables:")
        print("     Windows: set FRED_API_KEY=your_key")
        print("     Linux/Mac: export FRED_API_KEY=your_key")

    return keys_status


def collect_fred_data():
    """Collect FRED data"""
    print_banner("FRED Data Collection")

    api_key = os.getenv('FRED_API_KEY', '1e8480395f333cd2395ca758bdc3df1e')
    if not api_key:
        print("[X] Skipping FRED: API key not found")
        return False

    try:
        collector = FREDCollector(api_key=api_key)
        data = collector.collect_trade_indicators(start_date='2010-01-01')

        if data:
            collector.save_data(data, combined=True)
            collector.save_wide_format(data)
            print(f"[OK] FRED data collected: {len(data)} series")
            return True
        else:
            print("[X] No FRED data collected")
            return False

    except Exception as e:
        print(f"[X] Error collecting FRED data: {e}")
        return False


def collect_us_census_data():
    """Collect US Census Bureau data"""
    print_banner("US Census Bureau Data Collection")

    api_key = os.getenv('CENSUS_API_KEY')
    if not api_key:
        print("[!] Warning: No Census API key found. Rate limits will be lower.")

    try:
        collector = USCensusCollector(api_key=api_key)
        data = collector.collect_all_trade_data(start_year=2010)

        success = False
        if data['imports'] is not None and not data['imports'].empty:
            print(f"[OK] US Census imports collected: {len(data['imports'])} records")
            success = True

        if data['exports'] is not None and not data['exports'].empty:
            print(f"[OK] US Census exports collected: {len(data['exports'])} records")
            success = True

        return success

    except Exception as e:
        print(f"[X] Error collecting US Census data: {e}")
        return False


def collect_india_commerce_data():
    """Collect India Ministry of Commerce data"""
    print_banner("India Ministry of Commerce Data Collection")

    try:
        collector = IndiaCommerceCollector()

        # Create README with instructions
        collector.create_readme()

        # Try to load existing manual downloads
        combined_data = collector.load_manual_downloads()

        if not combined_data.empty:
            collector.save_data(combined_data, 'india_usa_trade_combined.csv')
            print(f"[OK] India Commerce data loaded: {len(combined_data)} records")
            return True
        else:
            print("[!] No India Commerce data found (manual download required)")
            print(f"  See instructions in: data/gold_standard/india_commerce/README.md")
            return False

    except Exception as e:
        print(f"[X] Error processing India Commerce data: {e}")
        return False


def main():
    """Main execution function"""
    print_banner("Gold Standard Data Collection")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check API keys
    keys_status = check_api_keys()

    # Collect data from all sources
    results = {
        'FRED': False,
        'US Census': False,
        'India Commerce': False
    }

    # 1. FRED (fastest and most reliable)
    if keys_status['FRED_API_KEY']:
        results['FRED'] = collect_fred_data()
    else:
        print_banner("FRED Data Collection")
        print("[X] Skipped: API key required")

    # 2. US Census Bureau
    results['US Census'] = collect_us_census_data()

    # 3. India Ministry of Commerce
    results['India Commerce'] = collect_india_commerce_data()

    # Summary
    print_banner("Collection Summary")

    for source, success in results.items():
        status = "[OK] Success" if success else "[X] Failed/Incomplete"
        print(f"{source}: {status}")

    successful = sum(results.values())
    print(f"\nTotal: {successful}/3 sources collected successfully")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Data location
    print_banner("Data Location")
    print("Collected data saved to:")
    print("  - FRED: data/gold_standard/fred/")
    print("  - US Census: data/gold_standard/us_census/")
    print("  - India Commerce: data/gold_standard/india_commerce/")

    return successful == 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

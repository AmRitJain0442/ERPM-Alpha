"""
Runner for V4 Simulation on January 2026 Data - Final Test

This script:
1. Prepares the Jan 2026 dataset (or uses existing)
2. Configures the V4 simulation for Jan 1-31, 2026
3. Runs the model with V2 themes support from GDELT data
"""

import os
import sys

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Configuration for January 2026
JAN2026_CONFIG = {
    'SIMULATION_START': '2026-01-01',
    'SIMULATION_END': '2026-01-31',
    'DATA_FILE': '../Super_Master_Dataset_Jan2026.csv',  # Updated dataset with Jan 2026 data
    'INDIA_NEWS_FILE': 'india_news_jan2026.csv',
    'USA_NEWS_FILE': 'usa_news_jan2026.csv',
    'RESULTS_FILE': 'simulation_results_v4_jan2026.json',
    'SUMMARY_CSV': 'simulation_summary_v4_jan2026.csv',
}


def check_and_prepare_data():
    """Check if data is ready, prepare if needed."""
    print("=" * 60)
    print("CHECKING DATA AVAILABILITY FOR JANUARY 2026")
    print("=" * 60)
    
    # Check if news files exist
    india_news = os.path.join(SCRIPT_DIR, JAN2026_CONFIG['INDIA_NEWS_FILE'])
    usa_news = os.path.join(SCRIPT_DIR, JAN2026_CONFIG['USA_NEWS_FILE'])
    
    if not os.path.exists(india_news) or not os.path.exists(usa_news):
        print("\nNews files not found. Preparing data from jan2026.csv...")
        print()
        
        # Run the data preparation script
        from prepare_jan2026_data import main as prepare_data
        prepare_data()
    else:
        print(f"✓ India news: {india_news}")
        print(f"✓ USA news: {usa_news}")
    
    # Check master dataset
    master_data = os.path.join(SCRIPT_DIR, JAN2026_CONFIG['DATA_FILE'])
    if os.path.exists(master_data):
        import pandas as pd
        df = pd.read_csv(master_data, parse_dates=['Date'])
        latest_date = df['Date'].max()
        print(f"✓ Master dataset available: data through {latest_date.date()}")
        
        # Check if Jan 2026 data is available
        jan_data = df[df['Date'] >= '2026-01-01']
        print(f"  → January 2026 trading days in dataset: {len(jan_data)}")
        
        if len(jan_data) < 5:
            print("\n⚠ Warning: Limited January 2026 data in master dataset.")
            print("  The simulation will run with available data.")
    else:
        print(f"✗ Master dataset not found at {master_data}")
        print("  Please ensure Super_Master_Dataset.csv is available")
        return False
    
    return True


def run_jan2026_simulation():
    """Run the V4 simulation configured for January 2026."""
    
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("=" * 60)
        print("GEMINI API KEY NOT FOUND")
        print("=" * 60)
        print()
        print("Set your API key:")
        print("  PowerShell:  $env:GEMINI_API_KEY=\"your_key\"")
        print("  CMD:         set GEMINI_API_KEY=your_key")
        print()

        user_key = input("Or enter key now: ").strip()
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
            api_key = user_key
        else:
            sys.exit(1)

    # Check and prepare data
    if not check_and_prepare_data():
        print("\nData preparation failed. Cannot run simulation.")
        sys.exit(1)

    # Import and patch the simulation module
    print("\n" + "=" * 70)
    print("V4 SIMULATION - JANUARY 2026 FINAL TEST")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Period: {JAN2026_CONFIG['SIMULATION_START']} to {JAN2026_CONFIG['SIMULATION_END']}")
    print(f"  News: Using V2 Themes from GDELT data")
    print(f"  Results: {JAN2026_CONFIG['RESULTS_FILE']}")
    print()

    # Import the V4 module and override configuration
    import gemini_market_simulation_v4 as v4
    from datetime import datetime
    
    # Override the simulation period
    v4.SIMULATION_START = datetime(2026, 1, 1)
    v4.SIMULATION_END = datetime(2026, 1, 31)
    
    # Override file paths - IMPORTANT: use the updated dataset
    v4.DATA_FILE = JAN2026_CONFIG['DATA_FILE']
    v4.RESULTS_FILE = JAN2026_CONFIG['RESULTS_FILE']
    v4.SUMMARY_CSV = JAN2026_CONFIG['SUMMARY_CSV']
    v4.INDIA_NEWS_FILE = JAN2026_CONFIG['INDIA_NEWS_FILE']
    v4.USA_NEWS_FILE = JAN2026_CONFIG['USA_NEWS_FILE']
    
    # Reduce warmup since we have historical data from 2019-2025
    # The model is already well-trained, we just need a few days to calibrate
    v4.WARMUP_DAYS = 3  # Reduced from 20 to 3 days
    
    # Use maximum news digest
    v4.MAX_HEADLINES_PER_DAY = 50  # More headlines for richer context
    v4.USE_NEWS_DIGEST = True
    
    print("Key improvements in V4:")
    print()
    print("  DEBIASING:")
    print("    - Symmetric prompt design (bull/bear cases presented equally)")
    print("    - Personas see their OWN historical accuracy")
    print("    - Direction labels randomized to prevent anchoring")
    print("    - Explicit uncertainty quantification in prompts")
    print()
    print("  NEWS INTEGRATION (V2 THEMES):")
    print("    - Headlines extracted from GDELT SOURCEURL fields")
    print("    - V2 Themes available for enhanced context")
    print("    - Daily news digest generated via LLM")
    print("    - Same news passed to ALL personas for interpretation")
    print()
    print("  STATISTICAL ROBUSTNESS:")
    print("    - Ridge regression with cross-validated regularization")
    print("    - Realized volatility as dynamic feature")
    print("    - Bootstrap confidence intervals")
    print("    - Rolling window recalibration")
    print()
    print("  ENSEMBLE AGGREGATION:")
    print("    - Trimmed weighted mean (remove top/bottom 10%)")
    print("    - Entropy-based information weighting")
    print("    - Bayesian model averaging with proper priors")
    print("    - Random walk benchmark comparison")
    print()
    
    input("Press Enter to start the simulation...")
    print()

    # Run simulation
    results = v4.run_simulation(api_key)
    
    print("\n" + "=" * 70)
    print("JANUARY 2026 SIMULATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Results saved to: {JAN2026_CONFIG['RESULTS_FILE']}")
    print(f"Summary saved to: {JAN2026_CONFIG['SUMMARY_CSV']}")
    
    return results


if __name__ == "__main__":
    try:
        results = run_jan2026_simulation()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

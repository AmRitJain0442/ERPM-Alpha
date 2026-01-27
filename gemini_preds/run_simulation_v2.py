"""
Runner script for Gemini Market Simulation V2 (Hybrid Statistical + LLM)

This version uses a fundamentally different approach:
1. Statistical model (OLS regression) provides calibrated baseline predictions
2. LLM personas provide qualitative sentiment-based adjustments
3. Ensemble combines both for final prediction

Key improvements:
- No more asking LLM for exact prices (which it's not good at)
- Statistical model captures quantitative relationships (US10Y, GOLD, DXY -> INR)
- LLM focuses on what it's good at: interpreting sentiment and news
- Bounded adjustments prevent wild predictions
"""

import os
import sys

def main():
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("=" * 60)
        print("GEMINI API KEY NOT FOUND")
        print("=" * 60)
        print()
        print("Please set your Gemini API key:")
        print()
        print("  Windows (CMD):    set GEMINI_API_KEY=your_key_here")
        print("  Windows (PS):     $env:GEMINI_API_KEY=\"your_key_here\"")
        print("  Linux/Mac:        export GEMINI_API_KEY=your_key_here")
        print()
        print("Get your API key from: https://aistudio.google.com/apikey")
        print()

        user_key = input("Or enter your API key now (press Enter to skip): ").strip()
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
        else:
            sys.exit(1)

    # Import and run simulation
    from gemini_market_simulation_v2 import run_simulation

    print("\n" + "=" * 60)
    print("GEMINI FOREX SIMULATION V2 - HYBRID APPROACH")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  1. Statistical Model: OLS regression on US10Y, GOLD, DXY, etc.")
    print("     - Captures quantitative correlations (US10Y r=0.84, GOLD r=0.80)")
    print("     - Provides calibrated baseline prediction")
    print()
    print("  2. LLM Personas: Interpret sentiment and news")
    print("     - Provide directional adjustment (not exact prices)")
    print("     - Bounded adjustments (max +/-0.5%)")
    print()
    print("  3. Ensemble: Weighted combination of both")
    print()
    print("This approach leverages each system's strengths:")
    print("  - Statistical model: Good at numerical relationships")
    print("  - LLM: Good at qualitative interpretation")
    print()
    print("Press Ctrl+C to stop at any time (progress is saved).")
    print()

    results = run_simulation(os.environ.get("GEMINI_API_KEY"))

    print("\nDone! Check simulation_results_v2.json and simulation_summary_v2.csv")

if __name__ == "__main__":
    main()

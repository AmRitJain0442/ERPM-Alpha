"""
Simple runner script for the Gemini Market Simulation.
Sets up environment and runs the simulation.
"""

import os
import sys

def main():
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("=" * 50)
        print("GEMINI API KEY NOT FOUND")
        print("=" * 50)
        print()
        print("Please set your Gemini API key:")
        print()
        print("  Windows (CMD):    set GEMINI_API_KEY=your_key_here")
        print("  Windows (PS):     $env:GEMINI_API_KEY=\"your_key_here\"")
        print("  Linux/Mac:        export GEMINI_API_KEY=your_key_here")
        print()
        print("Get your API key from: https://aistudio.google.com/apikey")
        print()

        # Optionally prompt for key
        user_key = input("Or enter your API key now (press Enter to skip): ").strip()
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
        else:
            sys.exit(1)

    # Import and run simulation
    from gemini_market_simulation import run_simulation

    print("\nStarting simulation...")
    print("This will take some time as we query Gemini for each persona on each trading day.")
    print("Press Ctrl+C to stop at any time (progress is saved).\n")

    results = run_simulation(os.environ.get("GEMINI_API_KEY"))

    print("\nDone! Check simulation_results.json and simulation_summary.csv for results.")

if __name__ == "__main__":
    main()

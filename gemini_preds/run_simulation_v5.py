"""
V5 Simulation Runner - Pure LLM Predictions
"""

import os
import sys

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)

from gemini_market_simulation_v5 import run_simulation

if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set it with: $env:GEMINI_API_KEY = 'your-api-key'")
        sys.exit(1)
    
    print("Starting V5 Simulation (Pure LLM)...")
    print()
    
    try:
        results = run_simulation(api_key)
        print(f"\nSimulation completed with {len(results)} days")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nSimulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

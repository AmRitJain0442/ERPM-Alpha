"""
Runner for V3 Simulation - Adversarial Personas + Dynamic Weights

Key fixes from V2:
1. Adversarial persona design (50/50 bull/bear split)
2. Statistical baseline shown to personas
3. Contrarian instructions to prevent herd behavior
4. Dynamic weights based on consensus strength
5. Tighter LLM adjustment bounds (max ±0.35%)
6. Regime-specific persona weight adjustments
"""

import os
import sys

def main():
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
        else:
            sys.exit(1)

    from gemini_market_simulation_v3 import run_simulation

    print("\n" + "=" * 60)
    print("V3 SIMULATION - ADVERSARIAL + DYNAMIC WEIGHTS")
    print("=" * 60)
    print()
    print("Fixes applied:")
    print("  1. Adversarial personas: 43% bearish, 42% bullish, 11% neutral")
    print("  2. Statistical baseline in prompts (personas see the model prediction)")
    print("  3. Contrarian instructions (challenge consensus)")
    print("  4. Dynamic weights (65-90% stat, based on consensus strength)")
    print("  5. Tighter bounds (max ±0.35% LLM adjustment)")
    print("  6. Regime-specific weights")
    print()
    print("Expected improvements:")
    print("  - Bullish/Bearish ratio closer to 50/50")
    print("  - LLM helps in >50% of cases")
    print("  - MAE lower than V2")
    print()

    results = run_simulation(os.environ.get("GEMINI_API_KEY"))
    print("\nDone! Results in simulation_results_v3.json")

if __name__ == "__main__":
    main()

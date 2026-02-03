"""
Runner for V4 Simulation - Robust Ensemble with Unbiased Design

Key improvements from V3:
1. Calibrated confidence intervals using historical error distribution
2. Bayesian-style updating with proper uncertainty propagation
3. Market microstructure features (realized volatility, jump detection)
4. Robust aggregation using trimmed means (outlier resistant)
5. Proper cross-validation for stat model parameters
6. Information-theoretic weighting (personas contribute based on entropy)
7. Debiased prompts that show BOTH bull and bear cases equally
8. Explicit calibration against naive forecasts (random walk benchmark)
9. Rolling evaluation window to detect regime shifts
10. Proper handling of look-ahead bias in feature engineering
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

    from gemini_market_simulation_v4 import run_simulation

    print("\n" + "=" * 70)
    print("V4 SIMULATION - ROBUST ENSEMBLE WITH DEBIASED DESIGN + NEWS DIGEST")
    print("=" * 70)
    print()
    print("Key improvements over V3:")
    print()
    print("  DEBIASING:")
    print("    - Symmetric prompt design (bull/bear cases presented equally)")
    print("    - Personas see their OWN historical accuracy")
    print("    - Direction labels randomized to prevent anchoring")
    print("    - Explicit uncertainty quantification in prompts")
    print()
    print("  NEWS INTEGRATION (NEW):")
    print("    - Headlines extracted from GDELT SOURCEURL fields")
    print("    - Daily news digest generated via LLM")
    print("    - Same news passed to ALL personas for interpretation")
    print("    - Each persona interprets news from their perspective")
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
    print("Expected outcomes:")
    print("  - Bull/Bear ratio: ~50/50 (vs 66/34 in V3)")
    print("  - LLM helps: >50% of cases")
    print("  - Lower MAE than stat-only baseline")
    print("  - News context improves direction accuracy")
    print()

    results = run_simulation(os.environ.get("GEMINI_API_KEY"))
    print("\nDone! Results in simulation_results_v4.json")


if __name__ == "__main__":
    main()

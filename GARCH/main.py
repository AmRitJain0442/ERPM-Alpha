"""
EGARCH + XGBoost Hybrid Volatility Model - Main Script

This script orchestrates the complete pipeline for INR/USD exchange rate
prediction using GDELT news data and EGARCH volatility modeling.

The Hybrid Approach:
1. EGARCH captures baseline volatility physics (clustering, asymmetry)
2. XGBoost corrects using news features (GDELT sentiment, Goldstein scores)

Usage:
    python main.py                    # Run with default settings
    python main.py --target price     # Predict price
    python main.py --target volatility # Predict volatility
    python main.py --compare          # Compare GARCH variants
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from config import DATA_PATHS, OUTPUT_DIR, GDELT_FEATURES
from egarch_model import EGARCHVolatilityModel, compare_garch_models
from hybrid_model import HybridVolatilityModel, HybridPipeline
from visualizations import (
    plot_volatility_series,
    plot_news_impact_curve,
    plot_predictions,
    plot_feature_importance,
    plot_asymmetry_analysis,
    plot_model_diagnostics,
    create_summary_dashboard,
    ensure_output_dir
)


def load_data() -> pd.DataFrame:
    """
    Load and merge all data sources.

    Returns:
        DataFrame with exchange rates and GDELT features
    """
    print("Loading data...")

    # Load exchange rates with Goldstein scores
    exchange_path = DATA_PATHS["exchange_rates"]
    if os.path.exists(exchange_path):
        df = pd.read_csv(exchange_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        print(f"  Loaded exchange rates: {len(df)} rows")
    else:
        raise FileNotFoundError(f"Exchange rate data not found: {exchange_path}")

    # Load GDELT training features if available
    training_path = DATA_PATHS["merged_training"]
    if os.path.exists(training_path):
        gdelt_df = pd.read_csv(training_path, parse_dates=["Date"])
        gdelt_df = gdelt_df.set_index("Date").sort_index()
        print(f"  Loaded GDELT features: {len(gdelt_df)} rows")

        # Merge with exchange data
        common_cols = [c for c in gdelt_df.columns if c not in df.columns]
        df = df.join(gdelt_df[common_cols], how="left")

    # Load FRED macro data if available
    fred_path = DATA_PATHS["fred_data"]
    if os.path.exists(fred_path):
        fred_df = pd.read_csv(fred_path, parse_dates=["date"])
        fred_df = fred_df.rename(columns={"date": "Date"}).set_index("Date")
        fred_df = fred_df.sort_index()

        # Select relevant columns
        fred_cols = ["DGS10", "DFF", "DTWEXBGS", "DCOILWTICO"]
        available_fred = [c for c in fred_cols if c in fred_df.columns]

        if available_fred:
            # Forward fill macro data (they update less frequently)
            fred_df = fred_df[available_fred].ffill()
            df = df.join(fred_df, how="left")
            print(f"  Added FRED macro data: {available_fred}")

    # Fill NaN in features
    df = df.ffill().dropna(subset=["USD_to_INR"])

    print(f"  Final dataset: {len(df)} rows, {len(df.columns)} columns")

    return df


def run_egarch_analysis(df: pd.DataFrame) -> dict:
    """
    Run standalone EGARCH analysis.

    Args:
        df: DataFrame with USD_to_INR column

    Returns:
        Dictionary with EGARCH results
    """
    print("\n" + "=" * 70)
    print("  EGARCH VOLATILITY ANALYSIS")
    print("=" * 70)

    # Prepare returns
    returns = df["USD_to_INR"].pct_change().dropna() * 100

    # Fit EGARCH
    print("\nFitting EGARCH(1,1,1) with Skewed-T distribution...")
    model = EGARCHVolatilityModel(p=1, q=1, o=1, dist="skewt", model_type="EGARCH")
    diagnostics = model.fit(returns)

    # Print results
    print("\n[Model Summary]")
    print(f"  Log-Likelihood: {diagnostics['log_likelihood']:.2f}")
    print(f"  AIC: {diagnostics['aic']:.2f}")
    print(f"  BIC: {diagnostics['bic']:.2f}")

    print("\n[Asymmetry Analysis]")
    print(f"  Leverage Effect: {diagnostics['leverage_effect']}")
    print(f"  Gamma (Asymmetry Param): {diagnostics['gamma']:.4f}")
    if diagnostics['gamma'] < 0:
        print("  Interpretation: Negative shocks (bad news) increase volatility MORE")
        print("                  than positive shocks (good news) of same magnitude.")

    print("\n[Persistence]")
    print(f"  Persistence: {diagnostics['persistence']:.4f}")
    print(f"  Half-Life: {diagnostics['half_life']:.1f} days")
    print("  Interpretation: Volatility shocks decay by 50% in ~{:.0f} trading days".format(
        diagnostics['half_life']))

    # Get conditional volatility
    cond_vol = model.get_conditional_volatility()

    # Get news impact curve
    nic = model.get_news_impact_curve()

    # Visualizations
    ensure_output_dir()

    # Plot volatility
    plot_volatility_series(
        cond_vol,
        returns=returns,
        title="EGARCH Conditional Volatility - INR/USD",
        save_path=os.path.join(OUTPUT_DIR, "egarch_volatility.png")
    )

    # Plot news impact curve
    plot_news_impact_curve(
        nic,
        title="News Impact Curve - Asymmetric Response",
        save_path=os.path.join(OUTPUT_DIR, "news_impact_curve.png")
    )

    # Residual diagnostics
    std_resid = model.get_standardized_residuals()
    plot_model_diagnostics(
        std_resid,
        title="EGARCH Residual Diagnostics",
        save_path=os.path.join(OUTPUT_DIR, "egarch_diagnostics.png")
    )

    return {
        "model": model,
        "diagnostics": diagnostics,
        "conditional_volatility": cond_vol,
        "news_impact_curve": nic,
        "standardized_residuals": std_resid,
    }


def run_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare different GARCH variants.

    Args:
        df: DataFrame with USD_to_INR column

    Returns:
        DataFrame with comparison results
    """
    print("\n" + "=" * 70)
    print("  GARCH MODEL COMPARISON")
    print("=" * 70)

    returns = df["USD_to_INR"].pct_change().dropna() * 100

    comparison = compare_garch_models(
        returns,
        models=["GARCH", "EGARCH", "GJR-GARCH"]
    )

    print("\nModel Comparison (sorted by AIC):")
    print(comparison.to_string(index=False))

    best_model = comparison.iloc[0]["Model"]
    print(f"\nBest model by AIC: {best_model}")

    if comparison.iloc[0]["Asymmetry"] != 0:
        print("Asymmetry detected - leverage effect present in INR/USD returns")

    # Save comparison
    comparison.to_csv(os.path.join(OUTPUT_DIR, "garch_comparison.csv"), index=False)

    return comparison


def run_hybrid_pipeline(df: pd.DataFrame, target_type: str = "price") -> dict:
    """
    Run the full EGARCH + XGBoost hybrid pipeline.

    Args:
        df: DataFrame with all features
        target_type: 'price', 'returns', or 'volatility'

    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 70)
    print("  HYBRID EGARCH + XGBOOST PIPELINE")
    print("=" * 70)
    print(f"  Target: {target_type}")

    # Identify available features
    available_features = []

    # GDELT features
    for f in GDELT_FEATURES:
        if f in df.columns:
            available_features.append(f)

    # Goldstein features from exchange data
    goldstein_cols = [c for c in df.columns if "Goldstein" in c or "goldstein" in c.lower()]
    available_features.extend(goldstein_cols)

    # Sentiment features
    sentiment_cols = [c for c in df.columns if "Sentiment" in c or "Tone" in c]
    available_features.extend([c for c in sentiment_cols if c not in available_features])

    # Macro features
    macro_cols = ["DGS10", "DFF", "DTWEXBGS", "DCOILWTICO"]
    available_features.extend([c for c in macro_cols if c in df.columns])

    # Remove duplicates and non-existent
    available_features = list(set([f for f in available_features if f in df.columns]))

    if not available_features:
        print("Warning: No GDELT features found. Using basic features.")
        available_features = [c for c in df.columns if c not in ["USD_to_INR", "Exchange_Rate_Change"]]

    print(f"\nUsing {len(available_features)} features")

    # Initialize and run pipeline
    pipeline = HybridPipeline(target_type=target_type)

    # Prepare data with price column
    df_model = df.copy()
    df_model["Price"] = df_model["USD_to_INR"]

    results = pipeline.run_full_pipeline(
        df=df_model,
        feature_cols=available_features,
        price_col="Price"
    )

    # Generate visualizations
    ensure_output_dir()

    # Predictions plot
    plot_predictions(
        actual=results["test_actual"],
        predicted=results["test_predictions"],
        title=f"Hybrid Model: {target_type.title()} Predictions",
        save_path=os.path.join(OUTPUT_DIR, f"hybrid_predictions_{target_type}.png")
    )

    # Feature importance
    plot_feature_importance(
        results["feature_importance"],
        top_n=15,
        title="Feature Importance (EGARCH + XGBoost)",
        save_path=os.path.join(OUTPUT_DIR, "feature_importance.png")
    )

    # Asymmetry analysis (if available)
    prepared_data = results["prepared_data"]
    if "Is_Panic" in prepared_data.columns and "GARCH_Vol" in prepared_data.columns:
        plot_asymmetry_analysis(
            prepared_data,
            save_path=os.path.join(OUTPUT_DIR, "asymmetry_analysis.png")
        )

    # Summary dashboard
    create_summary_dashboard(
        results,
        save_path=os.path.join(OUTPUT_DIR, "hybrid_dashboard.png")
    )

    # Save predictions
    pred_df = pd.DataFrame({
        "Date": results["test_actual"].index,
        "Actual": results["test_actual"].values,
        "Predicted": results["test_predictions"].values,
        "Error": results["test_actual"].values - results["test_predictions"].values
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f"predictions_{target_type}.csv"), index=False)

    # Save feature importance
    results["feature_importance"].to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False
    )

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EGARCH + XGBoost Hybrid Volatility Model"
    )
    parser.add_argument(
        "--target",
        choices=["price", "returns", "volatility"],
        default="price",
        help="Target variable to predict (default: price)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different GARCH variants"
    )
    parser.add_argument(
        "--egarch-only",
        action="store_true",
        help="Run only EGARCH analysis (no XGBoost)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  EGARCH + XGBOOST HYBRID VOLATILITY MODEL")
    print("  INR/USD Exchange Rate Prediction with GDELT News")
    print("=" * 70)

    try:
        # Load data
        df = load_data()

        # Run analyses based on arguments
        if args.compare:
            run_model_comparison(df)

        if args.egarch_only:
            run_egarch_analysis(df)
        else:
            # Run EGARCH analysis first
            egarch_results = run_egarch_analysis(df)

            # Run model comparison
            run_model_comparison(df)

            # Run full hybrid pipeline
            hybrid_results = run_hybrid_pipeline(df, target_type=args.target)

        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print(f"  Results saved to: {OUTPUT_DIR}/")
        print("=" * 70)

        # List output files
        if os.path.exists(OUTPUT_DIR):
            files = os.listdir(OUTPUT_DIR)
            print("\nOutput files:")
            for f in sorted(files):
                print(f"  - {f}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the required data files exist.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

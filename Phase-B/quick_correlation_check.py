"""
Quick script to check which features correlate best with IMF 3 noise
Run this after generating india_news_thematic_features.csv
"""

import pandas as pd
import numpy as np
from scipy import stats

def quick_correlation_analysis(news_csv, imf3_csv):
    """
    Quick correlation check between news features and IMF 3

    Args:
        news_csv: Path to india_news_thematic_features.csv
        imf3_csv: Path to IMF_3.csv (should have Date and IMF_3 columns)
    """

    print("="*70)
    print("QUICK CORRELATION CHECK: News Features vs IMF 3 Noise")
    print("="*70)

    # Load data
    print("\nLoading data...")
    news_df = pd.read_csv(news_csv)
    news_df['Date'] = pd.to_datetime(news_df['Date'])

    imf3_df = pd.read_csv(imf3_csv)
    imf3_df['Date'] = pd.to_datetime(imf3_df['Date'])

    # Merge
    merged = pd.merge(news_df, imf3_df[['Date', 'IMF_3']], on='Date', how='inner')
    print(f"Merged {len(merged)} days of overlapping data")
    print(f"Date range: {merged['Date'].min()} to {merged['Date'].max()}\n")

    # Calculate correlations
    features = [
        'Tone_Economy', 'Tone_Conflict', 'Tone_Policy', 'Tone_Corporate',
        'Goldstein_Weighted', 'Goldstein_Avg',
        'Count_Economy', 'Count_Conflict', 'Count_Total',
        'Volume_Spike', 'Volume_Spike_Economy', 'Volume_Spike_Conflict'
    ]

    results = []

    for feature in features:
        if feature in merged.columns:
            # Handle NaN
            valid_data = merged[[feature, 'IMF_3']].dropna()

            if len(valid_data) > 10:  # Need at least 10 points
                corr, p_val = stats.pearsonr(valid_data[feature], valid_data['IMF_3'])

                results.append({
                    'Feature': feature,
                    'Correlation': f"{corr:+.4f}",
                    'Abs_Corr': abs(corr),
                    'P-Value': f"{p_val:.4f}",
                    'Significant': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
                })

    # Sort by absolute correlation
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Abs_Corr', ascending=False)

    # Display
    print("="*70)
    print(f"{'Feature':<25} {'Correlation':<12} {'P-Value':<10} {'Sig'}")
    print("="*70)

    for _, row in results_df.iterrows():
        print(f"{row['Feature']:<25} {row['Correlation']:<12} {row['P-Value']:<10} {row['Significant']}")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    print("Correlation: +1.0 (perfect positive) to -1.0 (perfect negative)")
    print("\nStrong: |r| > 0.5  |  Moderate: |r| > 0.3  |  Weak: |r| > 0.1")

    # Recommendations
    print("\n" + "="*70)
    print("TOP PREDICTIVE FEATURES")
    print("="*70)

    top_3 = results_df.head(3)
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"{i}. {row['Feature']} (r={row['Correlation']}, {row['Significant']})")

    print("\nNext step: Use these top features in a regression model to predict IMF 3")

    return results_df


if __name__ == "__main__":
    import sys

    # Default paths
    news_csv = "india_news_thematic_features.csv"
    imf3_csv = "../IMF_3.csv"

    # Allow command line args
    if len(sys.argv) > 1:
        news_csv = sys.argv[1]
    if len(sys.argv) > 2:
        imf3_csv = sys.argv[2]

    try:
        results = quick_correlation_analysis(news_csv, imf3_csv)

        # Save results
        results.to_csv('quick_correlation_results.csv', index=False)
        print("\nResults saved to: quick_correlation_results.csv")

    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} [news_features.csv] [imf3.csv]")
        print("\nMake sure you have:")
        print("  1. Run thematic_filter.py first to generate india_news_thematic_features.csv")
        print("  2. Have an IMF_3.csv file with Date and IMF_3 columns")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

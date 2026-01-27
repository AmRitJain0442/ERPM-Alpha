"""
Analyze V2 simulation results - Compare Statistical vs LLM vs Ensemble performance.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(filepath="simulation_results_v2.json"):
    """Load simulation results from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_model_comparison(results):
    """Compare Statistical, LLM-adjusted, and Ensemble predictions."""

    # Filter to hybrid mode only (post-warmup)
    hybrid = [r for r in results if r.get('mode') == 'hybrid' and r.get('actual_price')]

    if not hybrid:
        print("No hybrid results with actual prices found.")
        return None

    data = []
    for r in hybrid:
        stat_error = abs(r.get('stat_error_pct', 0)) if r.get('stat_error_pct') is not None else None
        ensemble_error = abs(r.get('prediction_error_pct', 0)) if r.get('prediction_error_pct') is not None else None

        data.append({
            'date': r['date'][:10],
            'actual': r['actual_price'],
            'stat_pred': r.get('stat_prediction'),
            'ensemble_pred': r.get('final_prediction'),
            'llm_adj_pct': r.get('llm_adjustment_pct', 0),
            'stat_error_pct': stat_error,
            'ensemble_error_pct': ensemble_error,
            'r_squared': r.get('stat_r_squared'),
            'llm_consensus': r.get('llm_consensus', 'neutral'),
        })

    df = pd.DataFrame(data)

    print("=" * 70)
    print("MODEL COMPARISON (Post-Warmup Period)")
    print("=" * 70)

    # Statistical model performance
    stat_valid = df[df['stat_error_pct'].notna()]
    if not stat_valid.empty:
        print("\n[STATISTICAL MODEL]")
        print(f"  Mean Absolute Error: {stat_valid['stat_error_pct'].mean():.4f}%")
        print(f"  Median Error:        {stat_valid['stat_error_pct'].median():.4f}%")
        print(f"  Std Dev:             {stat_valid['stat_error_pct'].std():.4f}%")
        print(f"  Max Error:           {stat_valid['stat_error_pct'].max():.4f}%")
        print(f"  Avg R-squared:       {stat_valid['r_squared'].mean():.4f}")

    # Ensemble performance
    ens_valid = df[df['ensemble_error_pct'].notna()]
    if not ens_valid.empty:
        print("\n[ENSEMBLE MODEL (Stat + LLM)]")
        print(f"  Mean Absolute Error: {ens_valid['ensemble_error_pct'].mean():.4f}%")
        print(f"  Median Error:        {ens_valid['ensemble_error_pct'].median():.4f}%")
        print(f"  Std Dev:             {ens_valid['ensemble_error_pct'].std():.4f}%")
        print(f"  Max Error:           {ens_valid['ensemble_error_pct'].max():.4f}%")

    # Compare
    if not stat_valid.empty and not ens_valid.empty:
        stat_mae = stat_valid['stat_error_pct'].mean()
        ens_mae = ens_valid['ensemble_error_pct'].mean()

        print("\n[COMPARISON]")
        if ens_mae < stat_mae:
            improvement = (stat_mae - ens_mae) / stat_mae * 100
            print(f"  LLM adjustment IMPROVED prediction by {improvement:.1f}%")
        else:
            degradation = (ens_mae - stat_mae) / stat_mae * 100
            print(f"  LLM adjustment DEGRADED prediction by {degradation:.1f}%")

        # How often did LLM help vs hurt?
        df['llm_helped'] = df['ensemble_error_pct'] < df['stat_error_pct']
        helped_count = df['llm_helped'].sum()
        total = len(df[df['llm_helped'].notna()])
        print(f"  LLM helped in {helped_count}/{total} cases ({100*helped_count/total:.1f}%)")

    # LLM adjustment analysis
    print("\n[LLM ADJUSTMENT ANALYSIS]")
    print(f"  Mean adjustment:  {df['llm_adj_pct'].mean():+.4f}%")
    print(f"  Std dev:          {df['llm_adj_pct'].std():.4f}%")
    print(f"  Max bullish:      {df['llm_adj_pct'].max():+.4f}%")
    print(f"  Max bearish:      {df['llm_adj_pct'].min():+.4f}%")

    # Consensus distribution
    consensus_counts = df['llm_consensus'].value_counts()
    print("\n  Consensus distribution:")
    for consensus, count in consensus_counts.items():
        print(f"    {consensus}: {count} ({100*count/len(df):.1f}%)")

    return df


def analyze_direction_accuracy(results):
    """Analyze how often the model correctly predicted direction."""

    hybrid = [r for r in results if r.get('mode') == 'hybrid' and r.get('actual_price')]

    if not hybrid:
        return

    print("\n" + "=" * 70)
    print("DIRECTION ACCURACY ANALYSIS")
    print("=" * 70)

    stat_correct = 0
    ensemble_correct = 0
    total = 0

    for r in hybrid:
        if r.get('stat_prediction') is None or r.get('actual_price') is None:
            continue

        last_close = r['last_close']
        actual = r['actual_price']
        stat_pred = r['stat_prediction']
        ens_pred = r['final_prediction']

        actual_direction = 1 if actual > last_close else (-1 if actual < last_close else 0)
        stat_direction = 1 if stat_pred > last_close else (-1 if stat_pred < last_close else 0)
        ens_direction = 1 if ens_pred > last_close else (-1 if ens_pred < last_close else 0)

        total += 1
        if stat_direction == actual_direction:
            stat_correct += 1
        if ens_direction == actual_direction:
            ensemble_correct += 1

    if total > 0:
        print(f"\nStatistical Model: {stat_correct}/{total} ({100*stat_correct/total:.1f}%)")
        print(f"Ensemble Model:    {ensemble_correct}/{total} ({100*ensemble_correct/total:.1f}%)")


def analyze_persona_contributions(results):
    """Analyze how each persona contributes to predictions."""

    hybrid = [r for r in results if r.get('mode') == 'hybrid' and 'persona_predictions' in r]

    if not hybrid:
        return

    persona_stats = {}

    for r in hybrid:
        for p in r.get('persona_predictions', []):
            name = p.get('short_name', p.get('persona_name', 'Unknown')[:15])

            if name not in persona_stats:
                persona_stats[name] = {
                    'directions': [],
                    'magnitudes': [],
                    'confidences': [],
                    'weight': p.get('weight', 0)
                }

            if p.get('success'):
                persona_stats[name]['directions'].append(p.get('direction', 'neutral'))
                persona_stats[name]['magnitudes'].append(p.get('adjustment_magnitude', 'none'))
                persona_stats[name]['confidences'].append(p.get('confidence', 5))

    print("\n" + "=" * 70)
    print("PERSONA CONTRIBUTION ANALYSIS")
    print("=" * 70)

    summary = []
    for name, stats in persona_stats.items():
        if not stats['directions']:
            continue

        bullish = sum(1 for d in stats['directions'] if d == 'bullish_usd')
        bearish = sum(1 for d in stats['directions'] if d == 'bearish_usd')
        neutral = sum(1 for d in stats['directions'] if d == 'neutral')
        total = len(stats['directions'])

        summary.append({
            'Persona': name,
            'Weight': f"{stats['weight']*100:.0f}%",
            'Bullish%': f"{100*bullish/total:.0f}%",
            'Bearish%': f"{100*bearish/total:.0f}%",
            'Neutral%': f"{100*neutral/total:.0f}%",
            'Avg Conf': f"{np.mean(stats['confidences']):.1f}",
        })

    df = pd.DataFrame(summary)
    if not df.empty:
        print("\n" + df.to_string(index=False))


def plot_comparison(results, output_file="v2_comparison_chart.png"):
    """Create comparison charts."""

    hybrid = [r for r in results if r.get('mode') == 'hybrid' and r.get('actual_price')]

    if len(hybrid) < 5:
        print("Not enough data for charts")
        return

    dates = [r['date'][:10] for r in hybrid]
    actual = [r['actual_price'] for r in hybrid]
    stat_pred = [r.get('stat_prediction', r['actual_price']) for r in hybrid]
    ensemble_pred = [r.get('final_prediction', r['actual_price']) for r in hybrid]
    llm_adj = [r.get('llm_adjustment_pct', 0) for r in hybrid]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Prices
    ax1 = axes[0]
    ax1.plot(dates, actual, 'b-', label='Actual USD/INR', linewidth=2)
    ax1.plot(dates, stat_pred, 'g--', label='Statistical Model', linewidth=1.5, alpha=0.7)
    ax1.plot(dates, ensemble_pred, 'r--', label='Ensemble (Stat+LLM)', linewidth=1.5, alpha=0.7)
    ax1.set_title('USD/INR: Actual vs Predictions')
    ax1.set_ylabel('USD/INR Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Only show every nth label to avoid crowding
    n = max(1, len(dates) // 15)
    ax1.set_xticks(dates[::n])
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Errors
    ax2 = axes[1]
    stat_errors = [(s - a) / a * 100 for s, a in zip(stat_pred, actual)]
    ensemble_errors = [(e - a) / a * 100 for e, a in zip(ensemble_pred, actual)]

    x = np.arange(len(dates))
    width = 0.35

    ax2.bar(x - width/2, stat_errors, width, label='Statistical Error', color='green', alpha=0.6)
    ax2.bar(x + width/2, ensemble_errors, width, label='Ensemble Error', color='red', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Prediction Errors (%)')
    ax2.set_ylabel('Error %')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x[::n])
    ax2.set_xticklabels([dates[i] for i in range(0, len(dates), n)], rotation=45)

    # Plot 3: LLM Adjustments
    ax3 = axes[2]
    colors = ['green' if adj > 0 else 'red' if adj < 0 else 'gray' for adj in llm_adj]
    ax3.bar(dates, llm_adj, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('LLM Sentiment Adjustments (%)')
    ax3.set_ylabel('Adjustment %')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(dates[::n])
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nChart saved to: {output_file}")


def plot_error_distribution(results, output_file="v2_error_distribution.png"):
    """Plot error distributions for statistical vs ensemble."""

    hybrid = [r for r in results if r.get('mode') == 'hybrid' and r.get('actual_price')]

    if len(hybrid) < 10:
        return

    stat_errors = [r.get('stat_error_pct', 0) for r in hybrid if r.get('stat_error_pct') is not None]
    ensemble_errors = [r.get('prediction_error_pct', 0) for r in hybrid if r.get('prediction_error_pct') is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1 = axes[0]
    bins = np.linspace(-1, 1, 40)
    ax1.hist(stat_errors, bins=bins, alpha=0.5, label='Statistical', color='green')
    ax1.hist(ensemble_errors, bins=bins, alpha=0.5, label='Ensemble', color='red')
    ax1.axvline(x=0, color='black', linestyle='--')
    ax1.set_title('Error Distribution')
    ax1.set_xlabel('Error %')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    ax2.boxplot([stat_errors, ensemble_errors], labels=['Statistical', 'Ensemble'])
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Error Box Plot')
    ax2.set_ylabel('Error %')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f"Error distribution chart saved to: {output_file}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "simulation_results_v2.json")

    if not os.path.exists(results_path):
        print("No V2 simulation results found. Run run_simulation_v2.py first.")
        return

    print("Loading V2 results...")
    results = load_results(results_path)
    print(f"Found {len(results)} trading days of data\n")

    # Model comparison
    df = analyze_model_comparison(results)

    # Direction accuracy
    analyze_direction_accuracy(results)

    # Persona analysis
    analyze_persona_contributions(results)

    # Generate charts
    print("\nGenerating charts...")
    try:
        plot_comparison(results, os.path.join(script_dir, "v2_comparison_chart.png"))
        plot_error_distribution(results, os.path.join(script_dir, "v2_error_distribution.png"))
    except Exception as e:
        print(f"Could not generate charts: {e}")


if __name__ == "__main__":
    main()

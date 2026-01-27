"""
Analyze and visualize USD/INR forex simulation results.
Run this after the simulation completes.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results(filepath="simulation_results.json"):
    """Load simulation results from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_summary_table(results):
    """Create a summary table of predictions vs actuals."""
    data = []
    for r in results:
        data.append({
            'Date': r['date'][:10],
            'Previous Close': r['last_close'],
            'Predicted': r['weighted_prediction'],
            'Actual': r['actual_price'],
            'Error %': r['prediction_error_pct'],
            'Consensus': r['consensus'],
            'Confidence': r['avg_confidence']
        })
    return pd.DataFrame(data)


def analyze_persona_performance(results):
    """Analyze each persona's prediction accuracy."""
    persona_stats = {}

    for r in results:
        actual = r['actual_price']
        if actual is None:
            continue

        for p in r['persona_predictions']:
            name = p['persona_name']
            if name not in persona_stats:
                persona_stats[name] = {
                    'predictions': [],
                    'errors': [],
                    'weight': p['weight']
                }

            if p['success'] and p.get('predicted_usdinr'):
                pred = p['predicted_usdinr']
                error_pct = ((pred - actual) / actual) * 100
                persona_stats[name]['predictions'].append(pred)
                persona_stats[name]['errors'].append(error_pct)

    # Calculate statistics
    summary = []
    for name, stats in persona_stats.items():
        if stats['errors']:
            errors = stats['errors']
            summary.append({
                'Persona': name,
                'Weight': f"{stats['weight']*100:.0f}%",
                'Predictions': len(errors),
                'MAE %': f"{sum(abs(e) for e in errors)/len(errors):.2f}",
                'Bias %': f"{sum(errors)/len(errors):+.2f}",
                'Best': f"{min(abs(e) for e in errors):.2f}",
                'Worst': f"{max(abs(e) for e in errors):.2f}"
            })

    df = pd.DataFrame(summary)
    if not df.empty and 'MAE %' in df.columns:
        return df.sort_values('MAE %')
    return df


def plot_predictions_vs_actual(results, output_file="prediction_chart.png"):
    """Create a chart comparing predictions to actual prices."""
    dates = []
    predicted = []
    actual = []

    for r in results:
        if r['actual_price'] is not None:
            dates.append(r['date'][:10])
            predicted.append(r['weighted_prediction'])
            actual.append(r['actual_price'])

    plt.figure(figsize=(14, 7))

    plt.subplot(2, 1, 1)
    plt.plot(dates, actual, 'b-', label='Actual USD/INR', linewidth=2)
    plt.plot(dates, predicted, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    plt.fill_between(dates, actual, predicted, alpha=0.3, color='yellow')
    plt.title('Gemini Persona Simulation: USD/INR Predictions vs Actual')
    plt.ylabel('USD/INR Rate')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    errors = [(p - a) / a * 100 for p, a in zip(predicted, actual)]
    colors = ['green' if abs(e) < 1 else 'orange' if abs(e) < 2 else 'red' for e in errors]
    plt.bar(dates, errors, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Prediction Error (%)')
    plt.ylabel('Error %')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f"Chart saved to: {output_file}")


def plot_persona_comparison(results, output_file="persona_comparison.png"):
    """Create a chart showing persona prediction distribution."""
    persona_errors = {}

    for r in results:
        actual = r['actual_price']
        if actual is None:
            continue

        for p in r['persona_predictions']:
            if p['success'] and p.get('predicted_usdinr'):
                name = p['short_name'] if 'short_name' in p else p['persona_name'][:15]
                if name not in persona_errors:
                    persona_errors[name] = []
                error = ((p['predicted_usdinr'] - actual) / actual) * 100
                persona_errors[name].append(error)

    # Box plot
    plt.figure(figsize=(12, 6))
    names = list(persona_errors.keys())
    data = [persona_errors[n] for n in names]

    plt.boxplot(data, labels=names)
    plt.axhline(y=0, color='green', linestyle='--', linewidth=1)
    plt.title('USD/INR Prediction Error Distribution by Persona')
    plt.ylabel('Error %')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()
    print(f"Chart saved to: {output_file}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "simulation_results.json")

    if not os.path.exists(results_path):
        print("No simulation results found. Run the simulation first.")
        return

    print("Loading results...")
    results = load_results(results_path)
    print(f"Found {len(results)} trading days of data\n")

    # Summary table
    print("=" * 70)
    print("DAILY PREDICTIONS SUMMARY")
    print("=" * 70)
    summary_df = create_summary_table(results)
    print(summary_df.to_string(index=False))

    # Overall statistics
    valid = summary_df[summary_df['Error %'].notna()]
    if not valid.empty:
        print("\n" + "=" * 70)
        print("OVERALL PERFORMANCE")
        print("=" * 70)
        print(f"Mean Absolute Error:  {valid['Error %'].abs().mean():.2f}%")
        print(f"Mean Error (Bias):    {valid['Error %'].mean():+.2f}%")
        print(f"Std Dev of Error:     {valid['Error %'].std():.2f}%")
        print(f"Best Prediction:      {valid['Error %'].abs().min():.2f}%")
        print(f"Worst Prediction:     {valid['Error %'].abs().max():.2f}%")

    # Persona analysis
    print("\n" + "=" * 70)
    print("PERSONA PERFORMANCE")
    print("=" * 70)
    persona_df = analyze_persona_performance(results)
    print(persona_df.to_string(index=False))

    # Generate charts
    print("\nGenerating charts...")
    try:
        plot_predictions_vs_actual(results, os.path.join(script_dir, "prediction_chart.png"))
        plot_persona_comparison(results, os.path.join(script_dir, "persona_comparison.png"))
    except Exception as e:
        print(f"Could not generate charts: {e}")


if __name__ == "__main__":
    main()

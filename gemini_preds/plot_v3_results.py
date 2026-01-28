"""
Visualization and Analysis of V3 Simulation Results
Compares LLM ensemble predictions vs actual USD/INR rates
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import os

def load_results(filepath="simulation_results_v3.json"):
    """Load V3 simulation results."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_dataframe(results):
    """Convert results to DataFrame for analysis."""
    data = []
    for r in results:
        if r.get('actual_price') is not None:
            data.append({
                'date': pd.to_datetime(r['date']),
                'mode': r.get('mode', 'hybrid'),
                'last_close': r.get('last_close'),
                'stat_prediction': r.get('stat_prediction'),
                'final_prediction': r.get('final_prediction'),
                'actual_price': r.get('actual_price'),
                'prediction_error_pct': r.get('prediction_error_pct'),
                'stat_error_pct': r.get('stat_error_pct'),
                'llm_adjustment_pct': r.get('llm_adjustment_pct', 0),
                'bullish_pct': r.get('bullish_pct', 0),
                'bearish_pct': r.get('bearish_pct', 0),
                'stat_weight': r.get('stat_weight', 0.75),
                'llm_weight': r.get('llm_weight', 0.25),
                'stat_r_squared': r.get('stat_r_squared', 0),
            })
    return pd.DataFrame(data)


def plot_predictions_vs_actual(df, output_dir="."):
    """Main chart: Predictions vs Actual prices over time."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    
    # Filter to hybrid mode only for fair comparison
    hybrid = df[df['mode'] == 'hybrid'].copy()
    warmup = df[df['mode'] == 'warmup'].copy()
    
    # ===== Panel 1: Price Comparison =====
    ax1 = axes[0]
    
    # Actual prices
    ax1.plot(df['date'], df['actual_price'], 'b-', linewidth=1.5, label='Actual USD/INR', alpha=0.9)
    
    # Statistical predictions
    ax1.plot(df['date'], df['stat_prediction'], 'g--', linewidth=1, label='Statistical Model', alpha=0.7)
    
    # Final ensemble predictions (hybrid mode only)
    if not hybrid.empty:
        ax1.plot(hybrid['date'], hybrid['final_prediction'], 'r-', linewidth=1.2, 
                label='V3 Ensemble (Stat+LLM)', alpha=0.8)
    
    # Mark warmup period
    if not warmup.empty:
        ax1.axvspan(warmup['date'].min(), warmup['date'].max(), 
                   alpha=0.1, color='gray', label='Warmup Period')
    
    ax1.set_ylabel('USD/INR Rate', fontsize=12)
    ax1.set_title('V3 Simulation: Predictions vs Actual USD/INR (2023)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== Panel 2: Prediction Errors =====
    ax2 = axes[1]
    
    if not hybrid.empty:
        # Bar chart of errors
        colors = ['green' if e >= 0 else 'red' for e in hybrid['prediction_error_pct']]
        ax2.bar(hybrid['date'], hybrid['prediction_error_pct'], color=colors, alpha=0.6, width=1.5)
        
        # Moving average of absolute error
        hybrid_sorted = hybrid.sort_values('date')
        rolling_mae = hybrid_sorted['prediction_error_pct'].abs().rolling(window=20, min_periods=5).mean()
        ax2.plot(hybrid_sorted['date'], rolling_mae, 'k-', linewidth=2, label='20-day Rolling MAE')
        
        # Zero line
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Stats annotation
        mae = hybrid['prediction_error_pct'].abs().mean()
        bias = hybrid['prediction_error_pct'].mean()
        ax2.text(0.02, 0.95, f'MAE: {mae:.3f}%\nBias: {bias:+.3f}%', 
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_ylabel('Prediction Error (%)', fontsize=12)
    ax2.set_title('Prediction Errors (Positive = Overestimated, Negative = Underestimated)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.5, 1.5)
    
    # ===== Panel 3: LLM Contribution =====
    ax3 = axes[2]
    
    if not hybrid.empty:
        # LLM adjustment percentage
        ax3.fill_between(hybrid['date'], 0, hybrid['llm_adjustment_pct'], 
                        where=hybrid['llm_adjustment_pct'] >= 0, 
                        alpha=0.5, color='green', label='Bullish adjustment')
        ax3.fill_between(hybrid['date'], 0, hybrid['llm_adjustment_pct'], 
                        where=hybrid['llm_adjustment_pct'] < 0, 
                        alpha=0.5, color='red', label='Bearish adjustment')
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add sentiment balance on secondary axis
        ax3_twin = ax3.twinx()
        ax3_twin.plot(hybrid['date'], hybrid['bullish_pct'] * 100, 'g:', alpha=0.5, label='Bullish %')
        ax3_twin.plot(hybrid['date'], hybrid['bearish_pct'] * 100, 'r:', alpha=0.5, label='Bearish %')
        ax3_twin.set_ylabel('Sentiment Split (%)', fontsize=10, color='gray')
        ax3_twin.tick_params(axis='y', labelcolor='gray')
        ax3_twin.set_ylim(0, 100)
    
    ax3.set_ylabel('LLM Adjustment (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('LLM Sentiment Adjustments to Statistical Baseline', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.5, 0.5)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'v3_predictions_vs_actual.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_model_comparison(df, output_dir="."):
    """Compare Statistical-only vs Ensemble performance."""
    hybrid = df[df['mode'] == 'hybrid'].copy()
    
    if hybrid.empty:
        print("No hybrid data for model comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ===== Panel 1: Scatter - Stat vs Actual =====
    ax1 = axes[0, 0]
    ax1.scatter(hybrid['actual_price'], hybrid['stat_prediction'], alpha=0.5, s=20, c='green', label='Statistical')
    
    # Perfect prediction line
    min_val = min(hybrid['actual_price'].min(), hybrid['stat_prediction'].min())
    max_val = max(hybrid['actual_price'].max(), hybrid['stat_prediction'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect')
    
    stat_mae = hybrid['stat_error_pct'].abs().mean()
    ax1.set_xlabel('Actual USD/INR', fontsize=11)
    ax1.set_ylabel('Statistical Prediction', fontsize=11)
    ax1.set_title(f'Statistical Model\nMAE: {stat_mae:.3f}%', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Panel 2: Scatter - Ensemble vs Actual =====
    ax2 = axes[0, 1]
    ax2.scatter(hybrid['actual_price'], hybrid['final_prediction'], alpha=0.5, s=20, c='red', label='Ensemble')
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect')
    
    ens_mae = hybrid['prediction_error_pct'].abs().mean()
    ax2.set_xlabel('Actual USD/INR', fontsize=11)
    ax2.set_ylabel('Ensemble Prediction', fontsize=11)
    ax2.set_title(f'V3 Ensemble (Stat + LLM)\nMAE: {ens_mae:.3f}%', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ===== Panel 3: Error Distribution Comparison =====
    ax3 = axes[1, 0]
    ax3.hist(hybrid['stat_error_pct'].dropna(), bins=30, alpha=0.5, color='green', label='Statistical', density=True)
    ax3.hist(hybrid['prediction_error_pct'].dropna(), bins=30, alpha=0.5, color='red', label='Ensemble', density=True)
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Prediction Error (%)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Error Distribution Comparison', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ===== Panel 4: LLM Help Analysis =====
    ax4 = axes[1, 1]
    
    # Calculate when LLM helped
    valid = hybrid[hybrid['stat_error_pct'].notna()].copy()
    valid['llm_helped'] = valid['prediction_error_pct'].abs() < valid['stat_error_pct'].abs()
    valid['llm_hurt'] = valid['prediction_error_pct'].abs() > valid['stat_error_pct'].abs()
    
    helped_count = valid['llm_helped'].sum()
    hurt_count = valid['llm_hurt'].sum()
    same_count = len(valid) - helped_count - hurt_count
    
    labels = ['LLM Helped', 'LLM Hurt', 'No Change']
    sizes = [helped_count, hurt_count, same_count]
    colors = ['lightgreen', 'lightcoral', 'lightgray']
    explode = (0.05, 0.05, 0)
    
    ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax4.set_title(f'LLM Impact on Predictions\n(n={len(valid)} trading days)', fontsize=12)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'v3_model_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_direction_accuracy(df, output_dir="."):
    """Analyze direction prediction accuracy."""
    hybrid = df[df['mode'] == 'hybrid'].copy()
    
    if hybrid.empty:
        print("No hybrid data for direction analysis")
        return
    
    # Calculate direction accuracy
    hybrid['actual_direction'] = np.where(hybrid['actual_price'] > hybrid['last_close'], 'up', 'down')
    hybrid['predicted_direction'] = np.where(hybrid['final_prediction'] > hybrid['last_close'], 'up', 'down')
    hybrid['stat_direction'] = np.where(hybrid['stat_prediction'] > hybrid['last_close'], 'up', 'down')
    
    hybrid['ens_correct'] = hybrid['actual_direction'] == hybrid['predicted_direction']
    hybrid['stat_correct'] = hybrid['actual_direction'] == hybrid['stat_direction']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Direction Accuracy Over Time =====
    ax1 = axes[0]
    
    # Rolling accuracy
    window = 20
    hybrid_sorted = hybrid.sort_values('date')
    ens_rolling = hybrid_sorted['ens_correct'].rolling(window=window, min_periods=10).mean() * 100
    stat_rolling = hybrid_sorted['stat_correct'].rolling(window=window, min_periods=10).mean() * 100
    
    ax1.plot(hybrid_sorted['date'], ens_rolling, 'r-', linewidth=2, label='Ensemble', alpha=0.8)
    ax1.plot(hybrid_sorted['date'], stat_rolling, 'g--', linewidth=2, label='Statistical', alpha=0.8)
    ax1.axhline(y=50, color='gray', linestyle=':', linewidth=1, label='Random (50%)')
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Direction Accuracy (%)', fontsize=11)
    ax1.set_title(f'{window}-Day Rolling Direction Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(30, 80)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # ===== Summary Bar Chart =====
    ax2 = axes[1]
    
    ens_acc = hybrid['ens_correct'].mean() * 100
    stat_acc = hybrid['stat_correct'].mean() * 100
    
    bars = ax2.bar(['Statistical\nModel', 'V3 Ensemble\n(Stat + LLM)'], 
                   [stat_acc, ens_acc], 
                   color=['green', 'red'], alpha=0.7)
    
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Random baseline')
    ax2.set_ylabel('Direction Accuracy (%)', fontsize=11)
    ax2.set_title('Overall Direction Prediction Accuracy', fontsize=12)
    ax2.set_ylim(40, 70)
    
    # Add value labels on bars
    for bar, val in zip(bars, [stat_acc, ens_acc]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'v3_direction_accuracy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_stats(df):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 70)
    print("V3 SIMULATION COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    hybrid = df[df['mode'] == 'hybrid']
    warmup = df[df['mode'] == 'warmup']
    
    print(f"\nData Period: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total Trading Days: {len(df)}")
    print(f"  - Warmup Days: {len(warmup)}")
    print(f"  - Hybrid Days: {len(hybrid)}")
    
    if not hybrid.empty:
        print("\n" + "-" * 50)
        print("ENSEMBLE MODEL PERFORMANCE (Hybrid Mode Only)")
        print("-" * 50)
        
        ens_mae = hybrid['prediction_error_pct'].abs().mean()
        ens_bias = hybrid['prediction_error_pct'].mean()
        ens_std = hybrid['prediction_error_pct'].std()
        
        print(f"\nEnsemble (Stat + LLM):")
        print(f"  Mean Absolute Error:  {ens_mae:.4f}%")
        print(f"  Bias (Mean Error):    {ens_bias:+.4f}%")
        print(f"  Std Dev:              {ens_std:.4f}%")
        print(f"  Best Day:             {hybrid['prediction_error_pct'].abs().min():.4f}%")
        print(f"  Worst Day:            {hybrid['prediction_error_pct'].abs().max():.4f}%")
        
        stat_valid = hybrid[hybrid['stat_error_pct'].notna()]
        if not stat_valid.empty:
            stat_mae = stat_valid['stat_error_pct'].abs().mean()
            stat_bias = stat_valid['stat_error_pct'].mean()
            stat_std = stat_valid['stat_error_pct'].std()
            
            print(f"\nStatistical Model Only:")
            print(f"  Mean Absolute Error:  {stat_mae:.4f}%")
            print(f"  Bias (Mean Error):    {stat_bias:+.4f}%")
            print(f"  Std Dev:              {stat_std:.4f}%")
            
            # Improvement
            improvement = stat_mae - ens_mae
            print(f"\n  Ensemble vs Stat: {improvement:+.4f}% MAE difference")
            if improvement > 0:
                print(f"  --> LLM IMPROVED predictions by {improvement:.4f}%")
            else:
                print(f"  --> LLM DEGRADED predictions by {-improvement:.4f}%")
        
        # Direction accuracy
        hybrid_copy = hybrid.copy()
        hybrid_copy['actual_dir'] = hybrid_copy['actual_price'] > hybrid_copy['last_close']
        hybrid_copy['pred_dir'] = hybrid_copy['final_prediction'] > hybrid_copy['last_close']
        hybrid_copy['stat_dir'] = hybrid_copy['stat_prediction'] > hybrid_copy['last_close']
        
        ens_dir_acc = (hybrid_copy['actual_dir'] == hybrid_copy['pred_dir']).mean() * 100
        stat_dir_acc = (hybrid_copy['actual_dir'] == hybrid_copy['stat_dir']).mean() * 100
        
        print("\n" + "-" * 50)
        print("DIRECTION PREDICTION ACCURACY")
        print("-" * 50)
        print(f"  Ensemble:    {ens_dir_acc:.1f}%")
        print(f"  Statistical: {stat_dir_acc:.1f}%")
        print(f"  Random:      50.0%")
        
        # LLM impact
        print("\n" + "-" * 50)
        print("LLM CONTRIBUTION ANALYSIS")
        print("-" * 50)
        
        avg_bullish = hybrid['bullish_pct'].mean() * 100
        avg_bearish = hybrid['bearish_pct'].mean() * 100
        avg_adjustment = hybrid['llm_adjustment_pct'].mean()
        
        print(f"  Avg Bullish Sentiment: {avg_bullish:.1f}%")
        print(f"  Avg Bearish Sentiment: {avg_bearish:.1f}%")
        print(f"  Avg LLM Adjustment:    {avg_adjustment:+.4f}%")
        print(f"  Avg Statistical Weight: {hybrid['stat_weight'].mean()*100:.1f}%")
        print(f"  Avg LLM Weight:         {hybrid['llm_weight'].mean()*100:.1f}%")
        
        # When did LLM help?
        valid = hybrid[hybrid['stat_error_pct'].notna()].copy()
        valid['llm_helped'] = valid['prediction_error_pct'].abs() < valid['stat_error_pct'].abs()
        helped = valid['llm_helped'].sum()
        total = len(valid)
        
        print(f"\n  LLM helped in {helped}/{total} days ({100*helped/total:.1f}%)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "simulation_results_v3.json")
    
    print("Loading V3 simulation results...")
    results = load_results(results_path)
    print(f"Loaded {len(results)} days of data")
    
    df = prepare_dataframe(results)
    print(f"Prepared DataFrame with {len(df)} valid entries")
    
    # Generate all plots
    print("\nGenerating visualizations...")
    plot_predictions_vs_actual(df, script_dir)
    plot_model_comparison(df, script_dir)
    plot_direction_accuracy(df, script_dir)
    
    # Print summary
    generate_summary_stats(df)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - v3_predictions_vs_actual.png")
    print(f"  - v3_model_comparison.png")
    print(f"  - v3_direction_accuracy.png")


if __name__ == "__main__":
    main()

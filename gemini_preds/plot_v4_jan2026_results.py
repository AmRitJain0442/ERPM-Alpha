"""
Visualization for V4 January 2026 Simulation Results

Generates comprehensive plots analyzing the simulation performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_data():
    """Load simulation results."""
    # Load summary CSV
    summary_df = pd.read_csv('simulation_summary_v4_jan2026.csv')
    summary_df['date'] = pd.to_datetime(summary_df['date'])
    
    # Load detailed JSON results
    with open('simulation_results_v4_jan2026.json', 'r') as f:
        detailed_results = json.load(f)
    
    return summary_df, detailed_results


def plot_predictions_vs_actual(df):
    """Plot predictions vs actual prices."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Price predictions
    ax1 = axes[0]
    
    # Plot actual prices
    ax1.plot(df['date'], df['actual_price'], 'b-', linewidth=2.5, 
             marker='o', markersize=6, label='Actual USD/INR', zorder=3)
    
    # Plot final predictions (ensemble)
    ax1.plot(df['date'], df['final_prediction'], 'g--', linewidth=2, 
             marker='s', markersize=5, label='Ensemble Prediction', zorder=2)
    
    # Plot statistical predictions
    ax1.plot(df['date'], df['stat_prediction'], 'r:', linewidth=1.5, 
             marker='^', markersize=4, label='Statistical Prediction', alpha=0.7, zorder=1)
    
    # Highlight warmup period
    warmup_df = df[df['mode'] == 'warmup']
    if not warmup_df.empty:
        ax1.axvspan(warmup_df['date'].min(), warmup_df['date'].max(), 
                   alpha=0.2, color='gray', label='Warmup Period')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('USD/INR Exchange Rate')
    ax1.set_title('V4 Model: USD/INR Predictions vs Actual - January 2026', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Prediction errors
    ax2 = axes[1]
    
    hybrid_df = df[df['mode'] == 'hybrid'].copy()
    
    # Bar chart of errors
    x = np.arange(len(hybrid_df))
    width = 0.35
    
    ensemble_errors = hybrid_df['prediction_error_pct'].values
    stat_errors = hybrid_df['stat_error_pct'].values
    
    bars1 = ax2.bar(x - width/2, ensemble_errors, width, label='Ensemble Error', 
                    color=['green' if e >= 0 else 'red' for e in ensemble_errors], alpha=0.7)
    bars2 = ax2.bar(x + width/2, stat_errors, width, label='Stat-Only Error',
                    color='gray', alpha=0.5)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Prediction Error (%)')
    ax2.set_title('Prediction Errors: Ensemble vs Statistical Model (Hybrid Mode Only)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.strftime('%b %d') for d in hybrid_df['date']], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('v4_jan2026_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved: v4_jan2026_predictions.png")
    plt.show()


def plot_model_comparison(df):
    """Plot comparison metrics between ensemble and stat-only model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    hybrid_df = df[df['mode'] == 'hybrid'].copy()
    
    # Top Left: Error distribution
    ax1 = axes[0, 0]
    ensemble_errors = hybrid_df['prediction_error_pct'].abs()
    stat_errors = hybrid_df['stat_error_pct'].abs()
    
    bp = ax1.boxplot([ensemble_errors, stat_errors], 
                     labels=['Ensemble', 'Stat-Only'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('gray')
    bp['boxes'][1].set_alpha(0.6)
    
    ax1.set_ylabel('Absolute Error (%)')
    ax1.set_title('Error Distribution Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    ax1.scatter([1], [ensemble_errors.mean()], color='darkgreen', s=100, zorder=5, marker='D', label=f'Mean: {ensemble_errors.mean():.3f}%')
    ax1.scatter([2], [stat_errors.mean()], color='darkgray', s=100, zorder=5, marker='D', label=f'Mean: {stat_errors.mean():.3f}%')
    ax1.legend()
    
    # Top Right: LLM Contribution
    ax2 = axes[0, 1]
    
    # Count where LLM helped vs hurt
    llm_helped = (ensemble_errors < stat_errors).sum()
    llm_hurt = (ensemble_errors > stat_errors).sum()
    llm_same = (ensemble_errors == stat_errors).sum()
    
    colors = ['green', 'red', 'gray']
    sizes = [llm_helped, llm_hurt, llm_same]
    labels = [f'LLM Helped\n({llm_helped})', f'LLM Hurt\n({llm_hurt})', f'Same\n({llm_same})']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, explode=(0.05, 0, 0))
    ax2.set_title('LLM Contribution Analysis', fontweight='bold')
    
    # Bottom Left: Direction accuracy
    ax3 = axes[1, 0]
    
    # Calculate direction predictions
    hybrid_df['actual_direction'] = (hybrid_df['actual_price'] > hybrid_df['last_close']).map({True: 'Up', False: 'Down'})
    hybrid_df['pred_direction'] = (hybrid_df['final_prediction'] > hybrid_df['last_close']).map({True: 'Up', False: 'Down'})
    hybrid_df['stat_direction'] = (hybrid_df['stat_prediction'] > hybrid_df['last_close']).map({True: 'Up', False: 'Down'})
    
    ensemble_correct = (hybrid_df['actual_direction'] == hybrid_df['pred_direction']).sum()
    stat_correct = (hybrid_df['actual_direction'] == hybrid_df['stat_direction']).sum()
    total = len(hybrid_df)
    
    x = ['Ensemble', 'Stat-Only']
    y = [ensemble_correct / total * 100, stat_correct / total * 100]
    colors = ['green', 'gray']
    
    bars = ax3.bar(x, y, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='Random (50%)')
    
    for bar, val in zip(bars, y):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('Direction Accuracy (%)')
    ax3.set_title('Direction Prediction Accuracy', fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom Right: Weights over time
    ax4 = axes[1, 1]
    
    ax4.fill_between(hybrid_df['date'], 0, hybrid_df['stat_weight'] * 100, 
                     alpha=0.6, color='blue', label='Stat Weight')
    ax4.fill_between(hybrid_df['date'], hybrid_df['stat_weight'] * 100, 100,
                     alpha=0.6, color='orange', label='LLM Weight')
    
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Weight (%)')
    ax4.set_title('Dynamic Model Weights Over Time', fontweight='bold')
    ax4.legend(loc='center right')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax4.set_ylim(0, 100)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v4_jan2026_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: v4_jan2026_comparison.png")
    plt.show()


def plot_llm_sentiment(df):
    """Plot LLM sentiment and consensus metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    hybrid_df = df[df['mode'] == 'hybrid'].copy()
    
    # Top Left: Higher vs Lower percentages
    ax1 = axes[0, 0]
    
    x = np.arange(len(hybrid_df))
    width = 0.4
    
    ax1.bar(x - width/2, hybrid_df['higher_pct'] * 100, width, 
            label='Bullish USD (Higher)', color='green', alpha=0.7)
    ax1.bar(x + width/2, hybrid_df['lower_pct'] * 100, width,
            label='Bearish USD (Lower)', color='red', alpha=0.7)
    
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Persona Vote Share (%)')
    ax1.set_title('LLM Persona Consensus: Direction Votes', fontweight='bold')
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([d.strftime('%b %d') for d in hybrid_df['date']][::2], rotation=45)
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Top Right: Entropy confidence
    ax2 = axes[0, 1]
    
    colors = ['green' if c > 0.5 else 'orange' if c > 0.3 else 'red' 
              for c in hybrid_df['entropy_confidence']]
    
    ax2.bar(hybrid_df['date'], hybrid_df['entropy_confidence'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Moderate Confidence')
    ax2.axhline(y=1.0, color='green', linestyle=':', linewidth=1, label='High Confidence')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Entropy Confidence')
    ax2.set_title('LLM Consensus Confidence (1=Full Agreement)', fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom Left: LLM Adjustment percentages
    ax3 = axes[1, 0]
    
    colors = ['green' if a >= 0 else 'red' for a in hybrid_df['llm_adjustment_pct']]
    ax3.bar(hybrid_df['date'], hybrid_df['llm_adjustment_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax3.set_xlabel('Date')
    ax3.set_ylabel('LLM Adjustment (%)')
    ax3.set_title('LLM Price Adjustment Recommendations', fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bottom Right: Statistical R-squared over time
    ax4 = axes[1, 1]
    
    ax4.plot(df['date'], df['stat_r_squared'], 'b-', linewidth=2, marker='o', markersize=5)
    ax4.fill_between(df['date'], 0, df['stat_r_squared'], alpha=0.3)
    
    ax4.set_xlabel('Date')
    ax4.set_ylabel('R² Score')
    ax4.set_title('Statistical Model R² Over Time', fontweight='bold')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax4.set_ylim(0, 1)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add annotation for average R²
    avg_r2 = df['stat_r_squared'].mean()
    ax4.axhline(y=avg_r2, color='red', linestyle='--', linewidth=1.5)
    ax4.text(df['date'].iloc[-1], avg_r2 + 0.02, f'Avg: {avg_r2:.3f}', 
             color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('v4_jan2026_llm_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: v4_jan2026_llm_analysis.png")
    plt.show()


def plot_summary_dashboard(df):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 12))
    
    hybrid_df = df[df['mode'] == 'hybrid'].copy()
    
    # Calculate key metrics
    ensemble_mae = hybrid_df['prediction_error_pct'].abs().mean()
    stat_mae = hybrid_df['stat_error_pct'].abs().mean()
    ensemble_bias = hybrid_df['prediction_error_pct'].mean()
    stat_bias = hybrid_df['stat_error_pct'].mean()
    
    llm_helped = (hybrid_df['prediction_error_pct'].abs() < hybrid_df['stat_error_pct'].abs()).sum()
    llm_help_pct = llm_helped / len(hybrid_df) * 100
    
    # Direction accuracy
    hybrid_df['correct_direction'] = (
        ((hybrid_df['actual_price'] > hybrid_df['last_close']) & 
         (hybrid_df['final_prediction'] > hybrid_df['last_close'])) |
        ((hybrid_df['actual_price'] < hybrid_df['last_close']) & 
         (hybrid_df['final_prediction'] < hybrid_df['last_close']))
    )
    direction_accuracy = hybrid_df['correct_direction'].mean() * 100
    
    avg_higher_pct = hybrid_df['higher_pct'].mean() * 100
    avg_lower_pct = hybrid_df['lower_pct'].mean() * 100
    
    # Create title with key metrics
    fig.suptitle('V4 Simulation Results - January 2026 Final Test', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Main price chart (larger)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    
    ax1.plot(df['date'], df['actual_price'], 'b-', linewidth=2.5, 
             marker='o', markersize=6, label='Actual', zorder=3)
    ax1.plot(df['date'], df['final_prediction'], 'g--', linewidth=2, 
             marker='s', markersize=5, label='Ensemble', zorder=2)
    
    # Shade warmup
    warmup_df = df[df['mode'] == 'warmup']
    if not warmup_df.empty:
        ax1.axvspan(warmup_df['date'].min(), warmup_df['date'].max(), 
                   alpha=0.2, color='gray', label='Warmup')
    
    ax1.set_title('USD/INR: Predictions vs Actual', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Exchange Rate')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Metrics panel (right side)
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax2.axis('off')
    
    metrics_text = f"""
    ╔══════════════════════════════╗
    ║      KEY METRICS             ║
    ╠══════════════════════════════╣
    ║                              ║
    ║  Ensemble MAE:  {ensemble_mae:.4f}%     ║
    ║  Stat-Only MAE: {stat_mae:.4f}%     ║
    ║                              ║
    ║  Ensemble Bias: {ensemble_bias:+.4f}%    ║
    ║  Stat-Only Bias:{stat_bias:+.4f}%    ║
    ║                              ║
    ║  LLM Helped: {llm_help_pct:.1f}%          ║
    ║  ({llm_helped}/{len(hybrid_df)} trading days)        ║
    ║                              ║
    ║  Direction Acc: {direction_accuracy:.1f}%         ║
    ║                              ║
    ║  Avg Bull Vote: {avg_higher_pct:.1f}%         ║
    ║  Avg Bear Vote: {avg_lower_pct:.1f}%         ║
    ║                              ║
    ║  Hybrid Days: {len(hybrid_df)}             ║
    ║  Warmup Days: {len(df) - len(hybrid_df)}              ║
    ║                              ║
    ╚══════════════════════════════╝
    """
    
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Error comparison bar chart
    ax3 = plt.subplot2grid((3, 3), (2, 0))
    
    x = ['Ensemble', 'Stat-Only']
    y = [ensemble_mae, stat_mae]
    colors = ['green', 'gray']
    bars = ax3.bar(x, y, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, y):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('MAE (%)')
    ax3.set_title('Mean Absolute Error', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # LLM contribution pie
    ax4 = plt.subplot2grid((3, 3), (2, 1))
    
    helped = (hybrid_df['prediction_error_pct'].abs() < hybrid_df['stat_error_pct'].abs()).sum()
    hurt = (hybrid_df['prediction_error_pct'].abs() > hybrid_df['stat_error_pct'].abs()).sum()
    
    sizes = [helped, hurt]
    labels = [f'Helped ({helped})', f'Hurt ({hurt})']
    colors = ['green', 'red']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0))
    ax4.set_title('LLM Impact', fontweight='bold')
    
    # Price movement
    ax5 = plt.subplot2grid((3, 3), (2, 2))
    
    price_change = ((df['actual_price'].iloc[-1] - df['actual_price'].iloc[0]) / 
                   df['actual_price'].iloc[0] * 100)
    
    ax5.text(0.5, 0.6, f'{price_change:+.2f}%', transform=ax5.transAxes,
             fontsize=28, ha='center', va='center', fontweight='bold',
             color='red' if price_change > 0 else 'green')
    ax5.text(0.5, 0.3, f'INR Depreciation' if price_change > 0 else 'INR Appreciation',
             transform=ax5.transAxes, fontsize=12, ha='center')
    ax5.text(0.5, 0.1, f'{df["actual_price"].iloc[0]:.2f} → {df["actual_price"].iloc[-1]:.2f}',
             transform=ax5.transAxes, fontsize=10, ha='center', color='gray')
    ax5.set_title('Jan 2026 Price Change', fontweight='bold')
    ax5.axis('off')
    
    plt.tight_layout()
    plt.savefig('v4_jan2026_dashboard.png', dpi=150, bbox_inches='tight')
    print("Saved: v4_jan2026_dashboard.png")
    plt.show()


def main():
    print("=" * 60)
    print("V4 JANUARY 2026 SIMULATION - VISUALIZATION")
    print("=" * 60)
    print()
    
    # Load data
    df, detailed = load_data()
    print(f"Loaded {len(df)} days of simulation data")
    print(f"  Warmup days: {len(df[df['mode'] == 'warmup'])}")
    print(f"  Hybrid days: {len(df[df['mode'] == 'hybrid'])}")
    print()
    
    # Generate all plots
    print("Generating plots...")
    print()
    
    plot_predictions_vs_actual(df)
    plot_model_comparison(df)
    plot_llm_sentiment(df)
    plot_summary_dashboard(df)
    
    print()
    print("=" * 60)
    print("All visualizations saved!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  1. v4_jan2026_predictions.png - Predictions vs Actual")
    print("  2. v4_jan2026_comparison.png - Model Comparison")
    print("  3. v4_jan2026_llm_analysis.png - LLM Analysis")
    print("  4. v4_jan2026_dashboard.png - Summary Dashboard")


if __name__ == "__main__":
    main()

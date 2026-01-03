"""
Rolling Weekly Monte Carlo Forecast
====================================

Run once, get forecasts for:
- Week 1 (next 7 days)
- Week 2 (days 8-14)
- Week 3 (days 15-21)
- Week 4 (days 22-28)
- And beyond...

Each week builds on the previous week's distribution, showing how uncertainty compounds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)

np.random.seed(42)

print("="*80)
print("ROLLING WEEKLY MONTE CARLO FORECAST")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# How many weeks to forecast
N_WEEKS = 4

# How many simulations per week
N_SIMULATIONS = 10000

# Confidence levels for reporting
CONFIDENCE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]

print(f"\nConfiguration:")
print(f"  Number of weeks to forecast: {N_WEEKS}")
print(f"  Simulations per week: {N_SIMULATIONS:,}")
print(f"  Confidence levels: {[int(c*100) for c in CONFIDENCE_LEVELS]}%")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("[1/5] Loading historical data...")
print("="*80)

# Load exchange rate data
exchange_df = pd.read_csv('../usd_inr_exchange_rates_1year.csv')
exchange_df['Date'] = pd.to_datetime(exchange_df['Date'])

# Load political news merged data
political_df = pd.read_csv('../political_news_exchange_merged.csv')
political_df['Date'] = pd.to_datetime(political_df['Date'])

# Get current state
current_price = exchange_df['USD_to_INR'].iloc[-1]
current_date = exchange_df['Date'].iloc[-1]

print(f"\n  Current date: {current_date.strftime('%Y-%m-%d')}")
print(f"  Current exchange rate: {current_price:.4f} INR")

# Calculate historical parameters
returns = political_df['USD_to_INR'].pct_change().dropna()
log_returns = np.log(political_df['USD_to_INR'] / political_df['USD_to_INR'].shift(1)).dropna()

mean_return = returns.mean()
volatility = log_returns.std()

print(f"  Historical mean return: {mean_return:.6f} ({mean_return*100:.4f}% daily)")
print(f"  Historical volatility: {volatility:.6f} ({volatility*100:.4f}% daily)")

# ============================================================================
# EXTRACT REGIME PARAMETERS
# ============================================================================
print("\n" + "="*80)
print("[2/5] Extracting regime parameters from political news...")
print("="*80)

# Calculate regime-based parameters
political_df['Volatility_7d'] = political_df['USD_to_INR'].pct_change().rolling(7).std()
vol_median = political_df['Volatility_7d'].median()
political_df['High_Volatility'] = (political_df['Volatility_7d'] > vol_median).astype(int)
political_df['Returns'] = political_df['USD_to_INR'].pct_change()

# High/Low volatility parameters
high_vol_returns = political_df[political_df['High_Volatility'] == 1]['Returns'].dropna()
low_vol_returns = political_df[political_df['High_Volatility'] == 0]['Returns'].dropna()

high_vol_mean = high_vol_returns.mean()
high_vol_std = high_vol_returns.std()
low_vol_mean = low_vol_returns.mean()
low_vol_std = low_vol_returns.std()

print(f"\n  High Volatility Regime:")
print(f"    Mean return: {high_vol_mean:.6f}")
print(f"    Volatility: {high_vol_std:.6f}")
print(f"\n  Low Volatility Regime:")
print(f"    Mean return: {low_vol_mean:.6f}")
print(f"    Volatility: {low_vol_std:.6f}")

# Sentiment parameters
recent_sentiment = political_df['GoldsteinScale_mean'].iloc[-7:].mean()
print(f"\n  Recent political sentiment (7-day avg): {recent_sentiment:.4f}")

# Event-based jump parameters
event_75th = political_df['Event_count'].quantile(0.75)
high_event_days = (political_df['Event_count'] > event_75th).sum()
jump_probability = high_event_days / len(political_df)

print(f"\n  Event-driven jumps:")
print(f"    Jump probability: {jump_probability:.4f}")

# Regime transition probabilities
regime_changes = political_df['High_Volatility'].diff().abs().sum()
p_transition = regime_changes / len(political_df)

print(f"\n  Regime transitions:")
print(f"    Probability of regime change: {p_transition:.4f}")

# ============================================================================
# WEEKLY SIMULATION FUNCTION
# ============================================================================

def simulate_one_week_regime_switching(starting_prices, mu_high, sigma_high,
                                       mu_low, sigma_low, p_transition,
                                       days=7, n_sims=10000):
    """
    Simulate one week with regime-switching dynamics

    starting_prices: array of starting prices (from previous week's distribution)
    returns: array of ending prices for this week (one per simulation)
    """

    # If starting_prices is a single value, replicate it
    if isinstance(starting_prices, (int, float)):
        starting_prices = np.full(n_sims, starting_prices)

    # Initialize paths
    paths = np.zeros((n_sims, days + 1))
    paths[:, 0] = starting_prices

    # Initialize regimes (random start for each simulation)
    current_regime = np.random.choice([0, 1], size=n_sims, p=[0.5, 0.5])

    for day in range(1, days + 1):
        # Regime transitions
        transitions = np.random.rand(n_sims) < p_transition
        current_regime = np.where(transitions, 1 - current_regime, current_regime)

        # Generate returns based on regime
        Z = np.random.standard_normal(n_sims)

        for i in range(n_sims):
            if current_regime[i] == 1:  # High volatility
                mu, sigma = mu_high, sigma_high
            else:  # Low volatility
                mu, sigma = mu_low, sigma_low

            paths[i, day] = paths[i, day-1] * np.exp(
                (mu - 0.5 * sigma**2) + sigma * Z[i]
            )

    return paths


def calculate_weekly_statistics(paths, week_num, start_date):
    """
    Calculate statistics for a week's simulation
    """
    final_prices = paths[:, -1]

    stats = {
        'Week': week_num,
        'Start_Date': start_date.strftime('%Y-%m-%d'),
        'End_Date': (start_date + timedelta(days=6)).strftime('%Y-%m-%d'),
        'Mean': np.mean(final_prices),
        'Median': np.median(final_prices),
        'Std': np.std(final_prices),
        '5th_Pct': np.percentile(final_prices, 5),
        '25th_Pct': np.percentile(final_prices, 25),
        '75th_Pct': np.percentile(final_prices, 75),
        '95th_Pct': np.percentile(final_prices, 95),
        'Min': np.min(final_prices),
        'Max': np.max(final_prices)
    }

    return stats, final_prices


# ============================================================================
# RUN ROLLING WEEKLY SIMULATIONS
# ============================================================================
print("\n" + "="*80)
print("[3/5] Running rolling weekly simulations...")
print("="*80)

# Storage for results
all_weekly_stats = []
all_weekly_paths = []  # Store full paths for visualization
all_final_distributions = []  # Store final price distributions

# Initialize
current_starting_prices = current_price
current_sim_date = current_date

for week in range(1, N_WEEKS + 1):
    print(f"\n  Simulating Week {week}...")

    # Run simulation for this week
    week_paths = simulate_one_week_regime_switching(
        starting_prices=current_starting_prices,
        mu_high=high_vol_mean,
        sigma_high=high_vol_std,
        mu_low=low_vol_mean,
        sigma_low=low_vol_std,
        p_transition=p_transition,
        days=7,
        n_sims=N_SIMULATIONS
    )

    # Calculate statistics
    week_start = current_sim_date + timedelta(days=1)
    stats, final_prices = calculate_weekly_statistics(week_paths, week, week_start)

    all_weekly_stats.append(stats)
    all_weekly_paths.append(week_paths)
    all_final_distributions.append(final_prices)

    # Update for next week - use final distribution as starting point
    current_starting_prices = final_prices
    current_sim_date = week_start + timedelta(days=6)

    print(f"    Expected (median): {stats['Median']:.4f} INR")
    print(f"    Range (5th-95th): {stats['5th_Pct']:.4f} to {stats['95th_Pct']:.4f}")

print("\n  All weeks simulated successfully!")

# ============================================================================
# GENERATE SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("[4/5] Generating weekly forecasts...")
print("="*80)

# Create summary dataframe
summary_df = pd.DataFrame(all_weekly_stats)

# Add additional metrics
summary_df['Expected_Change_%'] = ((summary_df['Median'] - current_price) / current_price * 100)
summary_df['Best_Case_Change_%'] = ((summary_df['95th_Pct'] - current_price) / current_price * 100)
summary_df['Worst_Case_Change_%'] = ((summary_df['5th_Pct'] - current_price) / current_price * 100)
summary_df['Uncertainty_Range'] = summary_df['95th_Pct'] - summary_df['5th_Pct']

print("\n" + "="*80)
print("WEEKLY FORECAST SUMMARY")
print("="*80)
print(f"\nCurrent Price: {current_price:.4f} INR")
print(f"Current Date: {current_date.strftime('%Y-%m-%d')}\n")

# Display summary
display_cols = ['Week', 'Start_Date', 'End_Date', 'Median', '5th_Pct', '95th_Pct', 'Expected_Change_%']
print(summary_df[display_cols].to_string(index=False))

# Calculate probabilities for each week
print("\n" + "="*80)
print("WEEKLY PROBABILITIES")
print("="*80)

for week in range(1, N_WEEKS + 1):
    final_dist = all_final_distributions[week - 1]

    prob_increase = (final_dist > current_price).sum() / N_SIMULATIONS * 100
    prob_large_increase = (final_dist > current_price + 1.0).sum() / N_SIMULATIONS * 100
    prob_large_decrease = (final_dist < current_price - 1.0).sum() / N_SIMULATIONS * 100

    print(f"\nWeek {week} ({summary_df.iloc[week-1]['Start_Date']} to {summary_df.iloc[week-1]['End_Date']}):")
    print(f"  Probability of increase (> {current_price:.2f}): {prob_increase:.1f}%")
    print(f"  Probability of large increase (> {current_price+1:.2f}): {prob_large_increase:.1f}%")
    print(f"  Probability of large decrease (< {current_price-1:.2f}): {prob_large_decrease:.1f}%")

# Risk metrics
print("\n" + "="*80)
print("WEEKLY VALUE AT RISK (VaR)")
print("="*80)

for week in range(1, N_WEEKS + 1):
    final_dist = all_final_distributions[week - 1]
    returns = (final_dist - current_price) / current_price

    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)

    print(f"\nWeek {week}:")
    print(f"  VaR 95%: {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"  VaR 99%: {var_99:.4f} ({var_99*100:.2f}%)")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("[5/5] Creating visualizations...")
print("="*80)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# Color scheme for weeks
week_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

# 1. Individual weekly paths (all weeks overlaid)
ax1 = fig.add_subplot(gs[0, :2])
cumulative_days = 0

for week_idx in range(N_WEEKS):
    week_paths = all_weekly_paths[week_idx]
    sample_size = 50
    sample_indices = np.random.choice(N_SIMULATIONS, sample_size, replace=False)

    days_in_week = np.arange(7 + 1) + cumulative_days

    for i in sample_indices:
        ax1.plot(days_in_week, week_paths[i, :],
                alpha=0.15, color=week_colors[week_idx], linewidth=0.5)

    # Plot mean for this week
    mean_path = np.mean(week_paths, axis=0)
    ax1.plot(days_in_week, mean_path,
            color=week_colors[week_idx], linewidth=2.5,
            label=f'Week {week_idx+1} Mean', alpha=0.9)

    cumulative_days += 7

ax1.axhline(y=current_price, color='black', linestyle='--', linewidth=2,
           label=f'Current: {current_price:.2f}', alpha=0.7)
ax1.set_xlabel('Days from Today', fontweight='bold', fontsize=11)
ax1.set_ylabel('Exchange Rate (INR)', fontweight='bold', fontsize=11)
ax1.set_title('Rolling Weekly Simulations (50 paths per week)', fontweight='bold', fontsize=13)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Weekly distribution comparison
ax2 = fig.add_subplot(gs[0, 2])
positions = np.arange(1, N_WEEKS + 1)
box_data = [all_final_distributions[i] for i in range(N_WEEKS)]

bp = ax2.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                 showfliers=False)

for patch, color in zip(bp['boxes'], week_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.axhline(y=current_price, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_xlabel('Week', fontweight='bold', fontsize=11)
ax2.set_ylabel('Exchange Rate (INR)', fontweight='bold', fontsize=11)
ax2.set_title('Weekly Distribution Comparison', fontweight='bold', fontsize=13)
ax2.set_xticks(positions)
ax2.set_xticklabels([f'W{i}' for i in positions])
ax2.grid(True, alpha=0.3, axis='y')

# 3. Confidence intervals over time
ax3 = fig.add_subplot(gs[1, :])

# Collect statistics over all days
all_days = []
medians = []
p5s = []
p25s = []
p75s = []
p95s = []

cumulative_days = 0
for week_idx in range(N_WEEKS):
    week_paths = all_weekly_paths[week_idx]

    for day in range(8):  # 0 to 7
        current_day = cumulative_days + day
        all_days.append(current_day)

        prices_at_day = week_paths[:, day]
        medians.append(np.median(prices_at_day))
        p5s.append(np.percentile(prices_at_day, 5))
        p25s.append(np.percentile(prices_at_day, 25))
        p75s.append(np.percentile(prices_at_day, 75))
        p95s.append(np.percentile(prices_at_day, 95))

    cumulative_days += 7

ax3.plot(all_days, medians, color='red', linewidth=3, label='Median Forecast', zorder=5)
ax3.fill_between(all_days, p5s, p95s, alpha=0.2, color='blue', label='90% Confidence')
ax3.fill_between(all_days, p25s, p75s, alpha=0.3, color='blue', label='50% Confidence')
ax3.axhline(y=current_price, color='black', linestyle='--', linewidth=2, label=f'Current: {current_price:.2f}')

# Add week separators
for week in range(1, N_WEEKS):
    ax3.axvline(x=week*7, color='gray', linestyle=':', alpha=0.5)

ax3.set_xlabel('Days from Today', fontweight='bold', fontsize=11)
ax3.set_ylabel('Exchange Rate (INR)', fontweight='bold', fontsize=11)
ax3.set_title('Forecast with Confidence Intervals (4-Week Horizon)', fontweight='bold', fontsize=13)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Weekly expected values with error bars
ax4 = fig.add_subplot(gs[2, 0])
weeks = np.arange(1, N_WEEKS + 1)
medians_weekly = summary_df['Median'].values
lower_errors = summary_df['Median'].values - summary_df['5th_Pct'].values
upper_errors = summary_df['95th_Pct'].values - summary_df['Median'].values

ax4.errorbar(weeks, medians_weekly, yerr=[lower_errors, upper_errors],
            fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
            color='#3498db', ecolor='#e74c3c', alpha=0.8)
ax4.axhline(y=current_price, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('Week', fontweight='bold', fontsize=11)
ax4.set_ylabel('Exchange Rate (INR)', fontweight='bold', fontsize=11)
ax4.set_title('Weekly Median ± 90% CI', fontweight='bold', fontsize=13)
ax4.set_xticks(weeks)
ax4.grid(True, alpha=0.3)

# 5. Uncertainty growth over weeks
ax5 = fig.add_subplot(gs[2, 1])
uncertainty = summary_df['Uncertainty_Range'].values
ax5.bar(weeks, uncertainty, color=week_colors, alpha=0.7, edgecolor='black')
ax5.set_xlabel('Week', fontweight='bold', fontsize=11)
ax5.set_ylabel('Uncertainty Range (INR)', fontweight='bold', fontsize=11)
ax5.set_title('Uncertainty Growth (95th - 5th percentile)', fontweight='bold', fontsize=13)
ax5.set_xticks(weeks)
ax5.grid(True, alpha=0.3, axis='y')

# Add values on bars
for i, (w, u) in enumerate(zip(weeks, uncertainty)):
    ax5.text(w, u, f'{u:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 6. Probability of increase by week
ax6 = fig.add_subplot(gs[2, 2])
probs_increase = []
for week_idx in range(N_WEEKS):
    final_dist = all_final_distributions[week_idx]
    prob = (final_dist > current_price).sum() / N_SIMULATIONS * 100
    probs_increase.append(prob)

ax6.bar(weeks, probs_increase, color=week_colors, alpha=0.7, edgecolor='black')
ax6.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.5, label='50% (neutral)')
ax6.set_xlabel('Week', fontweight='bold', fontsize=11)
ax6.set_ylabel('Probability (%)', fontweight='bold', fontsize=11)
ax6.set_title(f'Probability of Increase (> {current_price:.2f})', fontweight='bold', fontsize=13)
ax6.set_xticks(weeks)
ax6.set_ylim([0, 100])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

for i, (w, p) in enumerate(zip(weeks, probs_increase)):
    ax6.text(w, p + 2, f'{p:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 7-9. Individual week distributions
for week_idx in range(min(3, N_WEEKS)):
    ax = fig.add_subplot(gs[3, week_idx])

    final_dist = all_final_distributions[week_idx]

    ax.hist(final_dist, bins=50, color=week_colors[week_idx], alpha=0.7, edgecolor='black')
    ax.axvline(x=current_price, color='black', linestyle='--', linewidth=2,
              label=f'Current: {current_price:.2f}')
    ax.axvline(x=summary_df.iloc[week_idx]['Median'], color='red', linewidth=2,
              label=f"Median: {summary_df.iloc[week_idx]['Median']:.2f}")
    ax.axvline(x=summary_df.iloc[week_idx]['5th_Pct'], color='orange',
              linestyle=':', linewidth=2, label='5th-95th')
    ax.axvline(x=summary_df.iloc[week_idx]['95th_Pct'], color='orange',
              linestyle=':', linewidth=2)

    ax.set_xlabel('Exchange Rate (INR)', fontweight='bold', fontsize=10)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=10)
    ax.set_title(f'Week {week_idx+1} Distribution\n({summary_df.iloc[week_idx]["Start_Date"]} to {summary_df.iloc[week_idx]["End_Date"]})',
                fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Rolling Weekly Monte Carlo Forecast - 4 Week Horizon',
            fontsize=16, fontweight='bold', y=0.998)

plt.savefig('weekly_rolling_forecast.png', dpi=300, bbox_inches='tight')
print("\n  Saved: weekly_rolling_forecast.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("Saving results...")
print("="*80)

# Save summary statistics
summary_df.to_csv('weekly_forecast_summary.csv', index=False)
print("  Saved: weekly_forecast_summary.csv")

# Create detailed weekly breakdowns
for week in range(1, N_WEEKS + 1):
    final_dist = all_final_distributions[week - 1]

    week_detail = pd.DataFrame({
        'Simulation': range(1, N_SIMULATIONS + 1),
        'Final_Price': final_dist,
        'Return_%': ((final_dist - current_price) / current_price * 100)
    })

    week_detail.to_csv(f'week_{week}_detailed_results.csv', index=False)

print(f"  Saved: week_1 to week_{N_WEEKS} detailed results")

# Create actionable weekly report
with open('weekly_forecast_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ROLLING WEEKLY MONTE CARLO FORECAST REPORT\n")
    f.write("="*80 + "\n\n")

    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Current Price: {current_price:.4f} INR\n")
    f.write(f"Current Date: {current_date.strftime('%Y-%m-%d')}\n")
    f.write(f"Forecast Horizon: {N_WEEKS} weeks\n")
    f.write(f"Simulations: {N_SIMULATIONS:,}\n\n")

    f.write("="*80 + "\n")
    f.write("WEEKLY FORECAST SUMMARY\n")
    f.write("="*80 + "\n\n")

    for week in range(1, N_WEEKS + 1):
        stats = summary_df.iloc[week - 1]
        final_dist = all_final_distributions[week - 1]

        prob_increase = (final_dist > current_price).sum() / N_SIMULATIONS * 100
        returns = (final_dist - current_price) / current_price
        var_95 = np.percentile(returns, 5)

        f.write(f"WEEK {week}: {stats['Start_Date']} to {stats['End_Date']}\n")
        f.write("-"*80 + "\n")
        f.write(f"Expected (Median):     {stats['Median']:.4f} INR ({stats['Expected_Change_%']:+.2f}%)\n")
        f.write(f"Best Case (95th):      {stats['95th_Pct']:.4f} INR ({stats['Best_Case_Change_%']:+.2f}%)\n")
        f.write(f"Worst Case (5th):      {stats['5th_Pct']:.4f} INR ({stats['Worst_Case_Change_%']:+.2f}%)\n")
        f.write(f"Uncertainty Range:     {stats['Uncertainty_Range']:.4f} INR\n")
        f.write(f"\nProbabilities:\n")
        f.write(f"  Increase (> {current_price:.2f}): {prob_increase:.1f}%\n")
        f.write(f"  Value at Risk (95%): {var_95*100:.2f}%\n")
        f.write("\n")

    f.write("="*80 + "\n")
    f.write("HOW TO USE THIS FORECAST\n")
    f.write("="*80 + "\n\n")

    f.write("Week 1 Forecast: Most reliable, use for immediate planning\n")
    f.write("Week 2-3 Forecast: Moderate uncertainty, good for medium-term planning\n")
    f.write("Week 4+ Forecast: High uncertainty, use for scenario planning only\n\n")

    f.write("Each week shows:\n")
    f.write("- Expected (median): Most likely outcome\n")
    f.write("- Best/Worst case: 90% confidence interval\n")
    f.write("- Probability of increase: Likelihood rate goes up\n")
    f.write("- Value at Risk: Maximum likely loss (95% confidence)\n\n")

    f.write("Note: Uncertainty increases with time. Later weeks have wider ranges.\n")

print("  Saved: weekly_forecast_report.txt")

print("\n" + "="*80)
print("WEEKLY ROLLING FORECAST COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. weekly_rolling_forecast.png (comprehensive visualization)")
print("  2. weekly_forecast_summary.csv (summary statistics)")
print("  3. week_1 to week_4_detailed_results.csv (detailed simulations)")
print("  4. weekly_forecast_report.txt (actionable report)")
print("\n" + "="*80)
print("\nQUICK REFERENCE:")
print("-"*80)

for week in range(1, N_WEEKS + 1):
    stats = summary_df.iloc[week - 1]
    print(f"\nWeek {week} ({stats['Start_Date']} to {stats['End_Date']}):")
    print(f"  Expected: {stats['Median']:.2f} INR ({stats['Expected_Change_%']:+.1f}%)")
    print(f"  Range: {stats['5th_Pct']:.2f} to {stats['95th_Pct']:.2f} INR")

print("\n" + "="*80)

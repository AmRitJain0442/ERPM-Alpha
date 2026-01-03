"""
Monte Carlo Simulation for USD/INR Exchange Rate
=================================================

This simulation incorporates meaningful patterns discovered from political news:
1. Regime-based volatility
2. Sentiment-driven drift
3. Event-driven jumps
4. Lead-lag effects

Models used:
- Geometric Brownian Motion (GBM)
- Regime-Switching Model
- Jump Diffusion Model
- Sentiment-Augmented Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

np.random.seed(42)

print("="*80)
print("MONTE CARLO SIMULATION FOR USD/INR EXCHANGE RATE")
print("="*80)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/6] Loading historical data...")

# Load exchange rate data
exchange_df = pd.read_csv('../usd_inr_exchange_rates_1year.csv')
exchange_df['Date'] = pd.to_datetime(exchange_df['Date'])

# Load political news merged data
political_df = pd.read_csv('../political_news_exchange_merged.csv')
political_df['Date'] = pd.to_datetime(political_df['Date'])

print(f"  Exchange rate data: {len(exchange_df)} observations")
print(f"  Political news data: {len(political_df)} observations")

# Calculate historical parameters
returns = exchange_df['USD_to_INR'].pct_change().dropna()
log_returns = np.log(exchange_df['USD_to_INR'] / exchange_df['USD_to_INR'].shift(1)).dropna()

current_price = exchange_df['USD_to_INR'].iloc[-1]
mean_return = returns.mean()
std_return = returns.std()
volatility = log_returns.std()

print(f"\n  Current exchange rate: {current_price:.4f} INR")
print(f"  Historical mean return: {mean_return:.6f} ({mean_return*100:.4f}%)")
print(f"  Historical volatility: {volatility:.6f} ({volatility*100:.4f}%)")

# ============================================================================
# PATTERN INTEGRATION: Calculate regime-based parameters
# ============================================================================
print("\n[2/6] Extracting patterns from political news...")

# Calculate rolling volatility
political_df['Volatility_7d'] = political_df['USD_to_INR'].pct_change().rolling(7).std()

# Define regimes based on volatility
vol_median = political_df['Volatility_7d'].median()
political_df['High_Volatility'] = (political_df['Volatility_7d'] > vol_median).astype(int)

# Calculate returns by regime
political_df['Returns'] = political_df['USD_to_INR'].pct_change()

high_vol_returns = political_df[political_df['High_Volatility'] == 1]['Returns'].dropna()
low_vol_returns = political_df[political_df['High_Volatility'] == 0]['Returns'].dropna()

high_vol_std = high_vol_returns.std()
low_vol_std = low_vol_returns.std()
high_vol_mean = high_vol_returns.mean()
low_vol_mean = low_vol_returns.mean()

print(f"\n  Regime Parameters:")
print(f"    High Volatility: mean={high_vol_mean:.6f}, std={high_vol_std:.6f}")
print(f"    Low Volatility:  mean={low_vol_mean:.6f}, std={low_vol_std:.6f}")

# Sentiment effect on drift
goldstein_median = political_df['GoldsteinScale_mean'].median()
positive_sentiment = political_df[political_df['GoldsteinScale_mean'] > goldstein_median]
negative_sentiment = political_df[political_df['GoldsteinScale_mean'] <= goldstein_median]

pos_sent_return = positive_sentiment['Returns'].mean()
neg_sent_return = negative_sentiment['Returns'].mean()

print(f"\n  Sentiment Effect:")
print(f"    Positive sentiment drift: {pos_sent_return:.6f}")
print(f"    Negative sentiment drift: {neg_sent_return:.6f}")

# Event intensity (for jump frequency)
event_75th = political_df['Event_count'].quantile(0.75)
high_event_days = (political_df['Event_count'] > event_75th).sum()
jump_probability = high_event_days / len(political_df)

print(f"\n  Jump Parameters:")
print(f"    High event days: {high_event_days}/{len(political_df)}")
print(f"    Jump probability: {jump_probability:.4f}")

# ============================================================================
# MONTE CARLO SIMULATION FUNCTIONS
# ============================================================================

def geometric_brownian_motion(S0, mu, sigma, T, dt, n_simulations):
    """
    Standard GBM: dS = mu*S*dt + sigma*S*dW

    S0: Initial price
    mu: Drift (mean return)
    sigma: Volatility
    T: Time horizon (days)
    dt: Time step (1 day)
    n_simulations: Number of paths
    """
    n_steps = int(T / dt)

    # Pre-allocate array
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0

    # Generate random shocks
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_simulations)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return paths


def regime_switching_simulation(S0, mu_high, sigma_high, mu_low, sigma_low,
                                 p_high_to_low, p_low_to_high, T, dt, n_simulations):
    """
    Regime-switching model with high/low volatility states
    """
    n_steps = int(T / dt)
    paths = np.zeros((n_simulations, n_steps + 1))
    regimes = np.zeros((n_simulations, n_steps + 1))

    paths[:, 0] = S0
    regimes[:, 0] = np.random.choice([0, 1], size=n_simulations, p=[0.5, 0.5])

    for t in range(1, n_steps + 1):
        # Transition between regimes
        for i in range(n_simulations):
            if regimes[i, t-1] == 1:  # High volatility
                regimes[i, t] = 1 if np.random.rand() > p_high_to_low else 0
            else:  # Low volatility
                regimes[i, t] = 1 if np.random.rand() > (1 - p_low_to_high) else 0

        # Generate returns based on regime
        Z = np.random.standard_normal(n_simulations)

        for i in range(n_simulations):
            if regimes[i, t] == 1:  # High volatility
                mu, sigma = mu_high, sigma_high
            else:  # Low volatility
                mu, sigma = mu_low, sigma_low

            paths[i, t] = paths[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i])

    return paths, regimes


def jump_diffusion_simulation(S0, mu, sigma, jump_prob, jump_mean, jump_std, T, dt, n_simulations):
    """
    Jump diffusion model (Merton): dS = mu*S*dt + sigma*S*dW + J*S*dN

    Adds random jumps to GBM to model sudden events
    """
    n_steps = int(T / dt)
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        # Continuous component (GBM)
        Z = np.random.standard_normal(n_simulations)
        continuous_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

        # Jump component
        jumps = np.random.binomial(1, jump_prob, n_simulations)
        jump_sizes = np.random.normal(jump_mean, jump_std, n_simulations)
        jump_return = jumps * jump_sizes

        # Combined
        paths[:, t] = paths[:, t-1] * np.exp(continuous_return + jump_return)

    return paths


def sentiment_augmented_simulation(S0, base_mu, base_sigma, sentiment_score,
                                   sentiment_impact, T, dt, n_simulations):
    """
    GBM with sentiment-adjusted drift

    mu = base_mu + sentiment_impact * sentiment_score
    """
    # Adjust drift based on sentiment
    adjusted_mu = base_mu + sentiment_impact * sentiment_score

    return geometric_brownian_motion(S0, adjusted_mu, base_sigma, T, dt, n_simulations)


# ============================================================================
# RUN SIMULATIONS
# ============================================================================
print("\n[3/6] Running Monte Carlo simulations...")

# Simulation parameters
T = 30  # 30 days forecast
dt = 1  # Daily steps
n_simulations = 10000

print(f"  Time horizon: {T} days")
print(f"  Number of simulations: {n_simulations:,}")

# 1. Standard GBM
print("\n  [1/4] Running standard GBM...")
gbm_paths = geometric_brownian_motion(
    S0=current_price,
    mu=mean_return,
    sigma=volatility,
    T=T,
    dt=dt,
    n_simulations=n_simulations
)

# 2. Regime-switching model
print("  [2/4] Running regime-switching model...")
p_high_to_low = 0.15  # 15% chance of switching from high to low vol
p_low_to_high = 0.10  # 10% chance of switching from low to high vol

regime_paths, regime_states = regime_switching_simulation(
    S0=current_price,
    mu_high=high_vol_mean,
    sigma_high=high_vol_std,
    mu_low=low_vol_mean,
    sigma_low=low_vol_std,
    p_high_to_low=p_high_to_low,
    p_low_to_high=p_low_to_high,
    T=T,
    dt=dt,
    n_simulations=n_simulations
)

# 3. Jump diffusion
print("  [3/4] Running jump diffusion model...")
jump_mean = -0.005  # Average jump size (slightly negative)
jump_std = 0.015    # Jump volatility

jump_paths = jump_diffusion_simulation(
    S0=current_price,
    mu=mean_return,
    sigma=volatility,
    jump_prob=jump_probability,
    jump_mean=jump_mean,
    jump_std=jump_std,
    T=T,
    dt=dt,
    n_simulations=n_simulations
)

# 4. Sentiment-augmented (using recent sentiment)
print("  [4/4] Running sentiment-augmented model...")
recent_sentiment = political_df['GoldsteinScale_mean'].iloc[-7:].mean()  # Last 7 days avg
sentiment_impact = 0.0005  # How much sentiment affects drift

sentiment_paths = sentiment_augmented_simulation(
    S0=current_price,
    base_mu=mean_return,
    base_sigma=volatility,
    sentiment_score=recent_sentiment,
    sentiment_impact=sentiment_impact,
    T=T,
    dt=dt,
    n_simulations=n_simulations
)

print("  All simulations complete!")

# ============================================================================
# CALCULATE STATISTICS
# ============================================================================
print("\n[4/6] Calculating statistics...")

def calculate_statistics(paths, model_name):
    """Calculate key statistics for simulation paths"""
    final_prices = paths[:, -1]

    stats_dict = {
        'Model': model_name,
        'Mean': np.mean(final_prices),
        'Median': np.median(final_prices),
        'Std': np.std(final_prices),
        '5th Percentile': np.percentile(final_prices, 5),
        '25th Percentile': np.percentile(final_prices, 25),
        '75th Percentile': np.percentile(final_prices, 75),
        '95th Percentile': np.percentile(final_prices, 95),
        'Min': np.min(final_prices),
        'Max': np.max(final_prices)
    }

    return stats_dict

models_stats = [
    calculate_statistics(gbm_paths, 'Standard GBM'),
    calculate_statistics(regime_paths, 'Regime-Switching'),
    calculate_statistics(jump_paths, 'Jump Diffusion'),
    calculate_statistics(sentiment_paths, 'Sentiment-Augmented')
]

stats_df = pd.DataFrame(models_stats)

print("\n" + "="*80)
print("SIMULATION RESULTS (30-day forecast)")
print("="*80)
print(stats_df.to_string(index=False))

# Value at Risk (VaR) analysis
print("\n" + "="*80)
print("RISK ANALYSIS - VALUE AT RISK (VaR)")
print("="*80)

def calculate_var(paths, confidence_levels=[0.95, 0.99]):
    """Calculate Value at Risk"""
    final_prices = paths[:, -1]
    returns = (final_prices - current_price) / current_price

    var_results = {}
    for conf in confidence_levels:
        var_results[f'VaR_{int(conf*100)}%'] = np.percentile(returns, (1-conf)*100)

    return var_results

print(f"\nCurrent Price: {current_price:.4f} INR")
print(f"\n{'Model':<25} {'VaR 95%':<15} {'VaR 99%':<15}")
print("-"*55)

for name, paths in [('Standard GBM', gbm_paths),
                     ('Regime-Switching', regime_paths),
                     ('Jump Diffusion', jump_paths),
                     ('Sentiment-Augmented', sentiment_paths)]:
    var = calculate_var(paths)
    print(f"{name:<25} {var['VaR_95%']:<15.4f} {var['VaR_99%']:<15.4f}")

# ============================================================================
# SCENARIO ANALYSIS
# ============================================================================
print("\n[5/6] Generating scenario analysis...")

# Best, worst, and expected scenarios
best_case = np.percentile(regime_paths[:, -1], 95)
worst_case = np.percentile(regime_paths[:, -1], 5)
expected_case = np.median(regime_paths[:, -1])

print("\n" + "="*80)
print("SCENARIO ANALYSIS (Regime-Switching Model)")
print("="*80)
print(f"\nCurrent Rate: {current_price:.4f} INR")
print(f"\nAfter 30 days:")
print(f"  Best Case (95th percentile):     {best_case:.4f} INR ({((best_case-current_price)/current_price*100):+.2f}%)")
print(f"  Expected Case (median):          {expected_case:.4f} INR ({((expected_case-current_price)/current_price*100):+.2f}%)")
print(f"  Worst Case (5th percentile):     {worst_case:.4f} INR ({((worst_case-current_price)/current_price*100):+.2f}%)")

# Probability of different outcomes
prob_increase = (regime_paths[:, -1] > current_price).sum() / n_simulations * 100
prob_decrease = (regime_paths[:, -1] < current_price).sum() / n_simulations * 100
prob_large_move = (np.abs(regime_paths[:, -1] - current_price) > 1.0).sum() / n_simulations * 100

print(f"\n" + "-"*80)
print("PROBABILITIES:")
print("-"*80)
print(f"  Probability of increase (rate > {current_price:.2f}): {prob_increase:.2f}%")
print(f"  Probability of decrease (rate < {current_price:.2f}): {prob_decrease:.2f}%")
print(f"  Probability of large move (>1 INR):  {prob_large_move:.2f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[6/6] Creating visualizations...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. Standard GBM paths
ax1 = fig.add_subplot(gs[0, 0])
sample_paths = np.random.choice(n_simulations, 100, replace=False)
for i in sample_paths:
    ax1.plot(gbm_paths[i, :], alpha=0.1, color='blue', linewidth=0.5)
ax1.plot(np.mean(gbm_paths, axis=0), color='red', linewidth=2, label='Mean path')
ax1.axhline(y=current_price, color='black', linestyle='--', label=f'Current: {current_price:.2f}')
ax1.set_xlabel('Days', fontweight='bold')
ax1.set_ylabel('Exchange Rate (INR)', fontweight='bold')
ax1.set_title('Standard GBM Simulation\n(100 sample paths)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Regime-switching paths
ax2 = fig.add_subplot(gs[0, 1])
for i in sample_paths:
    ax2.plot(regime_paths[i, :], alpha=0.1, color='green', linewidth=0.5)
ax2.plot(np.mean(regime_paths, axis=0), color='red', linewidth=2, label='Mean path')
ax2.axhline(y=current_price, color='black', linestyle='--', label=f'Current: {current_price:.2f}')
ax2.set_xlabel('Days', fontweight='bold')
ax2.set_ylabel('Exchange Rate (INR)', fontweight='bold')
ax2.set_title('Regime-Switching Simulation\n(100 sample paths)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Jump diffusion paths
ax3 = fig.add_subplot(gs[0, 2])
for i in sample_paths:
    ax3.plot(jump_paths[i, :], alpha=0.1, color='orange', linewidth=0.5)
ax3.plot(np.mean(jump_paths, axis=0), color='red', linewidth=2, label='Mean path')
ax3.axhline(y=current_price, color='black', linestyle='--', label=f'Current: {current_price:.2f}')
ax3.set_xlabel('Days', fontweight='bold')
ax3.set_ylabel('Exchange Rate (INR)', fontweight='bold')
ax3.set_title('Jump Diffusion Simulation\n(100 sample paths)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Distribution comparison (final prices)
ax4 = fig.add_subplot(gs[1, :])
ax4.hist(gbm_paths[:, -1], bins=50, alpha=0.5, label='Standard GBM', color='blue', density=True)
ax4.hist(regime_paths[:, -1], bins=50, alpha=0.5, label='Regime-Switching', color='green', density=True)
ax4.hist(jump_paths[:, -1], bins=50, alpha=0.5, label='Jump Diffusion', color='orange', density=True)
ax4.hist(sentiment_paths[:, -1], bins=50, alpha=0.5, label='Sentiment-Augmented', color='purple', density=True)
ax4.axvline(x=current_price, color='black', linestyle='--', linewidth=2, label=f'Current: {current_price:.2f}')
ax4.set_xlabel('Exchange Rate (INR)', fontweight='bold')
ax4.set_ylabel('Probability Density', fontweight='bold')
ax4.set_title('Distribution of Final Prices (30 days)', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Confidence intervals (regime-switching)
ax5 = fig.add_subplot(gs[2, :2])
time_steps = np.arange(0, T+1)
mean_path = np.mean(regime_paths, axis=0)
std_path = np.std(regime_paths, axis=0)
p5 = np.percentile(regime_paths, 5, axis=0)
p25 = np.percentile(regime_paths, 25, axis=0)
p75 = np.percentile(regime_paths, 75, axis=0)
p95 = np.percentile(regime_paths, 95, axis=0)

ax5.plot(time_steps, mean_path, color='red', linewidth=2, label='Mean')
ax5.fill_between(time_steps, p5, p95, alpha=0.2, color='blue', label='90% CI')
ax5.fill_between(time_steps, p25, p75, alpha=0.3, color='blue', label='50% CI')
ax5.axhline(y=current_price, color='black', linestyle='--', linewidth=2, label=f'Current: {current_price:.2f}')
ax5.set_xlabel('Days', fontweight='bold')
ax5.set_ylabel('Exchange Rate (INR)', fontweight='bold')
ax5.set_title('Regime-Switching Model: Confidence Intervals', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Fan chart
ax6 = fig.add_subplot(gs[2, 2])
percentiles = [5, 10, 25, 50, 75, 90, 95]
colors_fan = plt.cm.Blues(np.linspace(0.3, 0.9, len(percentiles)//2))

for i in range(len(percentiles)//2):
    lower = np.percentile(regime_paths, percentiles[i], axis=0)
    upper = np.percentile(regime_paths, percentiles[-(i+1)], axis=0)
    ax6.fill_between(time_steps, lower, upper, alpha=0.4, color=colors_fan[i])

ax6.plot(time_steps, np.median(regime_paths, axis=0), color='red', linewidth=2, label='Median')
ax6.axhline(y=current_price, color='black', linestyle='--', linewidth=2)
ax6.set_xlabel('Days', fontweight='bold')
ax6.set_ylabel('Exchange Rate (INR)', fontweight='bold')
ax6.set_title('Fan Chart (Prediction Uncertainty)', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Scenario comparison
ax7 = fig.add_subplot(gs[3, 0])
scenarios = ['Best\n(95th)', 'Expected\n(median)', 'Worst\n(5th)']
values = [best_case, expected_case, worst_case]
colors_scenario = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax7.bar(scenarios, values, color=colors_scenario, alpha=0.7)
ax7.axhline(y=current_price, color='black', linestyle='--', linewidth=2, label=f'Current: {current_price:.2f}')
ax7.set_ylabel('Exchange Rate (INR)', fontweight='bold')
ax7.set_title('30-Day Scenarios', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    height = bar.get_height()
    change = ((val - current_price) / current_price) * 100
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}\n({change:+.1f}%)',
            ha='center', va='bottom', fontweight='bold')

# 8. Return distribution
ax8 = fig.add_subplot(gs[3, 1])
returns_dist = (regime_paths[:, -1] - current_price) / current_price * 100
ax8.hist(returns_dist, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
ax8.axvline(x=np.percentile(returns_dist, 5), color='orange', linestyle='--', linewidth=2, label='5th percentile')
ax8.axvline(x=np.percentile(returns_dist, 95), color='green', linestyle='--', linewidth=2, label='95th percentile')
ax8.set_xlabel('Return (%)', fontweight='bold')
ax8.set_ylabel('Frequency', fontweight='bold')
ax8.set_title('30-Day Return Distribution', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Probability timeline
ax9 = fig.add_subplot(gs[3, 2])
prob_above_current = []
for t in range(T+1):
    prob = (regime_paths[:, t] > current_price).sum() / n_simulations * 100
    prob_above_current.append(prob)

ax9.plot(time_steps, prob_above_current, color='#2ecc71', linewidth=2, marker='o', markersize=4)
ax9.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% probability')
ax9.fill_between(time_steps, 0, prob_above_current, alpha=0.3, color='#2ecc71')
ax9.set_xlabel('Days', fontweight='bold')
ax9.set_ylabel('Probability (%)', fontweight='bold')
ax9.set_title(f'Probability of Rate > {current_price:.2f}', fontweight='bold')
ax9.set_ylim([0, 100])
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.suptitle('Monte Carlo Simulation: USD/INR Exchange Rate (30-day forecast)',
            fontsize=18, fontweight='bold', y=0.998)

plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
print("\nSaved: monte_carlo_results.png")

# ============================================================================
# SAVE DETAILED REPORT
# ============================================================================
print("\nSaving detailed report...")

# Save statistics to CSV
stats_df.to_csv('monte_carlo_statistics.csv', index=False)
print("Saved: monte_carlo_statistics.csv")

# Save sample paths
sample_results = pd.DataFrame({
    'Day': np.arange(T+1),
    'Mean': np.mean(regime_paths, axis=0),
    'Median': np.median(regime_paths, axis=0),
    '5th_Percentile': np.percentile(regime_paths, 5, axis=0),
    '25th_Percentile': np.percentile(regime_paths, 25, axis=0),
    '75th_Percentile': np.percentile(regime_paths, 75, axis=0),
    '95th_Percentile': np.percentile(regime_paths, 95, axis=0)
})
sample_results.to_csv('monte_carlo_forecast.csv', index=False)
print("Saved: monte_carlo_forecast.csv")

# Comprehensive text report
with open('monte_carlo_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("MONTE CARLO SIMULATION REPORT\n")
    f.write("USD/INR Exchange Rate Forecast (30 days)\n")
    f.write("="*80 + "\n\n")

    f.write("SIMULATION PARAMETERS\n")
    f.write("-"*80 + "\n")
    f.write(f"Current Exchange Rate: {current_price:.4f} INR\n")
    f.write(f"Number of Simulations: {n_simulations:,}\n")
    f.write(f"Time Horizon: {T} days\n")
    f.write(f"Historical Volatility: {volatility:.6f} ({volatility*100:.4f}% daily)\n")
    f.write(f"Historical Mean Return: {mean_return:.6f} ({mean_return*100:.4f}% daily)\n\n")

    f.write("PATTERNS INCORPORATED\n")
    f.write("-"*80 + "\n")
    f.write(f"1. Regime Switching:\n")
    f.write(f"   - High volatility: {high_vol_std:.6f}\n")
    f.write(f"   - Low volatility: {low_vol_std:.6f}\n")
    f.write(f"2. Political Sentiment:\n")
    f.write(f"   - Recent 7-day avg: {recent_sentiment:.4f}\n")
    f.write(f"   - Sentiment impact on drift: {sentiment_impact:.6f}\n")
    f.write(f"3. Jump Events:\n")
    f.write(f"   - Jump probability: {jump_probability:.4f}\n")
    f.write(f"   - Jump mean: {jump_mean:.6f}\n\n")

    f.write("30-DAY FORECAST (Regime-Switching Model)\n")
    f.write("-"*80 + "\n")
    f.write(f"Expected (Median):     {expected_case:.4f} INR ({((expected_case-current_price)/current_price*100):+.2f}%)\n")
    f.write(f"Best Case (95th):      {best_case:.4f} INR ({((best_case-current_price)/current_price*100):+.2f}%)\n")
    f.write(f"Worst Case (5th):      {worst_case:.4f} INR ({((worst_case-current_price)/current_price*100):+.2f}%)\n\n")

    f.write("PROBABILITIES\n")
    f.write("-"*80 + "\n")
    f.write(f"Probability of increase: {prob_increase:.2f}%\n")
    f.write(f"Probability of decrease: {prob_decrease:.2f}%\n")
    f.write(f"Probability of large move (>1 INR): {prob_large_move:.2f}%\n\n")

    f.write("MODEL COMPARISON\n")
    f.write("-"*80 + "\n")
    f.write(stats_df.to_string(index=False))
    f.write("\n\n")

    f.write("INTERPRETATION\n")
    f.write("-"*80 + "\n")
    f.write("This Monte Carlo simulation provides probabilistic forecasts by:\n")
    f.write("1. Running 10,000 different scenarios\n")
    f.write("2. Incorporating volatility regimes from political news patterns\n")
    f.write("3. Including sentiment-driven drift adjustments\n")
    f.write("4. Modeling jump events during high political activity\n\n")
    f.write("The regime-switching model is recommended as it captures\n")
    f.write("the varying volatility states observed in the political news data.\n")

print("Saved: monte_carlo_report.txt")

print("\n" + "="*80)
print("MONTE CARLO SIMULATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. monte_carlo_results.png (comprehensive visualization)")
print("  2. monte_carlo_statistics.csv (summary statistics)")
print("  3. monte_carlo_forecast.csv (daily forecast with confidence intervals)")
print("  4. monte_carlo_report.txt (detailed interpretation)")
print("\n" + "="*80)

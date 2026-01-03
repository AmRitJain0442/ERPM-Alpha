import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

print("=" * 80)
print("STEP 1: Calculate India Daily Goldstein Averages")
print("=" * 80)

# Read India data
print("Reading India combined data...")
df_india = pd.read_csv('india_news_combined_sorted.csv')
print(f"Total India records: {len(df_india)}")

# Convert SQLDATE to datetime
df_india['Date'] = pd.to_datetime(df_india['SQLDATE'], format='%Y%m%d')

# Calculate daily average for India
print("Calculating India daily averages...")
india_daily = df_india.groupby('Date').agg({
    'GoldsteinScale': ['mean', 'count']
}).reset_index()
india_daily.columns = ['Date', 'India_Avg_Goldstein', 'India_Event_Count']
india_daily = india_daily.sort_values('Date')

# Save India daily averages
india_daily.to_csv('india_daily_goldstein_averages.csv', index=False)
print(f"India daily averages saved. Date range: {india_daily['Date'].min()} to {india_daily['Date'].max()}")

print("\n" + "=" * 80)
print("STEP 2: Load USA Daily Goldstein Averages")
print("=" * 80)

# Load USA data
usa_daily = pd.read_csv('usa/usa_daily_goldstein_averages.csv')
usa_daily['Date'] = pd.to_datetime(usa_daily['Date'])
usa_daily = usa_daily[['Date', 'Avg_GoldsteinScale', 'Event_Count']].copy()
usa_daily.columns = ['Date', 'USA_Avg_Goldstein', 'USA_Event_Count']
print(f"USA daily averages loaded. Date range: {usa_daily['Date'].min()} to {usa_daily['Date'].max()}")

print("\n" + "=" * 80)
print("STEP 3: Combine USA and India Goldstein Scores")
print("=" * 80)

# Merge USA and India data
combined = pd.merge(usa_daily, india_daily, on='Date', how='outer')
combined = combined.sort_values('Date')

# Fill any missing values with forward fill, then backward fill
combined = combined.fillna(method='ffill').fillna(method='bfill')

# Calculate different combination methods
print("Calculating combined metrics using multiple methods...")

# Method 1: Simple average
combined['Combined_Simple_Avg'] = (combined['USA_Avg_Goldstein'] + combined['India_Avg_Goldstein']) / 2

# Method 2: Weighted average by event count
total_events = combined['USA_Event_Count'] + combined['India_Event_Count']
combined['Combined_Weighted_Avg'] = (
    (combined['USA_Avg_Goldstein'] * combined['USA_Event_Count']) +
    (combined['India_Avg_Goldstein'] * combined['India_Event_Count'])
) / total_events

# Method 3: Product (captures interaction)
combined['Combined_Product'] = combined['USA_Avg_Goldstein'] * combined['India_Avg_Goldstein']

# Method 4: Geometric mean
combined['Combined_Geometric_Mean'] = np.sqrt(
    np.abs(combined['USA_Avg_Goldstein'] * combined['India_Avg_Goldstein'])
) * np.sign(combined['USA_Avg_Goldstein'] * combined['India_Avg_Goldstein'])

# Method 5: Bilateral sentiment (difference captures divergence)
combined['USA_India_Sentiment_Diff'] = combined['USA_Avg_Goldstein'] - combined['India_Avg_Goldstein']

print(f"Combined data created with {len(combined)} days")

print("\n" + "=" * 80)
print("STEP 4: Merge with Exchange Rates")
print("=" * 80)

# Load exchange rate data
exchange_rates = pd.read_csv('usd_inr_exchange_rates_1year.csv')
exchange_rates['Date'] = pd.to_datetime(exchange_rates['Date'])
print(f"Exchange rates loaded. Date range: {exchange_rates['Date'].min()} to {exchange_rates['Date'].max()}")

# Merge all data
final_data = pd.merge(combined, exchange_rates, on='Date', how='inner')
final_data = final_data.sort_values('Date')

# Calculate exchange rate changes (daily returns)
final_data['Exchange_Rate_Change'] = final_data['USD_to_INR'].pct_change() * 100
final_data['Exchange_Rate_Change_Abs'] = final_data['USD_to_INR'].diff()

print(f"Final merged dataset has {len(final_data)} days of overlapping data")

# Save combined data
final_data.to_csv('combined_goldstein_exchange_rates.csv', index=False)
print("Combined data saved to: combined_goldstein_exchange_rates.csv")

print("\n" + "=" * 80)
print("STEP 5: Calculate Correlations")
print("=" * 80)

# Remove any NaN values for correlation calculation
analysis_data = final_data.dropna()

# Calculate correlations between different Goldstein metrics and exchange rates
correlations = {}

goldstein_metrics = [
    'USA_Avg_Goldstein',
    'India_Avg_Goldstein',
    'Combined_Simple_Avg',
    'Combined_Weighted_Avg',
    'Combined_Product',
    'Combined_Geometric_Mean',
    'USA_India_Sentiment_Diff'
]

exchange_metrics = [
    'USD_to_INR',
    'Exchange_Rate_Change',
    'Exchange_Rate_Change_Abs'
]

print("\nPearson Correlations:")
print("-" * 80)
for goldstein_col in goldstein_metrics:
    for exchange_col in exchange_metrics:
        if goldstein_col in analysis_data.columns and exchange_col in analysis_data.columns:
            # Remove any remaining NaN values for this pair
            temp_data = analysis_data[[goldstein_col, exchange_col]].dropna()
            if len(temp_data) > 0:
                corr, p_value = stats.pearsonr(temp_data[goldstein_col], temp_data[exchange_col])
                correlations[f"{goldstein_col} vs {exchange_col}"] = {
                    'correlation': corr,
                    'p_value': p_value
                }
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"{goldstein_col:30} vs {exchange_col:25}: r={corr:7.4f}, p={p_value:.4f} {significance}")

# Save correlations
corr_df = pd.DataFrame(correlations).T
corr_df.to_csv('goldstein_exchange_correlations.csv')
print("\nCorrelations saved to: goldstein_exchange_correlations.csv")

print("\n" + "=" * 80)
print("STEP 6: Create Visualizations")
print("=" * 80)

# Create comprehensive plots
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Goldstein Scores vs USD/INR Exchange Rates Analysis', fontsize=16, fontweight='bold')

# Plot 1: Exchange Rate over time
ax1 = axes[0, 0]
ax1.plot(analysis_data['Date'], analysis_data['USD_to_INR'], color='green', linewidth=2)
ax1.set_title('USD/INR Exchange Rate Over Time', fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('USD to INR Rate')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: USA vs India Goldstein Scores
ax2 = axes[0, 1]
ax2.plot(analysis_data['Date'], analysis_data['USA_Avg_Goldstein'], label='USA', color='blue', linewidth=1.5)
ax2.plot(analysis_data['Date'], analysis_data['India_Avg_Goldstein'], label='India', color='orange', linewidth=1.5)
ax2.set_title('USA vs India Daily Average Goldstein Scores', fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Goldstein Score')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Combined Goldstein (Weighted) vs Exchange Rate
ax3 = axes[1, 0]
ax3_twin = ax3.twinx()
ax3.plot(analysis_data['Date'], analysis_data['Combined_Weighted_Avg'], color='purple', linewidth=2, label='Combined Goldstein')
ax3_twin.plot(analysis_data['Date'], analysis_data['USD_to_INR'], color='green', linewidth=2, alpha=0.6, label='Exchange Rate')
ax3.set_title('Combined Goldstein (Weighted) vs Exchange Rate', fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Combined Goldstein Score', color='purple')
ax3_twin.set_ylabel('USD/INR Rate', color='green')
ax3.tick_params(axis='y', labelcolor='purple')
ax3_twin.tick_params(axis='y', labelcolor='green')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Scatter plot - Combined Goldstein vs Exchange Rate
ax4 = axes[1, 1]
scatter = ax4.scatter(analysis_data['Combined_Weighted_Avg'], analysis_data['USD_to_INR'],
                     c=range(len(analysis_data)), cmap='viridis', alpha=0.6, s=30)
z = np.polyfit(analysis_data['Combined_Weighted_Avg'], analysis_data['USD_to_INR'], 1)
p = np.poly1d(z)
ax4.plot(analysis_data['Combined_Weighted_Avg'], p(analysis_data['Combined_Weighted_Avg']),
         "r--", linewidth=2, label=f'Trend Line')
corr_val = correlations.get('Combined_Weighted_Avg vs USD_to_INR', {}).get('correlation', 0)
ax4.set_title(f'Correlation: Combined Goldstein vs Exchange Rate (r={corr_val:.4f})', fontweight='bold')
ax4.set_xlabel('Combined Goldstein Score (Weighted)')
ax4.set_ylabel('USD/INR Rate')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Time progression')

# Plot 5: Sentiment Difference vs Exchange Rate Change
ax5 = axes[2, 0]
ax5_twin = ax5.twinx()
ax5.plot(analysis_data['Date'], analysis_data['USA_India_Sentiment_Diff'], color='red', linewidth=1.5, label='Sentiment Diff')
ax5_twin.plot(analysis_data['Date'], analysis_data['Exchange_Rate_Change'], color='blue', linewidth=1.5, alpha=0.6, label='Exchange Rate Change %')
ax5.set_title('USA-India Sentiment Difference vs Exchange Rate Change', fontweight='bold')
ax5.set_xlabel('Date')
ax5.set_ylabel('Sentiment Difference (USA - India)', color='red')
ax5_twin.set_ylabel('Exchange Rate Change (%)', color='blue')
ax5.tick_params(axis='y', labelcolor='red')
ax5_twin.tick_params(axis='y', labelcolor='blue')
ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax5_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax5.grid(True, alpha=0.3)
ax5.tick_params(axis='x', rotation=45)

# Plot 6: Correlation Heatmap Summary
ax6 = axes[2, 1]
# Prepare correlation matrix for key metrics
key_metrics = ['USA_Avg_Goldstein', 'India_Avg_Goldstein', 'Combined_Weighted_Avg',
               'USA_India_Sentiment_Diff', 'USD_to_INR', 'Exchange_Rate_Change']
corr_matrix = analysis_data[key_metrics].corr()

im = ax6.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax6.set_xticks(range(len(key_metrics)))
ax6.set_yticks(range(len(key_metrics)))
ax6.set_xticklabels(['USA GS', 'India GS', 'Combined GS', 'Sentiment Diff', 'Exchange Rate', 'ER Change'], rotation=45, ha='right')
ax6.set_yticklabels(['USA GS', 'India GS', 'Combined GS', 'Sentiment Diff', 'Exchange Rate', 'ER Change'])
ax6.set_title('Correlation Matrix Heatmap', fontweight='bold')

# Add correlation values to heatmap
for i in range(len(key_metrics)):
    for j in range(len(key_metrics)):
        text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax6, label='Correlation')

plt.tight_layout()
plt.savefig('goldstein_exchange_rate_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: goldstein_exchange_rate_analysis.png")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nData Summary:")
print(f"  Total overlapping days: {len(analysis_data)}")
print(f"  Date range: {analysis_data['Date'].min().date()} to {analysis_data['Date'].max().date()}")
print(f"\nExchange Rate:")
print(f"  Mean: {analysis_data['USD_to_INR'].mean():.4f}")
print(f"  Std: {analysis_data['USD_to_INR'].std():.4f}")
print(f"  Range: {analysis_data['USD_to_INR'].min():.4f} to {analysis_data['USD_to_INR'].max():.4f}")
print(f"\nUSA Goldstein Score:")
print(f"  Mean: {analysis_data['USA_Avg_Goldstein'].mean():.4f}")
print(f"  Std: {analysis_data['USA_Avg_Goldstein'].std():.4f}")
print(f"\nIndia Goldstein Score:")
print(f"  Mean: {analysis_data['India_Avg_Goldstein'].mean():.4f}")
print(f"  Std: {analysis_data['India_Avg_Goldstein'].std():.4f}")
print(f"\nCombined Goldstein Score (Weighted):")
print(f"  Mean: {analysis_data['Combined_Weighted_Avg'].mean():.4f}")
print(f"  Std: {analysis_data['Combined_Weighted_Avg'].std():.4f}")

print("\n" + "=" * 80)
print("KEY FINDINGS - STRONGEST CORRELATIONS")
print("=" * 80)

# Sort correlations by absolute value
sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
print("\nTop 10 Strongest Correlations:")
for i, (pair, stats_dict) in enumerate(sorted_corrs[:10], 1):
    significance = "***" if stats_dict['p_value'] < 0.001 else "**" if stats_dict['p_value'] < 0.01 else "*" if stats_dict['p_value'] < 0.05 else ""
    print(f"{i:2}. {pair:60} r={stats_dict['correlation']:7.4f}, p={stats_dict['p_value']:.4f} {significance}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. india_daily_goldstein_averages.csv")
print("  2. combined_goldstein_exchange_rates.csv")
print("  3. goldstein_exchange_correlations.csv")
print("  4. goldstein_exchange_rate_analysis.png")

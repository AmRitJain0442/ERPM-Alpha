import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load political news data
print("Loading political news data...")
political_news = pd.read_csv('india_financial_political_news_filtered.csv')

# Convert SQLDATE to datetime
political_news['Date'] = pd.to_datetime(political_news['SQLDATE'].astype(str), format='%Y%m%d')

# Aggregate political news by date
print("Aggregating political news by date...")
daily_political = political_news.groupby('Date').agg({
    'GoldsteinScale': ['mean', 'std', 'count'],
    'AvgTone': ['mean', 'std'],
    'NumMentions': 'sum',
    'NumArticles': 'sum'
}).reset_index()

# Flatten column names
daily_political.columns = ['Date', 'GoldsteinScale_mean', 'GoldsteinScale_std', 'Event_count',
                           'AvgTone_mean', 'AvgTone_std', 'Total_mentions', 'Total_articles']

# Load exchange rate data
print("Loading exchange rate data...")
exchange_rates = pd.read_csv('usd_inr_exchange_rates_1year.csv')
exchange_rates['Date'] = pd.to_datetime(exchange_rates['Date'])

# Merge datasets
print("Merging datasets...")
merged_data = pd.merge(daily_political, exchange_rates, on='Date', how='inner')
merged_data = merged_data.sort_values('Date')

# Save merged data
merged_data.to_csv('political_news_exchange_merged.csv', index=False)
print(f"Merged data saved. Total rows: {len(merged_data)}")
print(f"Date range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")

# Calculate correlation
corr_goldstein = merged_data['GoldsteinScale_mean'].corr(merged_data['USD_to_INR'])
corr_tone = merged_data['AvgTone_mean'].corr(merged_data['USD_to_INR'])

print(f"\nCorrelation between Goldstein Scale and USD/INR: {corr_goldstein:.4f}")
print(f"Correlation between Avg Tone and USD/INR: {corr_tone:.4f}")

# Create visualizations
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Political News Sentiment vs USD/INR Exchange Rate Analysis', fontsize=16, fontweight='bold')

# Plot 1: Goldstein Scale vs Exchange Rate (Time Series)
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
ax1.plot(merged_data['Date'], merged_data['GoldsteinScale_mean'],
         color='blue', linewidth=2, label='Goldstein Scale', alpha=0.7)
ax1_twin.plot(merged_data['Date'], merged_data['USD_to_INR'],
              color='red', linewidth=2, label='USD/INR', alpha=0.7)
ax1.set_xlabel('Date', fontweight='bold')
ax1.set_ylabel('Goldstein Scale (Mean)', color='blue', fontweight='bold')
ax1_twin.set_ylabel('USD/INR Exchange Rate', color='red', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='blue')
ax1_twin.tick_params(axis='y', labelcolor='red')
ax1.set_title('Political Sentiment (Goldstein) vs Exchange Rate Over Time')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Average Tone vs Exchange Rate (Time Series)
ax2 = axes[0, 1]
ax2_twin = ax2.twinx()
ax2.plot(merged_data['Date'], merged_data['AvgTone_mean'],
         color='green', linewidth=2, label='Avg Tone', alpha=0.7)
ax2_twin.plot(merged_data['Date'], merged_data['USD_to_INR'],
              color='red', linewidth=2, label='USD/INR', alpha=0.7)
ax2.set_xlabel('Date', fontweight='bold')
ax2.set_ylabel('Average Tone (Mean)', color='green', fontweight='bold')
ax2_twin.set_ylabel('USD/INR Exchange Rate', color='red', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='green')
ax2_twin.tick_params(axis='y', labelcolor='red')
ax2.set_title('Political Tone vs Exchange Rate Over Time')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Scatter plot - Goldstein vs Exchange Rate
ax3 = axes[1, 0]
scatter1 = ax3.scatter(merged_data['GoldsteinScale_mean'], merged_data['USD_to_INR'],
                       c=merged_data['Event_count'], cmap='viridis', alpha=0.6, s=50)
ax3.set_xlabel('Goldstein Scale (Mean)', fontweight='bold')
ax3.set_ylabel('USD/INR Exchange Rate', fontweight='bold')
ax3.set_title(f'Goldstein vs Exchange Rate (Corr: {corr_goldstein:.4f})')
ax3.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax3)
cbar1.set_label('Number of Events', rotation=270, labelpad=15)

# Add trend line
z = np.polyfit(merged_data['GoldsteinScale_mean'], merged_data['USD_to_INR'], 1)
p = np.poly1d(z)
ax3.plot(merged_data['GoldsteinScale_mean'], p(merged_data['GoldsteinScale_mean']),
         "r--", alpha=0.8, linewidth=2, label='Trend line')
ax3.legend()

# Plot 4: Scatter plot - Tone vs Exchange Rate
ax4 = axes[1, 1]
scatter2 = ax4.scatter(merged_data['AvgTone_mean'], merged_data['USD_to_INR'],
                       c=merged_data['Event_count'], cmap='plasma', alpha=0.6, s=50)
ax4.set_xlabel('Average Tone (Mean)', fontweight='bold')
ax4.set_ylabel('USD/INR Exchange Rate', fontweight='bold')
ax4.set_title(f'Tone vs Exchange Rate (Corr: {corr_tone:.4f})')
ax4.grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax4)
cbar2.set_label('Number of Events', rotation=270, labelpad=15)

# Add trend line
z2 = np.polyfit(merged_data['AvgTone_mean'], merged_data['USD_to_INR'], 1)
p2 = np.poly1d(z2)
ax4.plot(merged_data['AvgTone_mean'], p2(merged_data['AvgTone_mean']),
         "r--", alpha=0.8, linewidth=2, label='Trend line')
ax4.legend()

# Plot 5: Number of Political Events Over Time
ax5 = axes[2, 0]
ax5_twin = ax5.twinx()
ax5.bar(merged_data['Date'], merged_data['Event_count'],
        color='purple', alpha=0.5, label='Event Count', width=1)
ax5_twin.plot(merged_data['Date'], merged_data['USD_to_INR'],
              color='red', linewidth=2, label='USD/INR', alpha=0.7)
ax5.set_xlabel('Date', fontweight='bold')
ax5.set_ylabel('Number of Political Events', color='purple', fontweight='bold')
ax5_twin.set_ylabel('USD/INR Exchange Rate', color='red', fontweight='bold')
ax5.tick_params(axis='y', labelcolor='purple')
ax5_twin.tick_params(axis='y', labelcolor='red')
ax5.set_title('Political Event Volume vs Exchange Rate')
ax5.grid(True, alpha=0.3)
ax5.legend(loc='upper left')
ax5_twin.legend(loc='upper right')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 6: Rolling correlation
ax6 = axes[2, 1]
window = 7  # 7-day rolling window
merged_data['Rolling_corr_goldstein'] = merged_data['GoldsteinScale_mean'].rolling(window=window).corr(
    merged_data['USD_to_INR'])
merged_data['Rolling_corr_tone'] = merged_data['AvgTone_mean'].rolling(window=window).corr(
    merged_data['USD_to_INR'])

ax6.plot(merged_data['Date'], merged_data['Rolling_corr_goldstein'],
         color='blue', linewidth=2, label=f'{window}-day Goldstein Corr', alpha=0.7)
ax6.plot(merged_data['Date'], merged_data['Rolling_corr_tone'],
         color='green', linewidth=2, label=f'{window}-day Tone Corr', alpha=0.7)
ax6.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax6.set_xlabel('Date', fontweight='bold')
ax6.set_ylabel('Rolling Correlation', fontweight='bold')
ax6.set_title(f'{window}-Day Rolling Correlation with Exchange Rate')
ax6.grid(True, alpha=0.3)
ax6.legend()
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('political_news_exchange_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'political_news_exchange_analysis.png'")

# Additional statistics
print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)
print(f"\nPolitical News Events:")
print(f"  Total events: {merged_data['Event_count'].sum()}")
print(f"  Average events per day: {merged_data['Event_count'].mean():.2f}")
print(f"  Total articles: {merged_data['Total_articles'].sum()}")

print(f"\nGoldstein Scale:")
print(f"  Mean: {merged_data['GoldsteinScale_mean'].mean():.4f}")
print(f"  Std: {merged_data['GoldsteinScale_mean'].std():.4f}")
print(f"  Min: {merged_data['GoldsteinScale_mean'].min():.4f}")
print(f"  Max: {merged_data['GoldsteinScale_mean'].max():.4f}")

print(f"\nAverage Tone:")
print(f"  Mean: {merged_data['AvgTone_mean'].mean():.4f}")
print(f"  Std: {merged_data['AvgTone_mean'].std():.4f}")
print(f"  Min: {merged_data['AvgTone_mean'].min():.4f}")
print(f"  Max: {merged_data['AvgTone_mean'].max():.4f}")

print(f"\nExchange Rate (USD/INR):")
print(f"  Mean: {merged_data['USD_to_INR'].mean():.4f}")
print(f"  Std: {merged_data['USD_to_INR'].std():.4f}")
print(f"  Min: {merged_data['USD_to_INR'].min():.4f}")
print(f"  Max: {merged_data['USD_to_INR'].max():.4f}")

print("\n" + "="*60)

# plt.show()  # Commented out to avoid blocking

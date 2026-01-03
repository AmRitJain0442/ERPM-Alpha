import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

print("Loading data...")
df = pd.read_csv('india_news_gz_combined_sorted.csv', low_memory=False)

print(f"Loaded {len(df):,} records")

# Convert SQLDATE to datetime
print("Converting dates...")
df['Date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')

# Calculate daily statistics for Goldstein Scale
print("Calculating daily statistics...")
daily_stats = df.groupby('Date')['GoldsteinScale'].agg(['mean', 'min', 'max']).reset_index()

# Create the plot with stock market style
print("Creating plot...")
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the daily average line
ax.plot(daily_stats['Date'], daily_stats['mean'], linewidth=2, color='#1f77b4', label='Daily Average', zorder=3)

# Add shaded area showing daily min/max range
ax.fill_between(daily_stats['Date'], daily_stats['min'], daily_stats['max'], 
                alpha=0.2, color='lightgray', label='Daily Range (Min-Max)')

# Fill area between average and zero
ax.fill_between(daily_stats['Date'], daily_stats['mean'], 0, alpha=0.2, color='#1f77b4')

# Add zero reference line
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (0)')

plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Goldstein Scale', fontsize=12, fontweight='bold')
plt.title('Goldstein Scale vs Time - India News Events (Daily Average with Range)', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45, ha='right')

plt.tight_layout()

# Save the plot
output_file = 'goldstein_scale_vs_time.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as {output_file}")

# Show some statistics
print(f"\nGoldstein Scale Statistics:")
print(f"  Mean: {df['GoldsteinScale'].mean():.2f}")
print(f"  Median: {df['GoldsteinScale'].median():.2f}")
print(f"  Min: {df['GoldsteinScale'].min():.2f}")
print(f"  Max: {df['GoldsteinScale'].max():.2f}")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

print("Loading filtered news data...")
print(f"Start time: {datetime.now()}")

# Load the filtered data
df = pd.read_csv('india_financial_political_news_filtered.csv', low_memory=False)
print(f"Total records: {len(df):,}")

# Convert SQLDATE to datetime
df['Date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Group by date and count events
daily_counts = df.groupby('Date').size().reset_index(name='EventCount')
daily_counts = daily_counts.sort_values('Date')

print(f"Total days with data: {len(daily_counts)}")
print(f"Average events per day: {daily_counts['EventCount'].mean():.2f}")
print(f"Max events in a day: {daily_counts['EventCount'].max()}")

# Calculate rolling averages
daily_counts['Rolling7Day'] = daily_counts['EventCount'].rolling(window=7, center=True).mean()
daily_counts['Rolling30Day'] = daily_counts['EventCount'].rolling(window=30, center=True).mean()

# Create figure with multiple subplots
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('India Financial & Political News Timeline (GDELT Data)', fontsize=16, fontweight='bold')

# Plot 1: Daily event counts with rolling average
ax1 = axes[0]
ax1.plot(daily_counts['Date'], daily_counts['EventCount'],
         alpha=0.3, color='steelblue', linewidth=0.5, label='Daily Count')
ax1.plot(daily_counts['Date'], daily_counts['Rolling7Day'],
         color='darkblue', linewidth=2, label='7-Day Rolling Avg')
ax1.plot(daily_counts['Date'], daily_counts['Rolling30Day'],
         color='red', linewidth=2, label='30-Day Rolling Avg')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Number of Events', fontsize=12)
ax1.set_title('Daily Event Counts with Rolling Averages', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Monthly aggregation
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_counts = df.groupby('YearMonth').size().reset_index(name='EventCount')
monthly_counts['Date'] = monthly_counts['YearMonth'].dt.to_timestamp()

ax2 = axes[1]
ax2.bar(monthly_counts['Date'], monthly_counts['EventCount'],
        width=25, color='teal', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Number of Events', fontsize=12)
ax2.set_title('Monthly Event Counts', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels on bars
for i, (date, count) in enumerate(zip(monthly_counts['Date'], monthly_counts['EventCount'])):
    ax2.text(date, count, f'{count:,}', ha='center', va='bottom', fontsize=9)

# Plot 3: Cumulative events over time
daily_counts['CumulativeEvents'] = daily_counts['EventCount'].cumsum()

ax3 = axes[2]
ax3.plot(daily_counts['Date'], daily_counts['CumulativeEvents'],
         color='darkgreen', linewidth=2)
ax3.fill_between(daily_counts['Date'], daily_counts['CumulativeEvents'],
                  alpha=0.3, color='lightgreen')
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Cumulative Events', fontsize=12)
ax3.set_title('Cumulative Event Count Over Time', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Format y-axis to show numbers with commas
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('india_news_timeline.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as: india_news_timeline.png")

# Additional statistics plot - Distribution by day of week and hour
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('India News Events - Temporal Patterns', fontsize=14, fontweight='bold')

# Day of week distribution
df['DayOfWeek'] = df['Date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df['DayOfWeek'].value_counts().reindex(day_order)

ax_day = axes2[0]
ax_day.bar(range(len(day_counts)), day_counts.values, color='coral', alpha=0.7, edgecolor='black')
ax_day.set_xticks(range(len(day_counts)))
ax_day.set_xticklabels(day_counts.index, rotation=45, ha='right')
ax_day.set_ylabel('Number of Events', fontsize=11)
ax_day.set_title('Events by Day of Week', fontsize=12)
ax_day.grid(True, alpha=0.3, axis='y')

# Year distribution
df['Year'] = df['Date'].dt.year
year_counts = df['Year'].value_counts().sort_index()

ax_year = axes2[1]
ax_year.bar(year_counts.index, year_counts.values, color='mediumpurple', alpha=0.7, edgecolor='black')
ax_year.set_xlabel('Year', fontsize=11)
ax_year.set_ylabel('Number of Events', fontsize=11)
ax_year.set_title('Events by Year', fontsize=12)
ax_year.grid(True, alpha=0.3, axis='y')
ax_year.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Add value labels
for year, count in zip(year_counts.index, year_counts.values):
    ax_year.text(year, count, f'{count:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('india_news_patterns.png', dpi=300, bbox_inches='tight')
print("Plot saved as: india_news_patterns.png")

# Print summary statistics
print("\n--- Summary Statistics ---")
print(f"Total events: {len(df):,}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Total days covered: {(df['Date'].max() - df['Date'].min()).days + 1}")
print(f"Days with events: {len(daily_counts)}")
print(f"\nDaily statistics:")
print(f"  Mean: {daily_counts['EventCount'].mean():.2f} events/day")
print(f"  Median: {daily_counts['EventCount'].median():.2f} events/day")
print(f"  Max: {daily_counts['EventCount'].max()} events/day")
print(f"  Min: {daily_counts['EventCount'].min()} events/day")
print(f"\nMonthly statistics:")
print(f"  Mean: {monthly_counts['EventCount'].mean():.2f} events/month")
print(f"  Total months: {len(monthly_counts)}")
print(f"\nTop 5 busiest days:")
top_days = daily_counts.nlargest(5, 'EventCount')[['Date', 'EventCount']]
for idx, row in top_days.iterrows():
    print(f"  {row['Date'].date()}: {row['EventCount']} events")

print(f"\nEnd time: {datetime.now()}")
print("Done!")

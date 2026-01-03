import pandas as pd

# Read the USA combined data
print("Reading USA combined data...")
df = pd.read_csv('usa/usa_news_combined_sorted.csv')

print(f"Total records: {len(df)}")

# Convert SQLDATE to datetime format
df['Date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')

# Calculate daily average of GoldsteinScale
print("Calculating daily average of Goldstein scores...")
daily_avg = df.groupby('Date').agg({
    'GoldsteinScale': ['mean', 'count', 'std', 'min', 'max']
}).reset_index()

# Flatten column names
daily_avg.columns = ['Date', 'Avg_GoldsteinScale', 'Event_Count', 'Std_GoldsteinScale', 'Min_GoldsteinScale', 'Max_GoldsteinScale']

# Sort by date
daily_avg = daily_avg.sort_values('Date')

# Save to CSV
output_file = 'usa/usa_daily_goldstein_averages.csv'
daily_avg.to_csv(output_file, index=False)

print(f"\nDaily averages saved to: {output_file}")
print(f"Date range: {daily_avg['Date'].min()} to {daily_avg['Date'].max()}")
print(f"Total days: {len(daily_avg)}")
print(f"\nFirst few rows:")
print(daily_avg.head(10))
print(f"\nLast few rows:")
print(daily_avg.tail(10))
print(f"\nOverall statistics:")
print(f"Mean Goldstein score across all days: {daily_avg['Avg_GoldsteinScale'].mean():.4f}")
print(f"Std deviation: {daily_avg['Avg_GoldsteinScale'].std():.4f}")

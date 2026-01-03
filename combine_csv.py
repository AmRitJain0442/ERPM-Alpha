import pandas as pd
import glob
import os

# Get all CSV files (excluding .gstmp files)
csv_files = glob.glob('india_news_data_*.csv')
csv_files = [f for f in csv_files if not f.endswith('.gstmp')]

print(f"Found {len(csv_files)} CSV files to combine")

# Read and combine all CSV files
all_data = []
for i, file in enumerate(csv_files, 1):
    try:
        df = pd.read_csv(file, low_memory=False)
        all_data.append(df)
        print(f"Processed {i}/{len(csv_files)}: {file} ({len(df)} rows)")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Combine all dataframes
print("\nCombining all data...")
combined_df = pd.concat(all_data, ignore_index=True)
print(f"Total rows: {len(combined_df)}")

# Sort by SQLDATE (date column)
print("Sorting by date...")
combined_df = combined_df.sort_values('SQLDATE')

# Save the combined and sorted data
output_file = 'india_news_combined_sorted.csv'
print(f"\nSaving to {output_file}...")
combined_df.to_csv(output_file, index=False)

print(f"\n✓ Successfully created {output_file}")
print(f"  Total records: {len(combined_df):,}")
print(f"  Date range: {combined_df['SQLDATE'].min()} to {combined_df['SQLDATE'].max()}")
print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

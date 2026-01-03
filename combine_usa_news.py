import pandas as pd
import gzip
import os
import glob
from pathlib import Path

# Path to USA folder
usa_folder = r"c:\Users\amrit\Desktop\gdelt_india\usa"

# Get all .csv.gz files
csv_files = sorted(glob.glob(os.path.join(usa_folder, "usa_news_*.csv.gz")))

print(f"Found {len(csv_files)} files to process")

# List to store dataframes
dfs = []

# Read and concatenate all files
for i, file in enumerate(csv_files):
    if i % 100 == 0:
        print(f"Processing file {i+1}/{len(csv_files)}: {os.path.basename(file)}")
    
    try:
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            df = pd.read_csv(f, sep=',', low_memory=False)
            dfs.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

print(f"Read {len(dfs)} files successfully")

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataframe shape: {combined_df.shape}")
    
    # Display column names to understand the date column
    print(f"Columns: {combined_df.columns.tolist()}")
    print(f"First few rows:\n{combined_df.head()}")
    
    # Assuming the first column or a date-like column exists
    # Common GDELT columns include 'GLOBALEVENTID', 'Day', 'MonthYear', etc.
    # We'll sort by the most likely date column
    
    if 'Day' in combined_df.columns:
        print("Sorting by 'Day' column...")
        combined_df = combined_df.sort_values('Day')
    elif 'MonthYear' in combined_df.columns:
        print("Sorting by 'MonthYear' column...")
        combined_df = combined_df.sort_values('MonthYear')
    elif combined_df.columns[1] is not None:
        print(f"Sorting by '{combined_df.columns[1]}' column...")
        combined_df = combined_df.sort_values(combined_df.columns[1])
    
    # Save to CSV (non-compressed for easier access)
    output_file = os.path.join(usa_folder, "usa_news_combined_sorted.csv")
    print(f"Saving to {output_file}...")
    combined_df.to_csv(output_file, index=False, sep=',')
    print(f"Successfully saved combined and sorted file!")
    print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
else:
    print("No files were successfully read!")

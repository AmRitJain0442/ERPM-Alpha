"""
Create Sample Datasets for Preview
This script creates small sample files from each dataset for quick review
"""

import pandas as pd
from pathlib import Path
import shutil
import json

# Configuration
WORKSPACE_ROOT = Path(__file__).parent
SAMPLE_DIR = WORKSPACE_ROOT / "sample_datasets"
SAMPLE_SIZE = 500  # Number of rows to include in each sample

# Dataset files to sample
DATASET_FILES = {
    "exchange_rates": [
        "usd_inr_exchange_rates_1year.csv",
        "usd_inr_exchange_rates_1year.json",
        "exchange_rate_goldstein_merged.csv",
        "combined_goldstein_exchange_rates.csv",
    ],
    "gdelt_news": [
        "combined-gdelt.csv",
        "india_news_combined_sorted.csv",
        "india_news_gz_combined_sorted.csv",
        "usa_news_combined_sorted.csv",
        "india_financial_political_news_filtered.csv",
        "india_daily_goldstein_averages.csv",
    ],
    "master_dataset": [
        "Super_Master_Dataset.csv",
    ],
    "correlation_analysis": [
        "goldstein_exchange_correlations.csv",
        "political_news_exchange_merged.csv",
        "bq-results-20260115-090715-1768468077035.csv",
        "IMF_3.csv",
    ],
    "trade_data": [
        "india_usa_trade/output/india_usa_trade_2010_2025.csv",
        "india_usa_trade/output/india_usa_trade_2019_2023.csv",
        "india_usa_trade/output/india_usa_trade_2022_2023.csv",
        "india_usa_trade/output/india_usa_trade_2023_2023.csv",
        "india_usa_trade/output/trade_balance_analysis.csv",
        "india_usa_trade/output/commodity_shift_multiyear.csv",
        "india_usa_trade/output/commodity_analysis_2023.csv",
        "india_usa_trade/output/commodity_analysis_2024.csv",
        "india_usa_trade/output/seasonality_analysis_2023.csv",
        "india_usa_trade/output/seasonality_analysis_2024.csv",
    ],
    "monte_carlo_forecasts": [
        "monte_carlo_simulation/monte_carlo_forecast.csv",
        "monte_carlo_simulation/monte_carlo_statistics.csv",
        "monte_carlo_simulation/weekly_forecast_summary.csv",
        "monte_carlo_simulation/week_1_detailed_results.csv",
        "monte_carlo_simulation/week_2_detailed_results.csv",
        "monte_carlo_simulation/week_3_detailed_results.csv",
        "monte_carlo_simulation/week_4_detailed_results.csv",
    ],
    "garch_model": [
        "GARCH/output/predictions_price.csv",
        "GARCH/output/feature_importance.csv",
        "GARCH/output/garch_comparison.csv",
    ],
}


def create_sample_csv(source_path, dest_path, sample_size):
    """Create a sample CSV file with first N rows"""
    try:
        # Read the CSV file
        df = pd.read_csv(source_path, nrows=sample_size)
        
        # Get file info
        total_rows = sum(1 for _ in open(source_path, 'r', encoding='utf-8', errors='ignore')) - 1
        file_size = source_path.stat().st_size / (1024 * 1024)  # Size in MB
        
        # Save sample
        df.to_csv(dest_path, index=False)
        
        return {
            'status': 'success',
            'sample_rows': len(df),
            'total_rows': total_rows,
            'file_size_mb': round(file_size, 2),
            'columns': list(df.columns)
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def create_sample_json(source_path, dest_path):
    """Create a sample JSON file"""
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If it's a list, sample it
        if isinstance(data, list):
            sample_data = data[:SAMPLE_SIZE]
        else:
            sample_data = data
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        file_size = source_path.stat().st_size / (1024 * 1024)
        return {
            'status': 'success',
            'sample_items': len(sample_data) if isinstance(sample_data, list) else 1,
            'file_size_mb': round(file_size, 2)
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def create_info_file(category_dir, file_info):
    """Create an info file with dataset statistics"""
    info_path = category_dir / "DATASET_INFO.txt"
    
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SAMPLE DATASETS - {category_dir.name.upper()}\n")
        f.write("=" * 80 + "\n\n")
        f.write("This folder contains sample data (first 500 rows) from each dataset.\n")
        f.write("These samples are for quick preview and analysis.\n\n")
        f.write("-" * 80 + "\n")
        f.write("FILE STATISTICS:\n")
        f.write("-" * 80 + "\n\n")
        
        for filename, info in file_info.items():
            f.write(f"📄 {filename}\n")
            if info['status'] == 'success':
                f.write(f"   Sample Rows: {info.get('sample_rows', 'N/A')}\n")
                f.write(f"   Total Rows: {info.get('total_rows', 'N/A')}\n")
                f.write(f"   File Size: {info.get('file_size_mb', 'N/A')} MB\n")
                if 'columns' in info:
                    f.write(f"   Columns: {', '.join(info['columns'][:5])}")
                    if len(info['columns']) > 5:
                        f.write(f" ... (+{len(info['columns']) - 5} more)")
                    f.write("\n")
            else:
                f.write(f"   Status: Error - {info.get('error', 'Unknown')}\n")
            f.write("\n")


def create_main_readme(sample_dir, all_stats):
    """Create main README for sample datasets"""
    readme_path = sample_dir / "README.md"
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# GDELT India Research Datasets - Sample Preview\n\n")
        f.write("This folder contains sample datasets (first 500 rows from each file) ")
        f.write("for quick preview and analysis.\n\n")
        f.write("## Purpose\n\n")
        f.write("These samples allow reviewers to:\n")
        f.write("- Understand the data structure and format\n")
        f.write("- Preview column names and data types\n")
        f.write("- Quickly analyze a subset of the data\n")
        f.write("- Validate data quality without downloading large files\n\n")
        f.write("## Dataset Categories\n\n")
        
        for category, stats in all_stats.items():
            f.write(f"### {category.replace('_', ' ').title()}\n\n")
            f.write(f"**Files:** {len(stats)} dataset files\n\n")
            
            for filename, info in stats.items():
                if info['status'] == 'success':
                    f.write(f"- **{filename}**\n")
                    total_rows = info.get('total_rows', 'N/A')
                    if isinstance(total_rows, int):
                        f.write(f"  - Sample: {info.get('sample_rows', 'N/A')} rows ")
                        f.write(f"(out of {total_rows:,} total)\n")
                    else:
                        f.write(f"  - Sample: {info.get('sample_rows', 'N/A')} rows/items\n")
                    f.write(f"  - Original Size: {info.get('file_size_mb', 'N/A')} MB\n")
            f.write("\n")
        
        f.write("## Full Dataset\n\n")
        f.write("The complete dataset is available on Hugging Face:\n")
        f.write("https://huggingface.co/datasets/AmritJain/gdelt-india-research-datasets\n\n")
        f.write("## Usage\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n\n")
        f.write("# Load a sample dataset\n")
        f.write("df = pd.read_csv('sample_datasets/exchange_rates/usd_inr_exchange_rates_1year.csv')\n")
        f.write("print(df.head())\n")
        f.write("print(df.info())\n")
        f.write("```\n\n")
        f.write("---\n")
        f.write("*Last Updated: January 2026*\n")


def main():
    """Main execution function"""
    print("=" * 80)
    print("Creating Sample Datasets for Review")
    print("=" * 80)
    
    # Clean up existing sample directory
    if SAMPLE_DIR.exists():
        print(f"\n🧹 Removing existing sample directory...")
        shutil.rmtree(SAMPLE_DIR)
    
    SAMPLE_DIR.mkdir(exist_ok=True)
    print(f"✓ Created sample directory: {SAMPLE_DIR}")
    
    all_stats = {}
    
    # Process each category
    for category, files in DATASET_FILES.items():
        print(f"\n📁 Processing {category}...")
        category_dir = SAMPLE_DIR / category
        category_dir.mkdir(exist_ok=True)
        
        file_info = {}
        
        for file_path in files:
            source = WORKSPACE_ROOT / file_path
            filename = Path(file_path).name
            dest = category_dir / filename
            
            if not source.exists():
                print(f"  ⚠️  Skipped: {filename} (not found)")
                file_info[filename] = {'status': 'error', 'error': 'File not found'}
                continue
            
            print(f"  📄 Processing: {filename}...")
            
            if filename.endswith('.csv'):
                info = create_sample_csv(source, dest, SAMPLE_SIZE)
            elif filename.endswith('.json'):
                info = create_sample_json(source, dest)
            else:
                print(f"  ⚠️  Skipped: {filename} (unsupported format)")
                continue
            
            file_info[filename] = info
            
            if info['status'] == 'success':
                print(f"     ✓ Created sample ({info.get('sample_rows', info.get('sample_items', 'N/A'))} rows/items)")
            else:
                print(f"     ✗ Error: {info['error']}")
        
        # Create info file for this category
        create_info_file(category_dir, file_info)
        all_stats[category] = file_info
        print(f"  ✓ Created DATASET_INFO.txt")
    
    # Create main README
    print(f"\n📝 Creating main README...")
    create_main_readme(SAMPLE_DIR, all_stats)
    print(f"✓ Created README.md")
    
    # Calculate total statistics
    total_files = sum(len(stats) for stats in all_stats.values())
    success_files = sum(1 for stats in all_stats.values() for info in stats.values() if info['status'] == 'success')
    
    print("\n" + "=" * 80)
    print("✅ SAMPLE DATASETS CREATED!")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"   Total Categories: {len(all_stats)}")
    print(f"   Total Files Processed: {success_files}/{total_files}")
    print(f"   Sample Directory: {SAMPLE_DIR}")
    print(f"\n💡 Share the '{SAMPLE_DIR.name}' folder with your reviewer!")
    print(f"   Each file contains the first {SAMPLE_SIZE} rows for preview.\n")


if __name__ == "__main__":
    main()

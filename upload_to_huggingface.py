"""
Upload GDELT India Research Datasets to Hugging Face Hub
This script organizes and uploads all dataset files to Hugging Face
"""

from huggingface_hub import HfApi, create_repo, upload_folder
import os
from pathlib import Path
import shutil

# Configuration
WORKSPACE_ROOT = Path(__file__).parent
DATASET_NAME = "gdelt-india-research-datasets"  # Change this to your preferred name
USERNAME = None  # Will be auto-detected or you can set manually

# Dataset files categorized by type
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


def create_dataset_structure():
    """Create organized directory structure for upload"""
    temp_dir = WORKSPACE_ROOT / "temp_hf_upload"
    
    # Clean up if exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    temp_dir.mkdir(exist_ok=True)
    
    # Copy files into organized structure
    for category, files in DATASET_FILES.items():
        category_dir = temp_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for file_path in files:
            source = WORKSPACE_ROOT / file_path
            if source.exists():
                dest = category_dir / Path(file_path).name
                shutil.copy2(source, dest)
                print(f"✓ Copied: {file_path} -> {category}/{Path(file_path).name}")
            else:
                print(f"⚠ Warning: File not found: {file_path}")
    
    return temp_dir


def create_readme(temp_dir):
    """Create README.md with dataset description"""
    readme_content = """# GDELT India Research Datasets

This dataset collection contains comprehensive data for analyzing the relationship between news sentiment, political events, and economic indicators for India-US relations.

## Dataset Categories

### 1. Exchange Rates
- **usd_inr_exchange_rates_1year.csv**: Daily USD-INR exchange rates for one year
- **exchange_rate_goldstein_merged.csv**: Exchange rates merged with Goldstein scale scores
- **combined_goldstein_exchange_rates.csv**: Combined analysis of Goldstein scores and exchange rate movements

### 2. GDELT News Data
- **combined-gdelt.csv**: Combined GDELT news events data
- **india_news_combined_sorted.csv**: Processed India-related news articles
- **india_news_gz_combined_sorted.csv**: Compressed India news dataset
- **usa_news_combined_sorted.csv**: USA news articles
- **india_financial_political_news_filtered.csv**: Filtered financial and political news
- **india_daily_goldstein_averages.csv**: Daily average Goldstein scores for India

### 3. Master Dataset
- **Super_Master_Dataset.csv**: Comprehensive merged dataset with all variables

### 4. Correlation Analysis
- **goldstein_exchange_correlations.csv**: Correlation metrics between Goldstein scale and exchange rates
- **political_news_exchange_merged.csv**: Political news sentiment merged with exchange rate data
- **bq-results-*.csv**: BigQuery analysis results
- **IMF_3.csv**: IMF economic indicators

### 5. Trade Data (India-USA)
- Multiple CSV files containing bilateral trade data from 2010-2025
- Trade balance analysis
- Commodity shift analysis across multiple years
- Seasonality analysis for 2023-2024

### 6. Monte Carlo Forecasts
- **monte_carlo_forecast.csv**: Exchange rate forecasts using Monte Carlo simulation
- **monte_carlo_statistics.csv**: Statistical summary of simulations
- **weekly_forecast_summary.csv**: Weekly rolling forecasts
- **week_*_detailed_results.csv**: Detailed results for each forecast week

### 7. GARCH Model Outputs
- **predictions_price.csv**: GARCH model price predictions
- **feature_importance.csv**: Feature importance analysis
- **garch_comparison.csv**: Comparison of different GARCH model variants

## Research Context

This dataset supports research on:
- Impact of political news sentiment on exchange rates
- GDELT event analysis for India-US relations
- Time series forecasting of exchange rates
- Volatility modeling using GARCH and Monte Carlo methods
- Trade balance and its correlation with economic indicators

## Data Sources

- **GDELT Project**: Global news event database
- **Exchange Rate Data**: Official currency exchange sources
- **Trade Data**: US Census Bureau and UN Comtrade
- **IMF**: International Monetary Fund economic indicators

## Citation

If you use this dataset, please cite:
```
GDELT India Research Datasets (2026)
Compiled from GDELT Project, IMF, and US Census Bureau
Available on Hugging Face Hub
```

## License

Please ensure compliance with individual data source licenses:
- GDELT data is available under their terms of use
- IMF data is publicly available
- Trade data from official government sources

## File Formats

All datasets are provided in CSV format for easy processing with:
- Python (pandas, numpy)
- R (data.table, tidyverse)
- Excel and other spreadsheet applications

## Updates

Dataset last updated: January 2026

For questions or issues, please open a discussion on the Hugging Face Hub.
"""
    
    readme_path = temp_dir / "README.md"
    readme_path.write_text(readme_content, encoding='utf-8')
    print(f"✓ Created README.md")
    return readme_path


def upload_to_huggingface(temp_dir, username=None):
    """Upload dataset to Hugging Face Hub"""
    from huggingface_hub import HfFolder
    
    api = HfApi()
    
    # Get username if not provided
    if username is None:
        try:
            # Try to get token first
            token = HfFolder.get_token()
            if token:
                username = api.whoami(token=token)["name"]
                print(f"✓ Detected username: {username}")
            else:
                raise Exception("No token found")
        except Exception as e:
            print(f"❌ Error: You need to login to Hugging Face first")
            print(f"   Run: hf auth login")
            print(f"   Error details: {e}")
            return False
    
    repo_id = f"{username}/{DATASET_NAME}"
    
    try:
        # Create repository
        print(f"\n📦 Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False  # Set to True if you want a private dataset
        )
        print(f"✓ Repository created/verified")
        
        # Upload files in smaller batches to handle large files better
        print(f"\n⬆️  Uploading files (this may take a while for large files)...")
        
        # Upload each category separately
        for category in os.listdir(temp_dir):
            category_path = temp_dir / category
            if category_path.is_dir():
                print(f"\n📂 Uploading {category}...")
                try:
                    api.upload_folder(
                        folder_path=str(category_path),
                        path_in_repo=category,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Upload {category} datasets"
                    )
                    print(f"✓ {category} uploaded successfully")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to upload {category}: {e}")
                    print(f"   Continuing with other categories...")
            elif category_path.is_file():
                # Upload README and other files
                print(f"📄 Uploading {category}...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(category_path),
                        path_in_repo=category,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Upload {category}"
                    )
                    print(f"✓ {category} uploaded successfully")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to upload {category}: {e}")
        
        print(f"\n✅ Upload complete! View at: https://huggingface.co/datasets/{repo_id}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        return False


def main():
    """Main execution function"""
    print("=" * 70)
    print("GDELT India Research Datasets - Hugging Face Upload")
    print("=" * 70)
    
    # Step 1: Create organized structure
    print("\n📁 Step 1: Organizing dataset files...")
    temp_dir = create_dataset_structure()
    
    # Step 2: Create README
    print("\n📝 Step 2: Creating README...")
    create_readme(temp_dir)
    
    # Step 3: Upload to Hugging Face
    print("\n🚀 Step 3: Uploading to Hugging Face Hub...")
    success = upload_to_huggingface(temp_dir, username=USERNAME)
    
    # Clean up
    if success:
        print(f"\n🧹 Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print(f"✓ Temporary directory removed")
        
        print("\n" + "=" * 70)
        print("✅ UPLOAD COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Your dataset is now available at:")
        print(f"   https://huggingface.co/datasets/{USERNAME or 'YOUR_USERNAME'}/{DATASET_NAME}")
        print(f"\n💡 You can now use it in your code with:")
        print(f"   from datasets import load_dataset")
        print(f"   dataset = load_dataset('{USERNAME or 'YOUR_USERNAME'}/{DATASET_NAME}')")
    else:
        print(f"\n⚠️  Upload failed. Temporary files kept at: {temp_dir}")
        print(f"   You can review and retry.")


if __name__ == "__main__":
    main()

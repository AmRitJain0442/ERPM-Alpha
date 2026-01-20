"""
India-USA Trade Data Analysis - Main Orchestration Script

This script provides a unified interface for fetching and analyzing
India-USA bilateral trade data from multiple sources.

Supported data sources:
1. US Census Bureau International Trade API
2. UN Comtrade via world_trade_data library
3. Local CSV files (Kaggle/Ministry of Commerce downloads)

Analytical capabilities:
1. Trade Deficit/Surplus Tracking
2. Commodity Shift Analysis (e.g., Electronics post-2018)
3. Seasonality Analysis
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from config import (
    CENSUS_API_KEY,
    HS_CODES,
    OUTPUT_DIR,
    DEFAULT_DATE_RANGE,
)
from census_api_fetcher import CensusTradeAPI
from comtrade_fetcher import ComtradeTradeAPI, WITS_AVAILABLE


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


class TradeAnalyzer:
    """
    Main class for India-USA trade data analysis.

    Combines data from multiple sources and provides analytical functions.
    """

    def __init__(self, data_source: str = "census"):
        """
        Initialize the Trade Analyzer.

        Args:
            data_source: "census" for US Census Bureau, "comtrade" for UN Comtrade
        """
        self.data_source = data_source

        if data_source == "census":
            self.api = CensusTradeAPI()
        elif data_source == "comtrade":
            if not WITS_AVAILABLE:
                raise ImportError("world_trade_data not installed for comtrade source")
            self.api = ComtradeTradeAPI()
        else:
            raise ValueError(f"Unknown data source: {data_source}")

        self.data = None
        ensure_output_dir()

    def fetch_multi_year_data(
        self,
        start_year: int = None,
        end_year: int = None
    ) -> pd.DataFrame:
        """
        Fetch trade data for multiple years.

        Args:
            start_year: Starting year (default from config)
            end_year: Ending year (default from config)

        Returns:
            DataFrame with multi-year trade data
        """
        start_year = start_year or DEFAULT_DATE_RANGE["start_year"]
        end_year = end_year or DEFAULT_DATE_RANGE["end_year"]

        print(f"\nFetching trade data from {start_year} to {end_year}...")
        print(f"Data source: {self.data_source.upper()}")
        print("-" * 50)

        if self.data_source == "census":
            self.data = self.api.fetch_yearly_trade_summary(start_year, end_year)
        else:
            years = [str(y) for y in range(start_year, end_year + 1)]
            self.data = self.api.fetch_bilateral_trade_summary(years)

        if not self.data.empty:
            output_file = os.path.join(
                OUTPUT_DIR,
                f"india_usa_trade_{start_year}_{end_year}.csv"
            )
            self.data.to_csv(output_file, index=False)
            print(f"\nData saved to: {output_file}")
            print(f"Total records: {len(self.data)}")

        return self.data

    def analyze_trade_balance(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Analysis 1: Trade Deficit/Surplus Tracking

        Compares exports vs imports over time to visualize the trade balance.

        Args:
            df: DataFrame with trade data (uses self.data if None)

        Returns:
            DataFrame with trade balance analysis
        """
        df = df if df is not None else self.data

        if df is None or df.empty:
            print("No data available. Fetch data first.")
            return pd.DataFrame()

        print("\n" + "=" * 60)
        print("ANALYSIS 1: Trade Balance (Deficit/Surplus) Tracking")
        print("=" * 60)

        if self.data_source == "census":
            # Census data has direction column and ALL_VAL_YR for yearly value
            if "ALL_VAL_YR" in df.columns:
                df["trade_value"] = pd.to_numeric(df["ALL_VAL_YR"], errors="coerce")

            balance_df = df.groupby(["fetch_year", "direction"]).agg({
                "trade_value": "sum"
            }).reset_index()

            # Pivot for comparison
            pivot_df = balance_df.pivot(
                index="fetch_year",
                columns="direction",
                values="trade_value"
            ).reset_index()

            if "exports" in pivot_df.columns and "imports" in pivot_df.columns:
                pivot_df["trade_balance"] = pivot_df["exports"] - pivot_df["imports"]
                pivot_df["surplus_deficit"] = pivot_df["trade_balance"].apply(
                    lambda x: "Surplus" if x > 0 else "Deficit"
                )

                print("\nUS Trade Balance with India (in USD):")
                print(pivot_df.to_string(index=False))

                # Save analysis
                output_file = os.path.join(OUTPUT_DIR, "trade_balance_analysis.csv")
                pivot_df.to_csv(output_file, index=False)
                print(f"\nSaved to: {output_file}")

                # Plot if available
                if PLOTTING_AVAILABLE:
                    self._plot_trade_balance(pivot_df)

                return pivot_df

        print("Trade balance analysis completed.")
        return df

    def analyze_commodity_shift(
        self,
        year: str = "2023",
        hs_codes: list = None
    ) -> pd.DataFrame:
        """
        Analysis 2: Commodity Shift Analysis

        Drill down into specific HS Codes to see how trade in specific
        categories has changed (e.g., electronics post-China tariffs in 2018).

        Args:
            year: Year to analyze
            hs_codes: List of HS codes to analyze (uses defaults if None)

        Returns:
            DataFrame with commodity-level analysis
        """
        print("\n" + "=" * 60)
        print("ANALYSIS 2: Commodity Shift Analysis")
        print("=" * 60)

        if hs_codes is None:
            hs_codes = list(HS_CODES.keys())[:7]  # Top 7 categories

        print(f"\nFetching commodity data for {year}...")
        print(f"HS Codes: {', '.join(hs_codes)}")

        if self.data_source == "census":
            commodity_data = self.api.fetch_commodity_breakdown(
                year=year,
                direction="imports",
                hs_codes=hs_codes
            )
        else:
            commodity_data = self.api.fetch_product_level_trade(
                year=year,
                products=hs_codes,
                direction="import"
            )

        if not commodity_data.empty:
            print("\nCommodity Breakdown:")
            print(commodity_data.to_string())

            output_file = os.path.join(OUTPUT_DIR, f"commodity_analysis_{year}.csv")
            commodity_data.to_csv(output_file, index=False)
            print(f"\nSaved to: {output_file}")

        return commodity_data

    def analyze_seasonality(
        self,
        year: str = "2023"
    ) -> pd.DataFrame:
        """
        Analysis 3: Seasonality Analysis

        Fetch monthly data to identify seasonal spikes in specific goods
        (e.g., agricultural products, textiles).

        Args:
            year: Year to analyze

        Returns:
            DataFrame with monthly trade data
        """
        print("\n" + "=" * 60)
        print("ANALYSIS 3: Seasonality Analysis")
        print("=" * 60)

        if self.data_source != "census":
            print("Seasonality analysis requires Census API (monthly data).")
            return pd.DataFrame()

        monthly_data = []

        print(f"\nFetching monthly data for {year}...")

        for month in range(1, 13):
            month_str = f"{month:02d}"  # Zero-pad: 1 -> "01", 12 -> "12"
            print(f"  Month {month_str}...", end=" ")

            try:
                exports = self.api.fetch_exports_to_india(year=year, month=month_str)
                if not exports.empty:
                    exports["month"] = month
                    monthly_data.append(exports)
                    print("exports OK", end=" ")
            except Exception as e:
                print(f"exports failed", end=" ")

            try:
                imports = self.api.fetch_imports_from_india(year=year, month=month_str)
                if not imports.empty:
                    imports["month"] = month
                    monthly_data.append(imports)
                    print("imports OK")
            except Exception as e:
                print("imports failed")

        if monthly_data:
            df = pd.concat(monthly_data, ignore_index=True)

            print("\nMonthly Trade Summary:")
            print(df.to_string())

            output_file = os.path.join(OUTPUT_DIR, f"seasonality_analysis_{year}.csv")
            df.to_csv(output_file, index=False)
            print(f"\nSaved to: {output_file}")

            # Plot if available
            if PLOTTING_AVAILABLE:
                self._plot_seasonality(df, year)

            return df

        return pd.DataFrame()

    def _plot_trade_balance(self, df: pd.DataFrame):
        """Plot trade balance over time."""
        if not PLOTTING_AVAILABLE:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        years = df["fetch_year"]
        exports = df.get("exports", pd.Series([0] * len(df)))
        imports = df.get("imports", pd.Series([0] * len(df)))
        balance = df.get("trade_balance", pd.Series([0] * len(df)))

        x = range(len(years))
        width = 0.35

        ax.bar([i - width/2 for i in x], exports/1e9, width, label="US Exports to India", color="green")
        ax.bar([i + width/2 for i in x], imports/1e9, width, label="US Imports from India", color="red")
        ax.plot(x, balance/1e9, "b-o", label="Trade Balance", linewidth=2)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)

        ax.set_xlabel("Year")
        ax.set_ylabel("Value (Billions USD)")
        ax.set_title("US-India Trade Balance Over Time")
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()

        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, "trade_balance_chart.png")
        plt.savefig(output_file, dpi=150)
        print(f"\nChart saved to: {output_file}")
        plt.close()

    def _plot_seasonality(self, df: pd.DataFrame, year: str):
        """Plot monthly trade seasonality."""
        if not PLOTTING_AVAILABLE:
            return

        if "ALL_VAL_MO" not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        for direction in df["direction"].unique():
            subset = df[df["direction"] == direction]
            monthly_values = subset.groupby("month")["ALL_VAL_MO"].sum() / 1e9
            ax.plot(monthly_values.index, monthly_values.values, "-o", label=direction.capitalize())

        ax.set_xlabel("Month")
        ax.set_ylabel("Value (Billions USD)")
        ax.set_title(f"US-India Trade Seasonality ({year})")
        ax.set_xticks(range(1, 13))
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(OUTPUT_DIR, f"seasonality_chart_{year}.png")
        plt.savefig(output_file, dpi=150)
        print(f"\nChart saved to: {output_file}")
        plt.close()


def load_local_csv(file_path: str) -> pd.DataFrame:
    """
    Option 3: Load data from local CSV file (Kaggle/Ministry downloads).

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with loaded data
    """
    print(f"\nLoading local CSV: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records with columns: {list(df.columns)}")
    return df


def main():
    """Main entry point for the trade analysis tool."""
    parser = argparse.ArgumentParser(
        description="India-USA Trade Data Analysis Tool"
    )
    parser.add_argument(
        "--source",
        choices=["census", "comtrade", "local"],
        default="census",
        help="Data source to use (default: census)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Start year for multi-year analysis (default: 2019)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for multi-year analysis (default: 2023)"
    )
    parser.add_argument(
        "--analysis",
        choices=["all", "balance", "commodity", "seasonality"],
        default="all",
        help="Type of analysis to run (default: all)"
    )
    parser.add_argument(
        "--local-file",
        type=str,
        help="Path to local CSV file (for --source local)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  INDIA-USA BILATERAL TRADE DATA ANALYSIS")
    print("=" * 70)
    print(f"  Data Source: {args.source.upper()}")
    print(f"  Analysis Period: {args.start_year} - {args.end_year}")
    print(f"  Analysis Type: {args.analysis.upper()}")
    print("=" * 70)

    if args.source == "local":
        if not args.local_file:
            print("Error: --local-file required when using --source local")
            sys.exit(1)
        df = load_local_csv(args.local_file)
        print("\nLocal CSV loaded. Use your own analysis code on the DataFrame.")
        return

    try:
        analyzer = TradeAnalyzer(data_source=args.source)

        # Fetch multi-year data
        data = analyzer.fetch_multi_year_data(
            start_year=args.start_year,
            end_year=args.end_year
        )

        if data.empty:
            print("\nNo data retrieved. Check your API key and network connection.")
            return

        # Determine the most recent year with data
        if "fetch_year" in data.columns:
            latest_year = data["fetch_year"].astype(str).max()
        else:
            latest_year = str(args.end_year)

        # Run requested analyses
        if args.analysis in ["all", "balance"]:
            analyzer.analyze_trade_balance()

        if args.analysis in ["all", "commodity"]:
            analyzer.analyze_commodity_shift(year=latest_year)

        if args.analysis in ["all", "seasonality"]:
            analyzer.analyze_seasonality(year=latest_year)

        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print(f"  Results saved to: {OUTPUT_DIR}/")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

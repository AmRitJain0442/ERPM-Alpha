"""
Commodity Shift Analysis - Track how specific sectors changed over time.

This analysis tracks the "China tariff effect" - how India's exports to the US
in key sectors (especially electronics) shifted after 2018 tariffs on China.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from census_api_fetcher import CensusTradeAPI
from config import OUTPUT_DIR, HS_CODES

def analyze_commodity_shift_over_time(
    start_year: int = 2015,
    end_year: int = 2024,
    hs_codes: list = None
):
    """
    Analyze how specific commodities changed over multiple years.

    Args:
        start_year: Starting year
        end_year: Ending year
        hs_codes: List of HS codes to track
    """
    if hs_codes is None:
        # Key sectors to track
        hs_codes = ["85", "84", "30", "71", "29", "62"]

    api = CensusTradeAPI()

    print("=" * 70)
    print("  COMMODITY SHIFT ANALYSIS (Multi-Year)")
    print("  Tracking India's export growth to USA by sector")
    print("=" * 70)

    all_data = []

    for year in range(start_year, end_year + 1):
        print(f"\nFetching {year}...", end=" ")
        try:
            df = api.fetch_commodity_breakdown(
                year=str(year),
                direction="imports",  # US imports from India
                hs_codes=hs_codes
            )
            if not df.empty:
                all_data.append(df)
                print(f"OK ({len(df)} commodities)")
            else:
                print("No data")
        except Exception as e:
            print(f"Failed: {e}")

    if not all_data:
        print("\nNo data retrieved.")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Create pivot table: Year vs Commodity
    pivot = combined.pivot_table(
        index="fetch_year",
        columns="HS_CODE",
        values="ALL_VAL_YR",
        aggfunc="sum"
    )

    # Add commodity descriptions
    hs_descriptions = {
        "85": "Electronics",
        "84": "Machinery",
        "30": "Pharmaceuticals",
        "71": "Gems & Jewelry",
        "29": "Organic Chemicals",
        "62": "Apparel",
        "52": "Cotton",
        "61": "Knitted Apparel",
    }

    pivot.columns = [hs_descriptions.get(c, c) for c in pivot.columns]

    # Convert to billions
    pivot_billions = pivot / 1e9

    print("\n" + "=" * 70)
    print("  US IMPORTS FROM INDIA BY SECTOR (Billions USD)")
    print("=" * 70)
    print(pivot_billions.round(2).to_string())

    # Calculate growth rates (2018 vs 2024 - post tariff era)
    if 2018 in pivot_billions.index and 2024 in pivot_billions.index:
        print("\n" + "=" * 70)
        print("  GROWTH ANALYSIS: 2018 vs 2024 (Post-China Tariff Era)")
        print("=" * 70)

        growth = pd.DataFrame({
            "2018 (B)": pivot_billions.loc[2018],
            "2024 (B)": pivot_billions.loc[2024],
        })
        growth["Growth (B)"] = growth["2024 (B)"] - growth["2018 (B)"]
        growth["Growth %"] = ((growth["2024 (B)"] / growth["2018 (B)"]) - 1) * 100
        growth = growth.sort_values("Growth %", ascending=False)

        print(growth.round(1).to_string())

    # Save data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "commodity_shift_multiyear.csv")
    pivot_billions.to_csv(output_file)
    print(f"\nSaved to: {output_file}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Reset index to make year a column for proper plotting
    plot_data = pivot_billions.reset_index()
    years = plot_data["fetch_year"].astype(int).values

    # Plot 1: Absolute values over time
    ax1 = axes[0]
    for col in pivot_billions.columns:
        ax1.plot(years, plot_data[col].values, marker='o', linewidth=2, label=col)
    ax1.set_title("US Imports from India by Sector (2015-2024)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Value (Billions USD)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=2018, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(2018.1, ax1.get_ylim()[1] * 0.95, 'China\nTariffs', fontsize=8, color='red')
    ax1.set_xticks(years)

    # Plot 2: Indexed growth (2015 = 100)
    ax2 = axes[1]
    indexed = (pivot_billions / pivot_billions.iloc[0]) * 100
    indexed_plot = indexed.reset_index()
    for col in indexed.columns:
        ax2.plot(years, indexed_plot[col].values, marker='o', linewidth=2, label=col)
    ax2.set_title("Indexed Growth (2015 = 100)")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Index (2015 = 100)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=2018, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axhline(y=100, color='black', linestyle='-', alpha=0.3)
    ax2.set_xticks(years)

    plt.tight_layout()
    chart_file = os.path.join(OUTPUT_DIR, "commodity_shift_chart.png")
    plt.savefig(chart_file, dpi=150)
    print(f"Chart saved to: {chart_file}")
    plt.close()

    return pivot_billions


if __name__ == "__main__":
    analyze_commodity_shift_over_time()

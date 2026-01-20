"""
UN Comtrade Data Fetcher via World Trade Data Library
Fetches India-USA trade data using the world_trade_data Python wrapper.

This approach uses standard ISO country codes (usa, ind) instead of obscure government codes.
Install: pip install world_trade_data
"""

import pandas as pd
from typing import Optional, Literal, List

try:
    import world_trade_data as wits
    WITS_AVAILABLE = True
except ImportError:
    WITS_AVAILABLE = False
    print("Warning: world_trade_data not installed. Run: pip install world_trade_data")


class ComtradeTradeAPI:
    """Fetches trade data from UN Comtrade via World Trade Data library."""

    # Trade indicators
    INDICATOR_IMPORT_VALUE = "MPRT-TRD-VL"  # Import Trade Value
    INDICATOR_EXPORT_VALUE = "XPRT-TRD-VL"  # Export Trade Value
    INDICATOR_IMPORT_SHARE = "MPRT-PRTNR-SHR"  # Import Partner Share
    INDICATOR_EXPORT_SHARE = "XPRT-PRTNR-SHR"  # Export Partner Share

    def __init__(self):
        """Initialize the Comtrade API client."""
        if not WITS_AVAILABLE:
            raise ImportError(
                "world_trade_data library not available. "
                "Install with: pip install world_trade_data"
            )

    def fetch_usa_imports_from_india(
        self,
        year: str,
        product: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch US imports from India.

        Args:
            year: Year to fetch (e.g., "2022")
            product: Optional product code (HS code)

        Returns:
            DataFrame with import data
        """
        params = {
            "indicator": self.INDICATOR_IMPORT_VALUE,
            "reporter": "usa",
            "partner": "ind",
            "year": year,
            "datasource": "tradestats-trade"
        }

        if product:
            params["product"] = product

        try:
            data = wits.get_indicator(**params)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_usa_exports_to_india(
        self,
        year: str,
        product: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch US exports to India.

        Args:
            year: Year to fetch (e.g., "2022")
            product: Optional product code (HS code)

        Returns:
            DataFrame with export data
        """
        params = {
            "indicator": self.INDICATOR_EXPORT_VALUE,
            "reporter": "usa",
            "partner": "ind",
            "year": year,
            "datasource": "tradestats-trade"
        }

        if product:
            params["product"] = product

        try:
            data = wits.get_indicator(**params)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_india_imports_from_usa(
        self,
        year: str,
        product: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch India's imports from USA (reverse perspective).

        Args:
            year: Year to fetch
            product: Optional product code

        Returns:
            DataFrame with import data
        """
        params = {
            "indicator": self.INDICATOR_IMPORT_VALUE,
            "reporter": "ind",
            "partner": "usa",
            "year": year,
            "datasource": "tradestats-trade"
        }

        if product:
            params["product"] = product

        try:
            data = wits.get_indicator(**params)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_india_exports_to_usa(
        self,
        year: str,
        product: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch India's exports to USA (reverse perspective).

        Args:
            year: Year to fetch
            product: Optional product code

        Returns:
            DataFrame with export data
        """
        params = {
            "indicator": self.INDICATOR_EXPORT_VALUE,
            "reporter": "ind",
            "partner": "usa",
            "year": year,
            "datasource": "tradestats-trade"
        }

        if product:
            params["product"] = product

        try:
            data = wits.get_indicator(**params)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_bilateral_trade_summary(
        self,
        years: List[str]
    ) -> pd.DataFrame:
        """
        Fetch comprehensive bilateral trade summary for multiple years.

        Args:
            years: List of years to fetch (e.g., ["2020", "2021", "2022"])

        Returns:
            DataFrame with bilateral trade data
        """
        all_data = []

        for year in years:
            print(f"Fetching bilateral trade data for {year}...")

            # US perspective
            try:
                usa_imports = self.fetch_usa_imports_from_india(year)
                if not usa_imports.empty:
                    usa_imports["perspective"] = "usa_reporter"
                    usa_imports["flow"] = "usa_imports_from_india"
                    all_data.append(usa_imports)
            except Exception as e:
                print(f"  Warning: {e}")

            try:
                usa_exports = self.fetch_usa_exports_to_india(year)
                if not usa_exports.empty:
                    usa_exports["perspective"] = "usa_reporter"
                    usa_exports["flow"] = "usa_exports_to_india"
                    all_data.append(usa_exports)
            except Exception as e:
                print(f"  Warning: {e}")

            # India perspective
            try:
                india_imports = self.fetch_india_imports_from_usa(year)
                if not india_imports.empty:
                    india_imports["perspective"] = "india_reporter"
                    india_imports["flow"] = "india_imports_from_usa"
                    all_data.append(india_imports)
            except Exception as e:
                print(f"  Warning: {e}")

            try:
                india_exports = self.fetch_india_exports_to_usa(year)
                if not india_exports.empty:
                    india_exports["perspective"] = "india_reporter"
                    india_exports["flow"] = "india_exports_to_usa"
                    all_data.append(india_exports)
            except Exception as e:
                print(f"  Warning: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def fetch_product_level_trade(
        self,
        year: str,
        products: Optional[List[str]] = None,
        direction: Literal["import", "export"] = "import"
    ) -> pd.DataFrame:
        """
        Fetch product-level trade data.

        Args:
            year: Year to fetch
            products: List of product codes (HS chapters). If None, uses common categories.
            direction: "import" or "export"

        Returns:
            DataFrame with product-level trade data
        """
        if products is None:
            # Common HS code chapters for India-US trade
            products = [
                "85",  # Electronics
                "84",  # Machinery
                "29",  # Organic chemicals
                "30",  # Pharmaceuticals
                "71",  # Gems and jewelry
                "52",  # Cotton
                "62",  # Apparel
            ]

        all_data = []
        indicator = self.INDICATOR_IMPORT_VALUE if direction == "import" else self.INDICATOR_EXPORT_VALUE

        for product in products:
            print(f"  Fetching product {product}...")
            try:
                data = wits.get_indicator(
                    indicator=indicator,
                    reporter="usa",
                    partner="ind",
                    year=year,
                    product=product,
                    datasource="tradestats-trade"
                )
                if not data.empty:
                    data["hs_chapter"] = product
                    all_data.append(data)
            except Exception as e:
                print(f"    Warning: Could not fetch product {product}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


def main():
    """Example usage of the Comtrade API via world_trade_data."""
    print("=" * 60)
    print("UN Comtrade - India-USA Trade Data Fetcher")
    print("(via world_trade_data library)")
    print("=" * 60)

    if not WITS_AVAILABLE:
        print("\nError: world_trade_data library not installed.")
        print("Install with: pip install world_trade_data")
        return

    api = ComtradeTradeAPI()

    # Example 1: Fetch US imports from India for 2022
    print("\n1. Fetching US imports from India (2022)...")
    try:
        usa_imports = api.fetch_usa_imports_from_india(year="2022")
        if not usa_imports.empty:
            print(usa_imports.head().to_string())
        else:
            print("No data returned")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Fetch US exports to India for 2022
    print("\n2. Fetching US exports to India (2022)...")
    try:
        usa_exports = api.fetch_usa_exports_to_india(year="2022")
        if not usa_exports.empty:
            print(usa_exports.head().to_string())
        else:
            print("No data returned")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Fetch bilateral summary for multiple years
    print("\n3. Fetching bilateral trade summary (2020-2022)...")
    try:
        summary = api.fetch_bilateral_trade_summary(years=["2020", "2021", "2022"])
        if not summary.empty:
            print(summary.to_string())
            summary.to_csv("india_usa_comtrade_summary.csv", index=False)
            print("\nSaved to india_usa_comtrade_summary.csv")
        else:
            print("No data returned")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

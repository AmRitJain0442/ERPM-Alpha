"""
US Census Bureau International Trade API Fetcher
Fetches India-USA trade data (exports and imports) using the official Census Bureau API.

Endpoint: https://api.census.gov/data/timeseries/intltrade/
India Country Code: 5330 (Schedule C code)
API Key: Get one at https://api.census.gov/data/key_signup.html
"""

import requests
import pandas as pd
from typing import Optional, Literal
from config import CENSUS_API_KEY


class CensusTradeAPI:
    """Fetches trade data from US Census Bureau International Trade API."""

    BASE_URL_EXPORTS = "https://api.census.gov/data/timeseries/intltrade/exports/hs"
    BASE_URL_IMPORTS = "https://api.census.gov/data/timeseries/intltrade/imports/hs"

    # Country codes (Schedule C)
    INDIA_CODE = "5330"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Census Trade API client.

        Args:
            api_key: Census Bureau API key. If None, uses config value.
        """
        self.api_key = api_key or CENSUS_API_KEY

    def fetch_exports_to_india(
        self,
        year: str,
        month: str = "12",
        commodity_code: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch US exports to India.

        Args:
            year: Year to fetch (e.g., "2023")
            month: Month to fetch (1-12). Use "12" for year-to-date data.
            commodity_code: Optional HS code to filter by commodity (e.g., "85" for electronics)

        Returns:
            DataFrame with export data
        """
        return self._fetch_data(
            direction="exports",
            year=year,
            month=month,
            commodity_code=commodity_code
        )

    def fetch_imports_from_india(
        self,
        year: str,
        month: str = "12",
        commodity_code: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch US imports from India.

        Args:
            year: Year to fetch (e.g., "2023")
            month: Month to fetch (1-12). Use "12" for year-to-date data.
            commodity_code: Optional HS code to filter by commodity

        Returns:
            DataFrame with import data
        """
        return self._fetch_data(
            direction="imports",
            year=year,
            month=month,
            commodity_code=commodity_code
        )

    def _fetch_data(
        self,
        direction: Literal["exports", "imports"],
        year: str,
        month: str,
        commodity_code: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Internal method to fetch trade data.

        Args:
            direction: "exports" or "imports"
            year: Year to fetch
            month: Month to fetch
            commodity_code: Optional HS code filter

        Returns:
            DataFrame with trade data
        """
        base_url = self.BASE_URL_EXPORTS if direction == "exports" else self.BASE_URL_IMPORTS

        # Define variables to fetch - exports uses ALL_VAL, imports uses GEN_VAL
        if direction == "exports":
            get_vars = "CTY_CODE,CTY_NAME,ALL_VAL_YR,ALL_VAL_MO"
        else:
            get_vars = "CTY_CODE,CTY_NAME,GEN_VAL_YR,GEN_VAL_MO"

        if commodity_code:
            get_vars += ",I_COMMODITY" if direction == "imports" else ",E_COMMODITY"

        params = {
            "get": get_vars,
            "YEAR": year,
            "MONTH": month,
            "CTY_CODE": self.INDIA_CODE,
        }

        if commodity_code:
            params["I_COMMODITY"] = commodity_code if direction == "imports" else None
            params["E_COMMODITY"] = commodity_code if direction == "exports" else None
            # Remove None entries
            params = {k: v for k, v in params.items() if v is not None}

        if self.api_key:
            params["key"] = self.api_key

        response = requests.get(base_url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()

            # Validate response format
            if not isinstance(data, list):
                raise Exception(f"Unexpected response format: {type(data)}")

            if len(data) < 2:
                return pd.DataFrame()

            # Ensure headers and rows are proper lists
            headers = list(data[0]) if data[0] else []
            rows = [list(row) if row else [] for row in data[1:]]

            if not headers or not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=headers)

            # Normalize column names between exports (ALL_VAL) and imports (GEN_VAL)
            rename_map = {
                "GEN_VAL_YR": "ALL_VAL_YR",
                "GEN_VAL_MO": "ALL_VAL_MO",
                "I_COMMODITY": "COMMODITY",
                "E_COMMODITY": "COMMODITY",
            }
            df = df.rename(columns=rename_map)

            # Convert value columns to numeric
            for col in ["ALL_VAL_YR", "ALL_VAL_MO"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["direction"] = direction
            df["fetch_year"] = year
            df["fetch_month"] = month
            return df
        else:
            raise Exception(f"API Error ({response.status_code}): {response.text}")

    def fetch_yearly_trade_summary(
        self,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Fetch yearly trade summary for multiple years.

        Args:
            start_year: Starting year (e.g., 2015)
            end_year: Ending year (e.g., 2023)

        Returns:
            DataFrame with yearly trade data for both exports and imports
        """
        all_data = []

        for year in range(start_year, end_year + 1):
            year_str = str(year)
            print(f"Fetching data for {year}...")

            try:
                exports = self.fetch_exports_to_india(year=year_str)
                if not exports.empty:
                    all_data.append(exports)
            except Exception as e:
                print(f"  Warning: Could not fetch exports for {year}: {e}")

            try:
                imports = self.fetch_imports_from_india(year=year_str)
                if not imports.empty:
                    all_data.append(imports)
            except Exception as e:
                print(f"  Warning: Could not fetch imports for {year}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def fetch_commodity_breakdown(
        self,
        year: str,
        direction: Literal["exports", "imports"] = "exports",
        hs_codes: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Fetch trade data broken down by commodity (HS 2-digit level).

        Args:
            year: Year to fetch
            direction: "exports" or "imports"
            hs_codes: List of HS codes to filter. If None, returns all commodities.

        Returns:
            DataFrame with commodity-level trade data
        """
        base_url = self.BASE_URL_EXPORTS if direction == "exports" else self.BASE_URL_IMPORTS

        # Use appropriate variable names for exports vs imports
        if direction == "exports":
            get_vars = "CTY_CODE,CTY_NAME,ALL_VAL_YR,E_COMMODITY,E_COMMODITY_LDESC"
        else:
            get_vars = "CTY_CODE,CTY_NAME,GEN_VAL_YR,I_COMMODITY,I_COMMODITY_LDESC"

        params = {
            "get": get_vars,
            "YEAR": year,
            "MONTH": "12",
            "CTY_CODE": self.INDIA_CODE,
            "COMM_LVL": "HS2",  # Get HS 2-digit level breakdown
        }

        if self.api_key:
            params["key"] = self.api_key

        response = requests.get(base_url, params=params, timeout=60)

        if response.status_code == 204:
            # No content - data not available for this period
            return pd.DataFrame()

        if response.status_code != 200:
            raise Exception(f"API Error ({response.status_code}): {response.text}")

        data = response.json()
        if not isinstance(data, list) or len(data) < 2:
            return pd.DataFrame()

        headers = list(data[0])
        rows = [list(row) for row in data[1:]]
        df = pd.DataFrame(rows, columns=headers)

        # Normalize column names
        rename_map = {
            "GEN_VAL_YR": "ALL_VAL_YR",
            "I_COMMODITY": "HS_CODE",
            "E_COMMODITY": "HS_CODE",
            "I_COMMODITY_LDESC": "COMMODITY_DESC",
            "E_COMMODITY_LDESC": "COMMODITY_DESC",
        }
        df = df.rename(columns=rename_map)

        # Convert value to numeric
        if "ALL_VAL_YR" in df.columns:
            df["ALL_VAL_YR"] = pd.to_numeric(df["ALL_VAL_YR"], errors="coerce")

        df["direction"] = direction
        df["fetch_year"] = year

        # Filter by HS codes if specified
        if hs_codes and "HS_CODE" in df.columns:
            df = df[df["HS_CODE"].isin(hs_codes)]

        # Sort by value descending
        if "ALL_VAL_YR" in df.columns:
            df = df.sort_values("ALL_VAL_YR", ascending=False)

        return df.reset_index(drop=True)


def main():
    """Example usage of the Census Trade API."""
    print("=" * 60)
    print("US Census Bureau - India-USA Trade Data Fetcher")
    print("=" * 60)

    api = CensusTradeAPI()

    # Example 1: Fetch 2023 exports to India
    print("\n1. Fetching US exports to India (2023)...")
    try:
        exports_2023 = api.fetch_exports_to_india(year="2023")
        if not exports_2023.empty:
            print(exports_2023.to_string())
        else:
            print("No data returned")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Fetch 2023 imports from India
    print("\n2. Fetching US imports from India (2023)...")
    try:
        imports_2023 = api.fetch_imports_from_india(year="2023")
        if not imports_2023.empty:
            print(imports_2023.to_string())
        else:
            print("No data returned")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Fetch multi-year summary
    print("\n3. Fetching 5-year trade summary (2019-2023)...")
    try:
        summary = api.fetch_yearly_trade_summary(start_year=2019, end_year=2023)
        if not summary.empty:
            print(summary.to_string())
            # Save to CSV
            summary.to_csv("india_usa_trade_summary.csv", index=False)
            print("\nSaved to india_usa_trade_summary.csv")
        else:
            print("No data returned")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

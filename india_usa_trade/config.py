"""
Configuration for India-USA Trade Data Fetchers

Set your API keys and other configuration options here.
"""

import os

# =============================================================================
# US CENSUS BUREAU API
# =============================================================================
# Get your free API key at: https://api.census.gov/data/key_signup.html
# The API works without a key but has lower rate limits
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")

# =============================================================================
# COUNTRY CODES
# =============================================================================
# US Census Bureau uses "Schedule C" codes
CENSUS_COUNTRY_CODES = {
    "india": "5330",
    "china": "5700",
    "japan": "5880",
    "germany": "4280",
    "uk": "4120",
    "canada": "1220",
    "mexico": "2010",
}

# ISO 3-letter codes (used by UN Comtrade / world_trade_data)
ISO_COUNTRY_CODES = {
    "india": "ind",
    "usa": "usa",
    "china": "chn",
    "japan": "jpn",
    "germany": "deu",
    "uk": "gbr",
}

# =============================================================================
# HS CODES (HARMONIZED SYSTEM)
# =============================================================================
# Common HS code chapters for India-USA trade analysis
HS_CODES = {
    "85": "Electrical machinery and equipment",
    "84": "Nuclear reactors, boilers, machinery",
    "29": "Organic chemicals",
    "30": "Pharmaceutical products",
    "71": "Natural/cultured pearls, precious stones, jewelry",
    "52": "Cotton",
    "62": "Apparel and clothing (not knitted)",
    "61": "Apparel and clothing (knitted)",
    "03": "Fish and crustaceans",
    "09": "Coffee, tea, mate and spices",
    "27": "Mineral fuels, oils",
    "72": "Iron and steel",
    "87": "Vehicles (not railway)",
    "90": "Optical, photographic, medical instruments",
    "39": "Plastics and articles thereof",
}

# =============================================================================
# DATA SOURCES
# =============================================================================
DATA_SOURCES = {
    "census_exports": "https://api.census.gov/data/timeseries/intltrade/exports",
    "census_imports": "https://api.census.gov/data/timeseries/intltrade/imports",
    "tradestat_india": "https://tradestat.commerce.gov.in/",
    "kaggle_india_trade": "https://www.kaggle.com/datasets?search=india+trade",
}

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
OUTPUT_DIR = "output"
DEFAULT_DATE_RANGE = {
    "start_year": 2015,
    "end_year": 2024,
}

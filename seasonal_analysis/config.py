"""Configuration for Seasonal Pattern Analysis"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "gold_standard")
TRADE_DIR = os.path.join(PROJECT_DIR, "india_usa_trade", "output")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data files
SEASONALITY_DATA = os.path.join(TRADE_DIR, "seasonality_analysis_2024.csv")
EXPORT_COMMODITY = os.path.join(DATA_DIR, "india_commerce", "TradeStat-Eidb-Export-Commodity-wise.csv")
IMPORT_COMMODITY = os.path.join(DATA_DIR, "india_commerce", "TradeStat-Eidb-Import-Commodity-wise.csv")
COMMODITY_SHIFT = os.path.join(TRADE_DIR, "commodity_shift_multiyear.csv")
TRADE_BALANCE = os.path.join(TRADE_DIR, "trade_balance_analysis.csv")

# Visualization settings
FIGURE_DPI = 150
FIGURE_SIZE = (14, 10)
COLOR_PALETTE = {
    'exports': '#2ecc71',
    'imports': '#e74c3c',
    'balance': '#3498db',
    'highlight': '#f39c12'
}

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

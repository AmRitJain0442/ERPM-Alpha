"""
Configuration for EGARCH + XGBoost Hybrid Volatility Model

This module contains all configuration parameters for the hybrid model
that combines EGARCH volatility forecasting with XGBoost news correction.
"""

import os

# =============================================================================
# DATA PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATHS = {
    "merged_training": os.path.join(BASE_DIR, "Phase-B", "merged_training_data.csv"),
    "exchange_rates": os.path.join(BASE_DIR, "combined_goldstein_exchange_rates.csv"),
    "fred_data": os.path.join(BASE_DIR, "data", "gold_standard", "fred", "fred_wide_format_20251230_021943.csv"),
    "usd_inr": os.path.join(BASE_DIR, "usd_inr_exchange_rates_1year.csv"),
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# =============================================================================
# EGARCH MODEL PARAMETERS
# =============================================================================
EGARCH_CONFIG = {
    "p": 1,              # GARCH lag order
    "q": 1,              # ARCH lag order
    "o": 1,              # Asymmetry order (captures leverage effect)
    "dist": "skewt",     # Distribution: 'normal', 't', 'skewt', 'ged'
    "mean": "Constant",  # Mean model: 'Constant', 'Zero', 'AR', 'ARX'
    "vol": "EGARCH",     # Volatility model: 'GARCH', 'EGARCH', 'GJR-GARCH'
    "rescale": True,     # Rescale returns for numerical stability
}

# Alternative: GJR-GARCH (also captures asymmetry)
GJR_GARCH_CONFIG = {
    "p": 1,
    "q": 1,
    "o": 1,
    "dist": "skewt",
    "mean": "Constant",
    "vol": "Garch",
    "power": 2.0,
}

# =============================================================================
# XGBOOST HYBRID MODEL PARAMETERS
# =============================================================================
XGBOOST_CONFIG = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
# Core GDELT features for news shock modeling
GDELT_FEATURES = [
    "Tone_Economy",
    "Tone_Conflict",
    "Tone_Policy",
    "Tone_Corporate",
    "Tone_Overall",
    "Goldstein_Weighted",
    "Goldstein_Avg",
    "Volume_Spike",
    "Volume_Spike_Economy",
    "Volume_Spike_Conflict",
]

# Derived features we will compute
DERIVED_FEATURES = [
    "GARCH_Vol",           # Conditional volatility from EGARCH
    "Returns",             # Price returns
    "Realized_Vol",        # Realized volatility (squared returns)
    "Panic_Index",         # Composite panic indicator
    "Diff_Stability",      # Sentiment stability measure
    "News_Shock",          # Unexpected news component
]

# Macro features from FRED
MACRO_FEATURES = [
    "DGS10",               # US 10-Year Treasury (US10Y)
    "DFF",                 # Federal Funds Rate
    "DTWEXBGS",            # Trade Weighted Dollar Index
    "DCOILWTICO",          # WTI Crude Oil Price
]

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAIN_CONFIG = {
    "train_ratio": 0.8,            # Train/test split ratio
    "validation_ratio": 0.1,       # Validation set (from training)
    "forecast_horizon": 1,         # Days ahead to predict
    "rolling_window": 20,          # Rolling window for volatility
    "min_train_size": 100,         # Minimum training samples
}

# =============================================================================
# ASYMMETRY ANALYSIS
# =============================================================================
# Thresholds for panic vs relief classification
ASYMMETRY_CONFIG = {
    "panic_threshold": -2.0,       # Tone below this = panic
    "relief_threshold": 2.0,       # Tone above this = relief
    "vol_spike_threshold": 1.5,    # Volatility spike multiplier
    "news_shock_window": 5,        # Days to measure shock persistence
}

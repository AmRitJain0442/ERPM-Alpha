"""
Configuration for LLM-as-Analyst USD/INR Prediction System.
All constants, API keys, regime definitions, model hyperparameters, feature lists.
"""

import os

# ─── Data Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Super_Master_Dataset.csv")
INDIA_NEWS_PATH = os.path.join(BASE_DIR, "india_news_combined_sorted.csv")
USA_NEWS_PATH = os.path.join(BASE_DIR, "usa_news_combined_sorted.csv")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# ─── Gemini API ───────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.3
GEMINI_TOP_P = 0.85
GEMINI_MAX_TOKENS = 1024
API_DELAY_SECONDS = 6
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2

# ─── Regime Definitions ──────────────────────────────────────────────────────
REGIMES = [
    "CALM_CARRY",
    "TRENDING_APPRECIATION",
    "TRENDING_DEPRECIATION",
    "HIGH_VOLATILITY",
    "CRISIS_STRESS",
]

REGIME_DESCRIPTIONS = {
    "CALM_CARRY": "Low volatility, range-bound, carry trade favorable",
    "TRENDING_APPRECIATION": "INR strengthening trend (USD/INR falling)",
    "TRENDING_DEPRECIATION": "INR weakening trend (USD/INR rising)",
    "HIGH_VOLATILITY": "Elevated volatility, uncertain direction",
    "CRISIS_STRESS": "Extreme volatility, potential tail events",
}

DEFAULT_REGIME = "CALM_CARRY"

# ─── Regime Detection Thresholds ─────────────────────────────────────────────
REGIME_THRESHOLDS = {
    "vol_low": 0.7,          # recent_vol / long_vol < 0.7 → low vol
    "vol_high": 1.5,         # recent_vol / long_vol > 1.5 → high vol
    "vol_crisis": 2.5,       # > 2.5 → crisis
    "trend_threshold": 0.003, # |MA5 - MA20| / MA20 > 0.3% → trending
    "panic_threshold": 0.7,   # IN_Panic_Index > 0.7 → crisis signal
    "tone_extreme": -3.0,     # Avg tone below this → crisis signal
}

# ─── Technical Indicator Parameters ──────────────────────────────────────────
MA_SHORT = 5
MA_LONG = 20
RSI_PERIOD = 14
ZSCORE_WINDOW = 20
MOMENTUM_WINDOW = 5
VOLATILITY_WINDOW = 20
VOLATILITY_LONG_WINDOW = 60

# ─── Feature Lists ───────────────────────────────────────────────────────────
PRICE_FEATURES = ["INR", "OIL", "GOLD", "US10Y", "DXY"]

MACRO_FEATURES = ["OIL", "GOLD", "US10Y", "DXY"]

GDELT_FEATURES = [
    "IN_Avg_Tone", "IN_Avg_Stability", "IN_Total_Mentions", "IN_Panic_Index",
    "US_Avg_Tone", "US_Avg_Stability", "US_Total_Mentions", "US_Panic_Index",
    "Diff_Stability", "Diff_Tone",
]

TECHNICAL_FEATURES = [
    "MA_5", "MA_20", "MA_momentum", "RSI", "INR_zscore",
    "INR_return", "realized_vol", "vol_ratio",
]

SIMULATION_FEATURE_NAMES = [
    "sim_direction_signal", "sim_appreciation_w", "sim_depreciation_w",
    "sim_neutral_w", "sim_magnitude", "sim_consensus_strength",
    "sim_entropy", "sim_uncertainty", "sim_avg_confidence", "sim_success_rate",
    "sim_technical_signal", "sim_fundamental_signal", "sim_carry_signal",
    "sim_sentiment_signal", "sim_flow_signal", "sim_quant_signal",
]

LLM_FEATURE_NAMES = [
    # Regime classifier (Task A) — one-hot
    "regime_CALM_CARRY", "regime_TRENDING_APPRECIATION",
    "regime_TRENDING_DEPRECIATION", "regime_HIGH_VOLATILITY",
    "regime_CRISIS_STRESS", "regime_confidence",
    # Event impact (Task B)
    "event_impact_mean", "event_impact_max", "event_impact_min",
    "event_count_positive", "event_count_negative",
    # Causal chains (Task C)
    "chain_count", "chain_avg_strength", "chain_max_strength",
    # Risk signals (Task D)
    "risk_flag_count", "risk_max_severity", "risk_avg_severity",
]

# ─── Model Hyperparameters Per Regime ────────────────────────────────────────
MODEL_CONFIG = {
    "CALM_CARRY": {
        "primary": "ridge",
        "ridge_alpha": 1.0,
    },
    "TRENDING_APPRECIATION": {
        "primary": "xgboost",
        "xgb_n_estimators": 200,
        "xgb_max_depth": 4,
        "xgb_learning_rate": 0.05,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
    },
    "TRENDING_DEPRECIATION": {
        "primary": "xgboost",
        "xgb_n_estimators": 200,
        "xgb_max_depth": 4,
        "xgb_learning_rate": 0.05,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
    },
    "HIGH_VOLATILITY": {
        "primary": "xgboost_egarch",
        "xgb_n_estimators": 300,
        "xgb_max_depth": 5,
        "xgb_learning_rate": 0.03,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
        "egarch_p": 1,
        "egarch_q": 1,
        "egarch_o": 1,
        "egarch_dist": "skewt",
    },
    "CRISIS_STRESS": {
        "primary": "xgboost_egarch",
        "xgb_n_estimators": 300,
        "xgb_max_depth": 5,
        "xgb_learning_rate": 0.03,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
        "egarch_p": 1,
        "egarch_q": 1,
        "egarch_o": 1,
        "egarch_dist": "skewt",
        "ci_multiplier": 1.5,  # Wider confidence intervals
    },
}

# ─── Training Configuration ──────────────────────────────────────────────────
MIN_TRAIN_SIZE = 252        # ~1 year of trading days
EXPANDING_WINDOW_START = 504 # ~2 years warmup
REFIT_FREQUENCY = 21         # Refit models every ~1 month
WALK_FORWARD_STEP = 1        # Predict 1 day ahead

# ─── LLM Weight Bounds ───────────────────────────────────────────────────────
INITIAL_LLM_WEIGHT = 0.05   # Start at 5% (matching current optimal)
MIN_LLM_WEIGHT = 0.0
MAX_LLM_WEIGHT = 0.30
LLM_WEIGHT_ADJUSTMENT_RATE = 0.01  # How fast weights adapt

# ─── News Headlines ──────────────────────────────────────────────────────────
MAX_HEADLINES_PER_DAY = 40
HEADLINE_MIN_LENGTH = 10
HEADLINE_MAX_LENGTH = 200

# ─── Context Packet Lookback ─────────────────────────────────────────────────
CONTEXT_PRICE_HISTORY_DAYS = 20
CONTEXT_NEWS_LOOKBACK_DAYS = 1

# ─── Ollama Market Agent Simulation ──────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3"              # Any pulled model: llama3.2, mistral, phi3, etc.
OLLAMA_TIMEOUT = 120               # Seconds per agent call (Ollama queues internally — be generous)
OLLAMA_NUM_AGENTS = 30             # Total agents defined in market_agents.py
OLLAMA_MAX_WORKERS = 10            # ThreadPoolExecutor HTTP concurrency
OLLAMA_TEMPERATURE = 0.7           # Higher than Gemini — diversity of views is the point
INITIAL_AGENT_WEIGHT = 0.05        # Start agents at 5% influence (same as LLM tasks)
AGENT_TRIM_FRACTION = 0.1          # Trim top/bottom 10% outlier agents before aggregating
# NOTE: Ollama processes one request at a time (single GPU).
# total_timeout = OLLAMA_TIMEOUT * num_agents (sequential worst case).
# For 30 agents at 120s each = 60 min max. Use --quick-agents for 10 agents (~20 min).

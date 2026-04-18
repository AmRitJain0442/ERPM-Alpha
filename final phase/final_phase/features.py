"""
Feature engineering for the final-phase system.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data import GOLDSTEIN_COLS, MASTER_SENTIMENT_COLS, POLITICAL_COLS, THEMATIC_COLS
from .config import RunConfig


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def _range_position(series: pd.Series, window: int) -> pd.Series:
    low = series.rolling(window).min()
    high = series.rolling(window).max()
    return (series - low) / (high - low).replace(0, np.nan)


def build_feature_frame(df: pd.DataFrame, config: RunConfig) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Build model features and forward targets.

    Targets are shifted by horizon + embargo to reduce immediate-price anchoring.
    """
    out = df.copy()
    out = out.sort_values("Date").reset_index(drop=True)

    # Hidden raw return for volatility modeling / evaluation support.
    out["raw_return_1d"] = out["INR"].pct_change()

    # Technical features derived from price history, but no raw INR level is used directly.
    out["tech_return_3"] = _pct_change(out["INR"], 3)
    out["tech_return_5"] = _pct_change(out["INR"], 5)
    out["tech_return_10"] = _pct_change(out["INR"], 10)
    out["tech_return_20"] = _pct_change(out["INR"], 20)
    out["tech_vol_5"] = out["raw_return_1d"].rolling(5).std()
    out["tech_vol_10"] = out["raw_return_1d"].rolling(10).std()
    out["tech_vol_20"] = out["raw_return_1d"].rolling(20).std()
    ma_5 = out["INR"].rolling(5).mean()
    ma_20 = out["INR"].rolling(20).mean()
    ma_60 = out["INR"].rolling(60).mean()
    out["tech_trend_5_20"] = (ma_5 - ma_20) / ma_20.replace(0, np.nan)
    out["tech_trend_20_60"] = (ma_20 - ma_60) / ma_60.replace(0, np.nan)
    out["tech_zscore_20"] = _rolling_zscore(out["INR"], 20)
    out["tech_range_pos_20"] = _range_position(out["INR"], 20)
    out["tech_drawdown_20"] = out["INR"] / out["INR"].rolling(20).max() - 1.0
    out["tech_momentum_stability"] = out["tech_return_5"] / out["tech_vol_20"].replace(0, np.nan)

    technical_cols = [
        "tech_return_3",
        "tech_return_5",
        "tech_return_10",
        "tech_return_20",
        "tech_vol_5",
        "tech_vol_10",
        "tech_vol_20",
        "tech_trend_5_20",
        "tech_trend_20_60",
        "tech_zscore_20",
        "tech_range_pos_20",
        "tech_drawdown_20",
        "tech_momentum_stability",
    ]

    # Macro features.
    macro_cols: List[str] = []
    for col in ["OIL", "GOLD", "US10Y", "DXY"]:
        ret3 = f"{col.lower()}_ret_3"
        ret5 = f"{col.lower()}_ret_5"
        ret10 = f"{col.lower()}_ret_10"
        z20 = f"{col.lower()}_z_20"
        out[ret3] = _pct_change(out[col], 3)
        out[ret5] = _pct_change(out[col], 5)
        out[ret10] = _pct_change(out[col], 10)
        out[z20] = _rolling_zscore(out[col], 20)
        macro_cols.extend([ret3, ret5, ret10, z20])

    out["macro_oil_dxy_pressure"] = out["oil_ret_5"] + out["dxy_ret_5"]
    out["macro_rates_dollar_pressure"] = out["us10y_ret_5"] + out["dxy_ret_5"]
    out["macro_gold_risk_divergence"] = out["gold_ret_5"] - out["dxy_ret_5"]
    macro_cols.extend(
        [
            "macro_oil_dxy_pressure",
            "macro_rates_dollar_pressure",
            "macro_gold_risk_divergence",
        ]
    )

    # Master sentiment features.
    out["sent_in_tone_5d_mean"] = out["IN_Avg_Tone"].rolling(5).mean()
    out["sent_in_tone_10d_mean"] = out["IN_Avg_Tone"].rolling(10).mean()
    out["sent_us_tone_5d_mean"] = out["US_Avg_Tone"].rolling(5).mean()
    out["sent_in_panic_5d_mean"] = out["IN_Panic_Index"].rolling(5).mean()
    out["sent_us_panic_5d_mean"] = out["US_Panic_Index"].rolling(5).mean()
    out["sent_diff_tone_5d_mean"] = out["Diff_Tone"].rolling(5).mean()
    out["sent_diff_stability_5d_mean"] = out["Diff_Stability"].rolling(5).mean()
    out["sent_in_mentions_spike"] = out["IN_Total_Mentions"] / out["IN_Total_Mentions"].rolling(5).mean() - 1.0
    out["sent_us_mentions_spike"] = out["US_Total_Mentions"] / out["US_Total_Mentions"].rolling(5).mean() - 1.0
    out["sent_cross_stress"] = out["IN_Panic_Index"] - out["Diff_Stability"]
    master_sentiment_cols = [
        "sent_in_tone_5d_mean",
        "sent_in_tone_10d_mean",
        "sent_us_tone_5d_mean",
        "sent_in_panic_5d_mean",
        "sent_us_panic_5d_mean",
        "sent_diff_tone_5d_mean",
        "sent_diff_stability_5d_mean",
        "sent_in_mentions_spike",
        "sent_us_mentions_spike",
        "sent_cross_stress",
    ]

    # Bilateral Goldstein features.
    out["gold_india_5d_mean"] = out["India_Avg_Goldstein"].rolling(5).mean()
    out["gold_usa_5d_mean"] = out["USA_Avg_Goldstein"].rolling(5).mean()
    out["gold_diff_level"] = out["India_Avg_Goldstein"] - out["USA_Avg_Goldstein"]
    out["gold_combined_weighted_5d"] = out["Combined_Weighted_Avg"].rolling(5).mean()
    out["gold_geo_5d"] = out["Combined_Geometric_Mean"].rolling(5).mean()
    out["gold_event_pressure"] = (
        np.log1p(out["India_Event_Count"].fillna(0)) - np.log1p(out["USA_Event_Count"].fillna(0))
    )
    goldstein_feature_cols = [
        "gold_india_5d_mean",
        "gold_usa_5d_mean",
        "gold_diff_level",
        "gold_combined_weighted_5d",
        "gold_geo_5d",
        "gold_event_pressure",
    ]

    # Thematic features.
    out["theme_economy_5d_mean"] = out["Tone_Economy"].rolling(5).mean()
    out["theme_conflict_5d_mean"] = out["Tone_Conflict"].rolling(5).mean()
    out["theme_policy_5d_mean"] = out["Tone_Policy"].rolling(5).mean()
    out["theme_corporate_5d_mean"] = out["Tone_Corporate"].rolling(5).mean()
    out["theme_goldstein_weighted_log"] = np.sign(out["Goldstein_Weighted"]) * np.log1p(out["Goldstein_Weighted"].abs())
    out["theme_volume_spike_3d_mean"] = out["Volume_Spike"].rolling(3).mean()
    out["theme_conflict_spike"] = out["Volume_Spike_Conflict"].rolling(3).mean()
    out["theme_total_count_log"] = np.log1p(out["Count_Total"])
    thematic_feature_cols = [
        "theme_economy_5d_mean",
        "theme_conflict_5d_mean",
        "theme_policy_5d_mean",
        "theme_corporate_5d_mean",
        "theme_goldstein_weighted_log",
        "theme_volume_spike_3d_mean",
        "theme_conflict_spike",
        "theme_total_count_log",
    ]

    # Political aggregate features.
    out["pol_goldstein_5d_mean"] = out["GoldsteinScale_mean"].rolling(5).mean()
    out["pol_tone_5d_mean"] = out["AvgTone_mean"].rolling(5).mean()
    out["pol_mentions_log"] = np.log1p(out["Total_mentions"])
    out["pol_articles_log"] = np.log1p(out["Total_articles"])
    out["pol_event_log"] = np.log1p(out["Event_count"])
    out["pol_event_spike"] = out["Event_count"] / out["Event_count"].rolling(5).mean() - 1.0
    political_feature_cols = [
        "pol_goldstein_5d_mean",
        "pol_tone_5d_mean",
        "pol_mentions_log",
        "pol_articles_log",
        "pol_event_log",
        "pol_event_spike",
    ]

    # Market memory features.
    out["mem_macro_pressure"] = out[["macro_oil_dxy_pressure", "macro_rates_dollar_pressure"]].mean(axis=1)
    out["mem_sentiment_pressure"] = (
        -out["sent_in_tone_5d_mean"].fillna(0)
        + out["sent_in_panic_5d_mean"].fillna(0)
        - out["gold_india_5d_mean"].fillna(0) * 0.1
    )
    out["mem_event_heat"] = (
        out["theme_volume_spike_3d_mean"].fillna(0) * 0.01
        + out["pol_event_spike"].fillna(0)
        + out["theme_conflict_spike"].fillna(0) * 0.01
    )
    out["mem_vol_stress"] = out["tech_vol_20"] / out["tech_vol_20"].rolling(20).mean()
    out["mem_carry_pressure"] = out["sent_diff_stability_5d_mean"].fillna(0) - out["macro_rates_dollar_pressure"].fillna(0)
    out["mem_reversal_pressure"] = -out["tech_zscore_20"].fillna(0) - out["tech_drawdown_20"].fillna(0)
    out["mem_regime_score"] = (
        out["tech_trend_5_20"].fillna(0) * 2
        + out["mem_vol_stress"].fillna(0)
        + out["mem_sentiment_pressure"].fillna(0) * 0.5
    )
    memory_cols = [
        "mem_macro_pressure",
        "mem_sentiment_pressure",
        "mem_event_heat",
        "mem_vol_stress",
        "mem_carry_pressure",
        "mem_reversal_pressure",
        "mem_regime_score",
    ]

    # Forward targets.
    forward_steps = config.forecast_horizon_days + config.embargo_days
    out["target_date"] = out["Date"].shift(-forward_steps)
    out["target_price"] = out["INR"].shift(-forward_steps)
    out["target_return"] = out["target_price"] / out["INR"] - 1.0
    out["target_direction"] = np.select(
        [
            out["target_return"] > config.neutral_return_threshold,
            out["target_return"] < -config.neutral_return_threshold,
        ],
        [1, -1],
        default=0,
    )

    # Human-friendly helper for evaluation.
    out["expected_abs_move_proxy"] = out["tech_vol_20"].fillna(out["tech_vol_10"]).fillna(0.003)

    feature_groups = {
        "technical": technical_cols,
        "macro": macro_cols,
        "master_sentiment": master_sentiment_cols,
        "goldstein": goldstein_feature_cols,
        "thematic": thematic_feature_cols,
        "political": political_feature_cols,
        "memory": memory_cols,
    }

    return out, feature_groups

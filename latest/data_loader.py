"""
Data loading and feature engineering.
Loads Super_Master_Dataset.csv, computes technicals (MA, RSI, z-score, momentum),
assembles daily context packets for LLM consumption.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

import config


def load_dataset(path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load Super_Master_Dataset.csv and parse dates."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on INR price series."""
    df = df.copy()

    # Moving averages
    df["MA_5"] = df["INR"].rolling(config.MA_SHORT).mean()
    df["MA_20"] = df["INR"].rolling(config.MA_LONG).mean()
    df["MA_momentum"] = (df["MA_5"] - df["MA_20"]) / df["MA_20"]

    # Returns
    df["INR_return"] = df["INR"].pct_change()

    # RSI
    delta = df["INR"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(config.RSI_PERIOD).mean()
    avg_loss = loss.rolling(config.RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Z-score
    roll_mean = df["INR"].rolling(config.ZSCORE_WINDOW).mean()
    roll_std = df["INR"].rolling(config.ZSCORE_WINDOW).std()
    df["INR_zscore"] = (df["INR"] - roll_mean) / roll_std.replace(0, np.nan)

    # Realized volatility
    df["realized_vol"] = df["INR_return"].rolling(config.VOLATILITY_WINDOW).std()
    long_vol = df["INR_return"].rolling(config.VOLATILITY_LONG_WINDOW).std()
    df["vol_ratio"] = df["realized_vol"] / long_vol.replace(0, np.nan)

    # Macro changes
    for col in config.MACRO_FEATURES:
        df[f"{col}_change"] = df[col].pct_change()

    return df


def detect_regime(row: pd.Series, df: pd.DataFrame, idx: int) -> str:
    """Detect market regime from technical + GDELT features.
    Uses only data up to and including `idx` (no look-ahead).
    """
    # Crisis check first (highest priority)
    panic = row.get("IN_Panic_Index", 0)
    tone = row.get("IN_Avg_Tone", 0)
    vol_ratio = row.get("vol_ratio", 1.0)

    if (
        panic > config.REGIME_THRESHOLDS["panic_threshold"]
        or tone < config.REGIME_THRESHOLDS["tone_extreme"]
        or vol_ratio > config.REGIME_THRESHOLDS["vol_crisis"]
    ):
        return "CRISIS_STRESS"

    # High volatility
    if vol_ratio > config.REGIME_THRESHOLDS["vol_high"]:
        return "HIGH_VOLATILITY"

    # Trend detection
    ma_mom = row.get("MA_momentum", 0)
    if abs(ma_mom) > config.REGIME_THRESHOLDS["trend_threshold"]:
        if ma_mom > 0:
            return "TRENDING_DEPRECIATION"  # MA5 > MA20 → INR weakening
        else:
            return "TRENDING_APPRECIATION"  # MA5 < MA20 → INR strengthening

    return "CALM_CARRY"


def build_context_packet(
    df: pd.DataFrame, target_idx: int, headlines: Optional[List[dict]] = None
) -> dict:
    """Assemble a daily context packet for LLM consumption.
    Uses only data up to target_idx - 1 (predict for target_idx).
    """
    # Lookback data (up to day before target)
    lookback_end = target_idx  # exclusive — features from 0..target_idx-1
    lookback_start = max(0, lookback_end - config.CONTEXT_PRICE_HISTORY_DAYS)
    history = df.iloc[lookback_start:lookback_end]

    if len(history) < 2:
        return {}

    latest = history.iloc[-1]
    prev = history.iloc[-2] if len(history) >= 2 else latest

    # Price history summary
    price_history = {
        "current_inr": round(float(latest["INR"]), 4),
        "prev_inr": round(float(prev["INR"]), 4),
        "inr_20d_high": round(float(history["INR"].max()), 4),
        "inr_20d_low": round(float(history["INR"].min()), 4),
        "inr_20d_mean": round(float(history["INR"].mean()), 4),
        "inr_5d_trend": round(
            float(history["INR"].iloc[-min(5, len(history)):].mean()
            - history["INR"].mean()), 4
        ),
    }

    # Technicals
    technicals = {}
    for feat in config.TECHNICAL_FEATURES:
        val = latest.get(feat)
        if pd.notna(val):
            technicals[feat] = round(float(val), 6)

    # Macro snapshot
    macro = {}
    for col in config.MACRO_FEATURES:
        val = latest.get(col)
        if pd.notna(val):
            macro[col] = round(float(val), 4)
        chg = latest.get(f"{col}_change")
        if pd.notna(chg):
            macro[f"{col}_change"] = round(float(chg), 6)

    # GDELT sentiment
    sentiment = {}
    for col in config.GDELT_FEATURES:
        val = latest.get(col)
        if pd.notna(val):
            sentiment[col] = round(float(val), 4)

    # Detected regime (from features, not LLM)
    stat_regime = detect_regime(latest, df, lookback_end - 1)

    packet = {
        "date": str(latest["Date"].date()) if hasattr(latest["Date"], "date") else str(latest["Date"]),
        "price_history": price_history,
        "technicals": technicals,
        "macro": macro,
        "sentiment": sentiment,
        "stat_regime": stat_regime,
    }

    if headlines:
        packet["headlines"] = headlines[:config.MAX_HEADLINES_PER_DAY]

    return packet


# ─── GDELT Headline Extraction ───────────────────────────────────────────────
# Patterns adapted from gemini_preds/news_digest.py

def extract_headline_from_url(url: str) -> Optional[str]:
    """Extract a human-readable headline from a GDELT SOURCEURL."""
    try:
        parsed = urlparse(url)
        path = parsed.path

        # Remove extensions
        for ext in [".html", ".htm", ".php", ".aspx", ".jsp", ".shtml", ".cms"]:
            if path.endswith(ext):
                path = path[: -len(ext)]

        # Get last path segment
        segments = [s for s in path.split("/") if len(s) > 3]
        if not segments:
            return None
        slug = segments[-1]

        # Remove IDs and dates
        slug = re.sub(r"[-_]?\d{5,}$", "", slug)
        slug = re.sub(r"\d{4}[-/]?\d{2}[-/]?\d{2}", "", slug)

        # Decode and clean
        slug = unquote(slug)
        slug = slug.replace("-", " ").replace("_", " ")
        slug = re.sub(r"\s+", " ", slug).strip()

        if len(slug) < config.HEADLINE_MIN_LENGTH or len(slug) > config.HEADLINE_MAX_LENGTH:
            return None
        if sum(c.isalpha() for c in slug) / max(len(slug), 1) < 0.5:
            return None

        return slug.title()

    except Exception:
        return None


def load_gdelt_headlines(
    filepath: str, target_date, lookback_days: int = 1, max_headlines: int = 40
) -> List[dict]:
    """Load GDELT headlines for a given date range."""
    if not os.path.exists(filepath):
        return []

    try:
        cols = ["SQLDATE", "SOURCEURL", "AvgTone", "GoldsteinScale", "NumMentions"]
        df = pd.read_csv(filepath, usecols=cols, dtype={"SQLDATE": str})
        df["Date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
        df.dropna(subset=["Date"], inplace=True)

        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)

        start_date = target_date - timedelta(days=lookback_days)
        mask = (df["Date"] >= start_date) & (df["Date"] <= target_date)
        subset = df[mask].copy()

        if subset.empty:
            return []

        headlines = []
        seen = set()
        for _, row in subset.iterrows():
            url = row.get("SOURCEURL", "")
            if not isinstance(url, str):
                continue
            headline = extract_headline_from_url(url)
            if headline and headline not in seen:
                seen.add(headline)
                headlines.append({
                    "headline": headline,
                    "tone": float(row.get("AvgTone", 0)),
                    "goldstein": float(row.get("GoldsteinScale", 0)),
                    "mentions": int(row.get("NumMentions", 1)),
                })

        # Sort by importance: mentions * |tone|
        headlines.sort(key=lambda h: h["mentions"] * abs(h["tone"]), reverse=True)
        return headlines[:max_headlines]

    except Exception as e:
        print(f"[DataLoader] Warning: could not load headlines from {filepath}: {e}")
        return []


def prepare_dataset(df: pd.DataFrame = None) -> pd.DataFrame:
    """Full pipeline: load → add technicals → add target."""
    if df is None:
        df = load_dataset()
    df = add_technicals(df)
    # Target: next day's INR (shifted by -1)
    df["target"] = df["INR"].shift(-1)
    return df

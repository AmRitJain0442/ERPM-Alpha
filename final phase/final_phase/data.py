"""
Dataset integration for the final-phase system.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .config import RunConfig


MASTER_SENTIMENT_COLS = [
    "IN_Avg_Tone",
    "IN_Avg_Stability",
    "IN_Total_Mentions",
    "IN_Panic_Index",
    "US_Avg_Tone",
    "US_Avg_Stability",
    "US_Total_Mentions",
    "US_Panic_Index",
    "Diff_Stability",
    "Diff_Tone",
]

GOLDSTEIN_COLS = [
    "USA_Avg_Goldstein",
    "USA_Event_Count",
    "India_Avg_Goldstein",
    "India_Event_Count",
    "Combined_Simple_Avg",
    "Combined_Weighted_Avg",
    "Combined_Product",
    "Combined_Geometric_Mean",
    "USA_India_Sentiment_Diff",
]

THEMATIC_COLS = [
    "Tone_Economy",
    "Tone_Conflict",
    "Tone_Policy",
    "Tone_Corporate",
    "Tone_Overall",
    "Goldstein_Weighted",
    "Goldstein_Avg",
    "Count_Economy",
    "Count_Conflict",
    "Count_Policy",
    "Count_Corporate",
    "Count_Total",
    "Volume_Spike",
    "Volume_Spike_Economy",
    "Volume_Spike_Conflict",
    "IMF_3",
]

POLITICAL_COLS = [
    "GoldsteinScale_mean",
    "GoldsteinScale_std",
    "Event_count",
    "AvgTone_mean",
    "AvgTone_std",
    "Total_mentions",
    "Total_articles",
]


def _read_csv(path, parse_dates=("Date",)) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=list(parse_dates))
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df


def load_integrated_dataset(config: RunConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge the repo's core modeling datasets into one date-aligned frame.
    """
    paths = config.paths

    master = _read_csv(paths.master_dataset)
    goldstein = _read_csv(paths.bilateral_goldstein)
    thematic = _read_csv(paths.thematic_dataset)
    political = _read_csv(paths.political_dataset)

    goldstein = goldstein[["Date"] + GOLDSTEIN_COLS].copy()
    thematic = thematic[["Date"] + THEMATIC_COLS].copy()
    political = political[["Date"] + POLITICAL_COLS].copy()

    df = master.copy()
    df = df.merge(goldstein, on="Date", how="left")
    df = df.merge(thematic, on="Date", how="left")
    df = df.merge(political, on="Date", how="left")
    df = df.sort_values("Date").reset_index(drop=True)

    summary = {
        "rows": int(len(df)),
        "date_min": str(df["Date"].min().date()) if not df.empty else None,
        "date_max": str(df["Date"].max().date()) if not df.empty else None,
        "rich_rows": int(
            df[GOLDSTEIN_COLS + THEMATIC_COLS + POLITICAL_COLS].notna().all(axis=1).sum()
        ),
        "columns": list(df.columns),
        "feature_coverage": {
            "master_sentiment_rows": int(df[MASTER_SENTIMENT_COLS].notna().all(axis=1).sum()),
            "goldstein_rows": int(df[GOLDSTEIN_COLS].notna().all(axis=1).sum()),
            "thematic_rows": int(df[THEMATIC_COLS].notna().all(axis=1).sum()),
            "political_rows": int(df[POLITICAL_COLS].notna().all(axis=1).sum()),
        },
    }

    if config.rich_only:
        required = GOLDSTEIN_COLS + THEMATIC_COLS + POLITICAL_COLS
        df = df[df[required].notna().all(axis=1)].reset_index(drop=True)
        summary["post_filter_rows"] = int(len(df))
        summary["post_filter_date_min"] = str(df["Date"].min().date()) if not df.empty else None
        summary["post_filter_date_max"] = str(df["Date"].max().date()) if not df.empty else None

    return df, summary

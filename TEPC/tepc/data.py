"""
Local-data ingestion for the INR/USD TEPC pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

from .config import RunConfig


@dataclass
class MarketDataset:
    merged: pd.DataFrame
    node_frame: pd.DataFrame
    node_transforms: pd.DataFrame


def _load_indexed_csv(path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "Date"}).set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    return df


def _getcol(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _merge_prefer_existing(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.reindex(left.index.union(right.index)).sort_index().copy()
    for column in right.columns:
        candidate = pd.to_numeric(right[column], errors="coerce").reindex(merged.index)
        if column in merged.columns:
            base = pd.to_numeric(merged[column], errors="coerce")
            merged[column] = base.combine_first(candidate)
        else:
            merged[column] = candidate
    return merged


def _signed_shock_transform(series: pd.Series) -> pd.Series:
    s = series.astype(float).ffill()
    if (s.dropna() > 0).all():
        return np.log(s).diff()
    delta = s.diff()
    return np.sign(delta) * np.log1p(delta.abs())


def load_market_dataset(config: RunConfig) -> MarketDataset:
    paths = config.paths

    master = _load_indexed_csv(paths.master_dataset, "Date").rename(
        columns={
            "INR": "INRUSD",
            "OIL": "BRENT",
            "GOLD": "GOLD",
            "US10Y": "US10Y",
            "DXY": "DXY",
        }
    )

    thematic = _load_indexed_csv(paths.thematic_dataset, "Date").rename(
        columns={
            "Tone_Economy": "theme_tone_economy",
            "Tone_Conflict": "theme_tone_conflict",
            "Tone_Policy": "theme_tone_policy",
            "Tone_Corporate": "theme_tone_corporate",
            "Tone_Overall": "theme_tone_overall",
            "Goldstein_Weighted": "theme_goldstein_weighted",
            "Goldstein_Avg": "theme_goldstein_avg",
            "Count_Total": "theme_count_total",
            "Volume_Spike": "theme_volume_spike",
            "Volume_Spike_Economy": "theme_volume_spike_economy",
            "Volume_Spike_Conflict": "theme_volume_spike_conflict",
            "IMF_3": "theme_imf_3",
        }
    )

    goldstein = _load_indexed_csv(paths.goldstein_dataset, "Date").rename(
        columns={
            "USA_Avg_Goldstein": "gs_usa_avg",
            "USA_Event_Count": "gs_usa_event_count",
            "India_Avg_Goldstein": "gs_india_avg",
            "India_Event_Count": "gs_india_event_count",
            "Combined_Simple_Avg": "gs_combined_simple_avg",
            "Combined_Weighted_Avg": "gs_weighted_avg",
            "Combined_Product": "gs_combined_product",
            "Combined_Geometric_Mean": "gs_geometric_mean",
            "USA_India_Sentiment_Diff": "gs_us_india_diff",
            "USD_to_INR": "gs_usd_to_inr",
            "Exchange_Rate_Change": "gs_exchange_rate_change",
            "Exchange_Rate_Change_Abs": "gs_exchange_rate_change_abs",
        }
    )

    political = _load_indexed_csv(paths.political_dataset, "Date").rename(
        columns={
            "GoldsteinScale_mean": "pol_goldstein_mean",
            "GoldsteinScale_std": "pol_goldstein_std",
            "Event_count": "pol_event_count",
            "AvgTone_mean": "pol_tone_mean",
            "AvgTone_std": "pol_tone_std",
            "Total_mentions": "pol_total_mentions",
            "Total_articles": "pol_total_articles",
            "USD_to_INR": "pol_usd_to_inr",
        }
    )

    fred = _load_indexed_csv(paths.fred_dataset, "date").rename(
        columns={
            "DEXINUS": "fred_inr_proxy",
            "DFF": "fred_ffr",
            "DGS10": "fred_us10y",
            "DTWEXBGS": "fred_dxy",
            "DCOILWTICO": "fred_oil",
        }
    )

    tepc_market = (
        _load_indexed_csv(paths.tepc_market_dataset, "Date")
        if paths.tepc_market_dataset.exists()
        else pd.DataFrame()
    )
    tepc_gdelt = (
        _load_indexed_csv(paths.tepc_gdelt_dataset, "Date")
        if paths.tepc_gdelt_dataset.exists()
        else pd.DataFrame()
    )

    merged = master.join(thematic, how="left")
    merged = merged.join(goldstein, how="left")
    merged = merged.join(political, how="left")
    merged = merged.join(fred, how="left")
    if not tepc_market.empty:
        merged = _merge_prefer_existing(merged, tepc_market)
    if not tepc_gdelt.empty:
        merged = _merge_prefer_existing(merged, tepc_gdelt)
    merged = merged.sort_index()

    if "INRUSD" in merged.columns:
        merged = merged.loc[merged["INRUSD"].notna()].copy()

    short_fill_cols = [
        "INRUSD",
        "DXY",
        "BRENT",
        "GOLD",
        "US10Y",
        "EURUSD",
        "GBPINR",
        "USDCNH",
        "CNHUSD",
        "INRUSD_FRANKFURTER",
        "FRED_DGS10",
        "FRED_DFF",
        "FRED_DTWEXBGS",
        "FRED_DCOILWTICO",
        "FRED_DEXINUS",
        "india_fx_volume",
        "india_fx_tone",
        "usd_macro_volume",
        "usd_macro_tone",
        "geo_risk_volume",
        "geo_risk_tone",
    ]
    for col in short_fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill(limit=3)

    node_frame = pd.DataFrame(index=merged.index)
    node_frame["INRUSD"] = _getcol(merged, "INRUSD").combine_first(_getcol(merged, "gs_usd_to_inr", np.nan))
    node_frame["DXY"] = _getcol(merged, "DXY").combine_first(_getcol(merged, "fred_dxy", np.nan))
    node_frame["BRENT"] = _getcol(merged, "BRENT").combine_first(_getcol(merged, "fred_oil", np.nan))
    node_frame["GOLD"] = _getcol(merged, "GOLD")
    node_frame["US10Y"] = _getcol(merged, "US10Y").combine_first(_getcol(merged, "fred_us10y", np.nan))

    for live_col in ["EURUSD", "GBPINR", "USDCNH", "CNHUSD"]:
        if live_col in merged.columns:
            node_frame[live_col] = _getcol(merged, live_col)

    in_sent = (
        -0.35 * _getcol(merged, "IN_Avg_Tone")
        + 0.90 * _getcol(merged, "IN_Panic_Index")
        + 0.08 * np.log1p(_getcol(merged, "IN_Total_Mentions").clip(lower=0))
    ).fillna(0.0)
    us_sent = (
        -0.35 * _getcol(merged, "US_Avg_Tone")
        + 0.90 * _getcol(merged, "US_Panic_Index")
        + 0.08 * np.log1p(_getcol(merged, "US_Total_Mentions").clip(lower=0))
    ).fillna(0.0)
    goldstein_spread = (
        _getcol(merged, "gs_us_india_diff").combine_first(_getcol(merged, "Diff_Tone", 0.0))
    ).fillna(0.0)
    geo_risk = (
        0.45 * _getcol(merged, "pol_event_count").clip(lower=0)
        + 0.25 * _getcol(merged, "pol_goldstein_std").abs()
        + 0.20 * np.log1p(_getcol(merged, "pol_total_mentions").clip(lower=0))
        + 0.10 * _getcol(merged, "pol_tone_std").abs()
    ).fillna(0.0)
    theme_pressure = (
        -0.40 * _getcol(merged, "theme_tone_overall")
        + 0.25 * _getcol(merged, "theme_volume_spike")
        + 0.20 * _getcol(merged, "theme_imf_3").abs()
        + 0.15 * _getcol(merged, "theme_tone_conflict").abs()
    ).fillna(0.0)

    node_frame["INDIA_SENTIMENT"] = in_sent
    node_frame["US_SENTIMENT"] = us_sent
    node_frame["GOLDSTEIN_SPREAD"] = goldstein_spread
    node_frame["GEO_RISK"] = geo_risk
    node_frame["THEME_PRESSURE"] = theme_pressure

    if "india_fx_volume" in merged.columns:
        node_frame["LIVE_INDIA_FX_NEWS"] = (
            np.log1p(_getcol(merged, "india_fx_volume").clip(lower=0))
            - 0.12 * _getcol(merged, "india_fx_tone")
        )
    if "usd_macro_volume" in merged.columns:
        node_frame["LIVE_USD_MACRO_NEWS"] = (
            np.log1p(_getcol(merged, "usd_macro_volume").clip(lower=0))
            - 0.12 * _getcol(merged, "usd_macro_tone")
        )
    if "geo_risk_volume" in merged.columns:
        node_frame["LIVE_GEO_RISK"] = (
            np.log1p(_getcol(merged, "geo_risk_volume").clip(lower=0))
            - 0.15 * _getcol(merged, "geo_risk_tone")
        )

    core_nodes = [
        column
        for column in [
            "INRUSD",
            "DXY",
            "BRENT",
            "GOLD",
            "US10Y",
            "INDIA_SENTIMENT",
            "US_SENTIMENT",
            "GOLDSTEIN_SPREAD",
            "GEO_RISK",
            "THEME_PRESSURE",
        ]
        if column in node_frame.columns
    ]
    optional_nodes = [column for column in node_frame.columns if column not in core_nodes]
    optional_meta = {}
    for column in optional_nodes:
        first_valid = node_frame[column].first_valid_index()
        if first_valid is None:
            continue
        tail = node_frame.loc[first_valid:, column]
        coverage = float(tail.notna().mean())
        if coverage >= config.min_node_coverage:
            optional_meta[column] = {"first_valid": first_valid, "coverage": coverage}

    required_history = (
        config.train_min_days
        + config.test_days
        + max(config.corr_window, config.chaos_lookback_days, config.volatility_window)
        + config.forecast_horizon_days
    )
    selected_optional = list(optional_meta)
    chosen = pd.DataFrame()
    while selected_optional:
        start_date = max(optional_meta[column]["first_valid"] for column in selected_optional)
        candidate = (
            node_frame[core_nodes + selected_optional]
            .loc[start_date:]
            .sort_index()
            .ffill(limit=3)
            .dropna()
        )
        if len(candidate) >= required_history:
            chosen = candidate
            break
        drop_column = max(
            selected_optional,
            key=lambda column: (
                optional_meta[column]["first_valid"],
                -optional_meta[column]["coverage"],
            ),
        )
        selected_optional.remove(drop_column)

    if chosen.empty:
        chosen = node_frame[core_nodes].sort_index().ffill(limit=3).dropna()

    node_frame = chosen
    merged = merged.loc[node_frame.index].copy()

    node_transforms = pd.DataFrame(
        {column: _signed_shock_transform(node_frame[column]) for column in node_frame.columns},
        index=node_frame.index,
    )

    return MarketDataset(
        merged=merged,
        node_frame=node_frame,
        node_transforms=node_transforms,
    )

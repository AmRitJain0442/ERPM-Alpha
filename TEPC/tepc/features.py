"""
Feature construction for the TEPC pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

from .chaos import compute_chaos_features
from .config import RunConfig
from .data import MarketDataset
from .topology import compute_dynamic_topology


@dataclass
class FeatureBundle:
    frame: pd.DataFrame
    groups: Dict[str, List[str]]
    dataset_summary: Dict
    node_frame: pd.DataFrame
    node_transforms: pd.DataFrame


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0.0, np.nan)
    return (series - mean) / std


def _range_position(series: pd.Series, window: int) -> pd.Series:
    rolling_min = series.rolling(window).min()
    rolling_max = series.rolling(window).max()
    return (series - rolling_min) / (rolling_max - rolling_min + 1e-9)


def _future_realized_volatility(shocks: pd.Series, window: int, offset: int) -> pd.Series:
    values = []
    idx = shocks.index
    for pos in range(len(shocks)):
        future_window = shocks.iloc[pos + offset : pos + offset + window].dropna()
        values.append(float(future_window.abs().mean()) if len(future_window) == window else np.nan)
    return pd.Series(values, index=idx, dtype=float)


def build_feature_bundle(dataset: MarketDataset, config: RunConfig) -> FeatureBundle:
    merged = dataset.merged
    node_frame = dataset.node_frame
    node_transforms = dataset.node_transforms

    market_nodes = [col for col in ["INRUSD", "DXY", "BRENT", "GOLD", "US10Y"] if col in node_frame.columns]
    macro_feature_nodes = [col for col in market_nodes if col != "INRUSD"]
    alt_nodes = [col for col in node_frame.columns if col not in market_nodes and col != "INRUSD"]

    macro = pd.DataFrame(index=node_frame.index)
    for node in macro_feature_nodes:
        macro[f"{node.lower()}_shock_1"] = node_transforms[node]
        macro[f"{node.lower()}_shock_mean_5"] = node_transforms[node].rolling(5).mean()
        macro[f"{node.lower()}_shock_vol_10"] = node_transforms[node].rolling(10).std()

    if {"BRENT", "DXY"}.issubset(macro_feature_nodes):
        macro["brent_dxy_interaction"] = node_transforms["BRENT"] * node_transforms["DXY"]
    else:
        macro["brent_dxy_interaction"] = 0.0

    alt = pd.DataFrame(index=node_frame.index)
    for node in alt_nodes:
        alt[f"{node.lower()}_shock_1"] = node_transforms[node]
        alt[f"{node.lower()}_mean_5"] = node_transforms[node].rolling(5).mean()

    for source_col in [
        "theme_tone_conflict",
        "theme_tone_policy",
        "theme_volume_spike",
        "theme_imf_3",
        "gs_weighted_avg",
        "gs_us_india_diff",
        "pol_event_count",
        "pol_total_mentions",
        "pol_goldstein_std",
        "india_fx_goldstein",
        "india_fx_articles",
        "usd_macro_goldstein",
        "usd_macro_articles",
        "geo_risk_goldstein",
        "geo_risk_articles",
    ]:
        if source_col in merged.columns:
            alt[source_col] = merged[source_col].astype(float)

    alt_coverage = alt.notna().mean()
    keep_alt = [col for col in alt.columns if alt_coverage[col] >= 0.8]
    alt = alt[keep_alt]

    topology_result = compute_dynamic_topology(node_transforms.dropna(), config)
    topology = topology_result.features
    chaos = compute_chaos_features(
        node_frame,
        node_transforms,
        topology_result.adjacency_by_date,
        config,
    )

    target = pd.DataFrame(index=node_frame.index)
    target_offset = config.response_lag_days + config.forecast_horizon_days - 1
    target["decision_date"] = node_frame.index
    target["target_date"] = node_frame.index.to_series().shift(-target_offset)
    target["current_rate"] = node_frame["INRUSD"]
    target["future_return"] = node_frame["INRUSD"].shift(-target_offset) / node_frame["INRUSD"] - 1.0
    target["future_volatility"] = _future_realized_volatility(
        node_transforms["INRUSD"].fillna(0.0),
        config.volatility_window,
        offset=config.response_lag_days,
    )
    target["future_label_int"] = target["future_return"].map(
        lambda value: 1 if value > config.breakout_threshold else (-1 if value < -config.breakout_threshold else 0)
    )
    target["future_label"] = target["future_label_int"].map({-1: "down", 0: "range", 1: "up"})
    target["actual_rate"] = node_frame["INRUSD"].shift(-target_offset)

    groups = {
        "macro": list(macro.columns),
        "alt": list(alt.columns),
        "topology": list(topology.columns),
        "chaos": list(chaos.columns),
    }

    frame = pd.concat([target, macro, alt, topology, chaos], axis=1)
    feature_cols = groups["macro"] + groups["alt"] + groups["topology"] + groups["chaos"]
    frame = frame.dropna(subset=feature_cols + ["future_return", "future_volatility", "target_date", "actual_rate"])

    dataset_summary = {
        "n_rows": int(len(frame)),
        "date_start": str(frame.index.min().date()) if not frame.empty else None,
        "date_end": str(frame.index.max().date()) if not frame.empty else None,
        "nodes": list(node_frame.columns),
        "node_count": int(len(node_frame.columns)),
        "feature_group_counts": {group: len(cols) for group, cols in groups.items()},
        "market_nodes": market_nodes,
        "macro_feature_nodes": macro_feature_nodes,
        "alternative_nodes": alt_nodes,
        "response_lag_days": int(config.response_lag_days),
        "forecast_horizon_days": int(config.forecast_horizon_days),
    }

    return FeatureBundle(
        frame=frame,
        groups=groups,
        dataset_summary=dataset_summary,
        node_frame=node_frame.loc[frame.index],
        node_transforms=node_transforms.loc[frame.index],
    )

"""
Live data pullers for TEPC market and GDELT feeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional
import json
import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import DataPaths, MODULE_ROOT


YF_SYMBOLS = {
    "INRUSD": "INR=X",
    "EURUSD": "EURUSD=X",
    "GBPINR": "GBPINR=X",
    "USDCNH": "USDCNH=X",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "US10Y": "^TNX",
    "DXY": "DX-Y.NYB",
}


FRED_SERIES = {
    "FRED_DGS10": "DGS10",
    "FRED_DFF": "DFF",
    "FRED_DTWEXBGS": "DTWEXBGS",
    "FRED_DCOILWTICO": "DCOILWTICO",
    "FRED_DEXINUS": "DEXINUS",
}


GDELT_QUERIES = {
    "india_fx": '("india" OR "rupee" OR "inr" OR "rbi")',
    "usd_macro": '("federal reserve" OR "us treasury" OR "dollar" OR "10-year yield")',
    "geo_risk": '("war" OR "conflict" OR "sanctions" OR "tariff" OR "missile" OR "attack" OR "oil shock")',
}


RAW_GDELT_USECOLS = [
    "SQLDATE",
    "Actor1Name",
    "Actor2Name",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumArticles",
    "AvgTone",
    "SOURCEURL",
]

INDIA_FX_KEYWORDS = [
    "economy",
    "inflation",
    "rbi",
    "reserve bank",
    "tax",
    "gdp",
    "fiscal",
    "monetary",
    "interest rate",
    "repo rate",
    "trade",
    "rupee",
    "currency",
    "exchange rate",
    "foreign exchange",
    "forex",
    "fed",
    "federal reserve",
    "stock market",
    "bond",
    "yield",
    "budget",
    "oil",
    "gold",
]

USD_MACRO_KEYWORDS = [
    "federal reserve",
    "fed",
    "treasury",
    "dollar",
    "bond",
    "yield",
    "interest rate",
    "inflation",
    "tariff",
    "trade",
    "oil",
    "gold",
    "economic",
    "economy",
    "recession",
    "payroll",
    "cpi",
    "pce",
]

GEO_RISK_KEYWORDS = [
    "war",
    "military",
    "missile",
    "attack",
    "conflict",
    "sanction",
    "embargo",
    "tariff",
    "terror",
    "threat",
    "border",
    "standoff",
    "riot",
    "unrest",
    "oil shock",
]


@dataclass
class PullConfig:
    start_date: str = "2024-12-01"
    end_date: Optional[str] = None
    output_dir: Optional[Path] = None
    gdelt_pause_seconds: float = 7.0
    gdelt_chunk_size: int = 100_000

    def resolve_output_dir(self) -> Path:
        return self.output_dir or (MODULE_ROOT / "data")

    def resolve_end_date(self) -> str:
        return self.end_date or date.today().isoformat()


def _extract_close(data: pd.DataFrame, symbol: str) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        close_df = data["Close"]
        if isinstance(close_df, pd.DataFrame):
            series = close_df.iloc[:, 0]
        else:
            series = close_df
    else:
        series = data["Close"]
    return pd.Series(series.astype(float).values, index=pd.to_datetime(data.index).normalize(), name=symbol)


def _preferred_series(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    primary_series = frame[primary] if primary in frame.columns else pd.Series(index=frame.index, dtype=float)
    fallback_series = frame[fallback] if fallback in frame.columns else pd.Series(index=frame.index, dtype=float)
    return primary_series.combine_first(fallback_series).astype(float)


def _read_fred_series(series_id: str, output_name: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    fred_df = pd.read_csv(url)
    date_col = next(
        (column for column in ["DATE", "date", "observation_date", "Date"] if column in fred_df.columns),
        None,
    )
    if date_col is None:
        raise ValueError(f"Could not locate FRED date column for series {series_id}. Columns: {list(fred_df.columns)}")

    value_candidates = [column for column in fred_df.columns if column != date_col]
    if not value_candidates:
        raise ValueError(f"Could not locate FRED value column for series {series_id}.")

    value_col = series_id if series_id in value_candidates else value_candidates[0]
    fred_df = fred_df.rename(columns={date_col: "Date", value_col: output_name}).copy()
    fred_df["Date"] = pd.to_datetime(fred_df["Date"], errors="coerce")
    fred_df[output_name] = pd.to_numeric(fred_df[output_name], errors="coerce")
    fred_df = fred_df.dropna(subset=["Date"]).set_index("Date")
    fred_df.index = pd.to_datetime(fred_df.index).normalize()
    return fred_df.loc[(fred_df.index >= pd.Timestamp(start)) & (fred_df.index <= pd.Timestamp(end)), [output_name]]


def _read_indexed_csv(path: Path, date_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[date_col])
    frame = frame.rename(columns={date_col: "Date"}).set_index("Date").sort_index()
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame


def _compile_keyword_regex(keywords) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(keyword) for keyword in keywords), re.IGNORECASE)


INDIA_FX_REGEX = _compile_keyword_regex(INDIA_FX_KEYWORDS)
USD_MACRO_REGEX = _compile_keyword_regex(USD_MACRO_KEYWORDS)
GEO_RISK_REGEX = _compile_keyword_regex(GEO_RISK_KEYWORDS)


def _daily_signal_block(
    dates: pd.Series,
    mentions: pd.Series,
    articles: pd.Series,
    tone: pd.Series,
    goldstein: pd.Series,
    mask: pd.Series,
) -> pd.DataFrame:
    if not bool(mask.any()):
        return pd.DataFrame(columns=["weight", "volume", "articles", "tone_num", "gold_num"])

    clipped_mentions = mentions[mask].clip(lower=0.0)
    weights = clipped_mentions.where(clipped_mentions > 0.0, 1.0)
    frame = pd.DataFrame(
        {
            "Date": dates[mask].to_numpy(),
            "weight": weights.to_numpy(dtype=float),
            "volume": clipped_mentions.to_numpy(dtype=float),
            "articles": articles[mask].clip(lower=0.0).to_numpy(dtype=float),
            "tone_num": (tone[mask] * weights).to_numpy(dtype=float),
            "gold_num": (goldstein[mask] * weights).to_numpy(dtype=float),
        }
    )
    return frame.groupby("Date", as_index=True)[["weight", "volume", "articles", "tone_num", "gold_num"]].sum()


def _finalize_signal(blocks: list[pd.DataFrame], prefix: str) -> pd.DataFrame:
    columns = [f"{prefix}_volume", f"{prefix}_tone", f"{prefix}_goldstein", f"{prefix}_articles"]
    if not blocks:
        return pd.DataFrame(columns=columns)

    grouped = pd.concat(blocks).groupby(level=0).sum().sort_index()
    denom = grouped["weight"].replace(0.0, np.nan)
    out = pd.DataFrame(index=grouped.index)
    out[f"{prefix}_volume"] = grouped["volume"]
    out[f"{prefix}_tone"] = grouped["tone_num"] / denom
    out[f"{prefix}_goldstein"] = grouped["gold_num"] / denom
    out[f"{prefix}_articles"] = grouped["articles"]
    return out


def _build_local_gdelt_fallback(start: str, end: str) -> pd.DataFrame:
    paths = DataPaths()
    frames = []

    if paths.thematic_dataset.exists():
        thematic = _read_indexed_csv(paths.thematic_dataset, "Date")
        fallback = pd.DataFrame(index=thematic.index)
        if "Count_Total" in thematic.columns:
            fallback["india_fx_volume"] = pd.to_numeric(thematic["Count_Total"], errors="coerce")
        if "Tone_Overall" in thematic.columns:
            fallback["india_fx_tone"] = pd.to_numeric(thematic["Tone_Overall"], errors="coerce")
        frames.append(fallback)

    if paths.goldstein_dataset.exists():
        goldstein = _read_indexed_csv(paths.goldstein_dataset, "Date")
        fallback = pd.DataFrame(index=goldstein.index)
        if "USA_Event_Count" in goldstein.columns:
            fallback["usd_macro_volume"] = pd.to_numeric(goldstein["USA_Event_Count"], errors="coerce")
        if "USA_Avg_Goldstein" in goldstein.columns:
            fallback["usd_macro_tone"] = pd.to_numeric(goldstein["USA_Avg_Goldstein"], errors="coerce")
        if "India_Event_Count" in goldstein.columns:
            fallback["india_fx_volume"] = fallback.get("india_fx_volume", 0.0) + pd.to_numeric(
                goldstein["India_Event_Count"], errors="coerce"
            ).fillna(0.0)
        if "India_Avg_Goldstein" in goldstein.columns:
            india_goldstein = pd.to_numeric(goldstein["India_Avg_Goldstein"], errors="coerce")
            fallback["india_fx_tone"] = fallback.get("india_fx_tone", india_goldstein).combine_first(india_goldstein)
        frames.append(fallback)

    if paths.political_dataset.exists():
        political = _read_indexed_csv(paths.political_dataset, "Date")
        fallback = pd.DataFrame(index=political.index)
        if "Event_count" in political.columns:
            fallback["geo_risk_volume"] = pd.to_numeric(political["Event_count"], errors="coerce")
        if "AvgTone_mean" in political.columns:
            fallback["geo_risk_tone"] = pd.to_numeric(political["AvgTone_mean"], errors="coerce")
        frames.append(fallback)

    merged = pd.DataFrame()
    for frame in frames:
        merged = frame if merged.empty else merged.join(frame, how="outer", rsuffix="_dup")
        dup_cols = [col for col in merged.columns if col.endswith("_dup")]
        for dup_col in dup_cols:
            base_col = dup_col.removesuffix("_dup")
            merged[base_col] = merged[base_col].combine_first(merged[dup_col])
            merged = merged.drop(columns=[dup_col])

    if merged.empty:
        return merged

    merged = merged.sort_index()
    merged = merged.loc[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]
    merged.index.name = "Date"
    return merged


def _aggregate_raw_gdelt_file(
    path: Path,
    signal_key: str,
    regex: re.Pattern[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    signal_blocks: list[pd.DataFrame] = []
    risk_blocks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        path,
        usecols=RAW_GDELT_USECOLS,
        chunksize=chunk_size,
        low_memory=False,
    ):
        dates = pd.to_datetime(chunk["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce").dt.normalize()
        date_mask = dates.between(start, end)
        if not bool(date_mask.any()):
            continue

        chunk = chunk.loc[date_mask].copy()
        dates = dates.loc[date_mask]
        mentions = pd.to_numeric(chunk["NumMentions"], errors="coerce").fillna(0.0)
        articles = pd.to_numeric(chunk["NumArticles"], errors="coerce").fillna(0.0)
        tone = pd.to_numeric(chunk["AvgTone"], errors="coerce").fillna(0.0)
        goldstein = pd.to_numeric(chunk["GoldsteinScale"], errors="coerce").fillna(0.0)
        quad_class = pd.to_numeric(chunk["QuadClass"], errors="coerce").fillna(0.0)

        text = (
            chunk["SOURCEURL"].fillna("").astype(str)
            + " "
            + chunk["Actor1Name"].fillna("").astype(str)
            + " "
            + chunk["Actor2Name"].fillna("").astype(str)
        ).str.lower()

        signal_mask = text.str.contains(regex, na=False)
        risk_mask = text.str.contains(GEO_RISK_REGEX, na=False) | ((goldstein < -2.0) & (quad_class >= 3.0))

        signal_block = _daily_signal_block(dates, mentions, articles, tone, goldstein, signal_mask)
        risk_block = _daily_signal_block(dates, mentions, articles, tone, goldstein, risk_mask)
        if not signal_block.empty:
            signal_blocks.append(signal_block)
        if not risk_block.empty:
            risk_blocks.append(risk_block)

    return signal_blocks, risk_blocks


def fetch_local_raw_gdelt_daily(config: PullConfig) -> pd.DataFrame:
    paths = DataPaths()
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.resolve_end_date())

    if not (paths.india_raw_gdelt_dataset.exists() and paths.usa_raw_gdelt_dataset.exists()):
        return pd.DataFrame()

    india_blocks, india_risk_blocks = _aggregate_raw_gdelt_file(
        paths.india_raw_gdelt_dataset,
        "india_fx",
        INDIA_FX_REGEX,
        start,
        end,
        config.gdelt_chunk_size,
    )
    usa_blocks, usa_risk_blocks = _aggregate_raw_gdelt_file(
        paths.usa_raw_gdelt_dataset,
        "usd_macro",
        USD_MACRO_REGEX,
        start,
        end,
        config.gdelt_chunk_size,
    )

    india_daily = _finalize_signal(india_blocks, "india_fx")
    usa_daily = _finalize_signal(usa_blocks, "usd_macro")
    risk_daily = _finalize_signal(india_risk_blocks + usa_risk_blocks, "geo_risk")

    merged = india_daily.join(usa_daily, how="outer")
    merged = merged.join(risk_daily, how="outer")
    merged = merged.sort_index()
    if not merged.empty:
        merged = merged.loc[(merged.index >= start.normalize()) & (merged.index <= end.normalize())]
    merged.index.name = "Date"
    return merged


def fetch_market_data(config: PullConfig) -> pd.DataFrame:
    start = config.start_date
    end = config.resolve_end_date()
    rows = {}

    for column, ticker in YF_SYMBOLS.items():
        history = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            timeout=30,
        )
        series = _extract_close(history, ticker)
        if not series.empty:
            rows[column] = series

    market = pd.DataFrame(rows).sort_index()

    if "USDCNH" in market.columns:
        market["CNHUSD"] = 1.0 / market["USDCNH"].replace(0.0, np.nan)

    frankfurter = requests.get(
        f"https://api.frankfurter.app/{start}..{end}",
        params={"from": "USD", "to": "INR"},
        timeout=60,
    )
    frankfurter.raise_for_status()
    ff_payload = frankfurter.json()
    if ff_payload.get("rates"):
        ff_df = pd.DataFrame.from_dict(ff_payload["rates"], orient="index")
        ff_df.index = pd.to_datetime(ff_df.index).normalize()
        ff_df = ff_df.rename(columns={"INR": "INRUSD_FRANKFURTER"})
        market = market.join(ff_df, how="outer")

    for out_col, series_id in FRED_SERIES.items():
        fred_df = _read_fred_series(series_id, out_col, start, end)
        market = market.join(fred_df, how="outer")

    market = market.sort_index()
    market["INRUSD"] = _preferred_series(market, "INRUSD", "INRUSD_FRANKFURTER")
    market["US10Y"] = _preferred_series(market, "US10Y", "FRED_DGS10")
    market["DXY"] = _preferred_series(market, "DXY", "FRED_DTWEXBGS")
    if "FRED_DCOILWTICO" in market.columns:
        market["WTI"] = market["FRED_DCOILWTICO"].astype(float)
    market.index.name = "Date"
    return market


def _gdelt_request(query: str, mode: str, timespan_days: int, pause_seconds: float) -> Dict:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "timespan": f"{timespan_days}days",
    }
    for attempt in range(4):
        resp = requests.get(url, params=params, timeout=120)
        text = resp.text.strip()
        if resp.status_code == 429 or "Please limit requests" in text:
            time.sleep(pause_seconds * (attempt + 1))
            continue
        if resp.status_code != 200:
            resp.raise_for_status()
        try:
            payload = resp.json()
        except requests.exceptions.JSONDecodeError:
            time.sleep(pause_seconds * (attempt + 1))
            continue
        time.sleep(pause_seconds)
        return payload
    raise RuntimeError(f"GDELT request failed after retries for query={query!r}, mode={mode!r}")


def _timeline_to_daily(payload: Dict, value_name: str) -> pd.DataFrame:
    data = payload.get("timeline", [])
    if not data:
        return pd.DataFrame(columns=[value_name])
    points = data[0].get("data", [])
    if not points:
        return pd.DataFrame(columns=[value_name])
    frame = pd.DataFrame(points)
    frame["Date"] = pd.to_datetime(frame["date"], format="%Y%m%dT%H%M%SZ", errors="coerce").dt.normalize()
    frame[value_name] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["Date", value_name]).groupby("Date", as_index=True)[value_name].mean().to_frame()
    return frame


def fetch_gdelt_daily(config: PullConfig) -> pd.DataFrame:
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.resolve_end_date())
    timespan_days = max(int((end - start).days) + 5, 30)

    merged = pd.DataFrame()
    for key, query in GDELT_QUERIES.items():
        vol_payload = _gdelt_request(query, "TimelineVolRaw", timespan_days, config.gdelt_pause_seconds)
        tone_payload = _gdelt_request(query, "TimelineTone", timespan_days, config.gdelt_pause_seconds)
        vol_df = _timeline_to_daily(vol_payload, f"{key}_volume")
        tone_df = _timeline_to_daily(tone_payload, f"{key}_tone")
        block = vol_df.join(tone_df, how="outer")
        merged = block if merged.empty else merged.join(block, how="outer")

    merged = merged.sort_index()
    if not merged.empty:
        merged = merged.loc[(merged.index >= start.normalize()) & (merged.index <= end.normalize())]
    merged.index.name = "Date"
    return merged


def pull_and_store_data(config: PullConfig) -> Dict:
    output_dir = config.resolve_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    gdelt_source = "local_raw"

    market = fetch_market_data(config)
    try:
        gdelt = fetch_local_raw_gdelt_daily(config)
        if gdelt.empty:
            raise RuntimeError("Local raw GDELT aggregation returned no rows.")
    except Exception as raw_exc:
        gdelt_source = "api"
        try:
            gdelt = fetch_gdelt_daily(config)
        except Exception as api_exc:
            gdelt_source = "local_fallback"
            gdelt = _build_local_gdelt_fallback(config.start_date, config.resolve_end_date())
            if gdelt.empty:
                warnings.append(
                    f"Local raw GDELT failed ({raw_exc}); API failed ({api_exc}); fallback was empty."
                )
            else:
                warnings.append(
                    f"Local raw GDELT failed ({raw_exc}); API failed ({api_exc}); local fallback feed was used."
                )
    combined = market.join(gdelt, how="outer").sort_index()

    market_path = output_dir / "market_nodes_daily.csv"
    gdelt_path = output_dir / "gdelt_daily.csv"
    combined_path = output_dir / "combined_feed.csv"
    metadata_path = output_dir / "metadata.json"

    market.to_csv(market_path)
    gdelt.to_csv(gdelt_path)
    combined.to_csv(combined_path)

    metadata = {
        "start_date": config.start_date,
        "end_date": config.resolve_end_date(),
        "market_rows": int(len(market)),
        "gdelt_rows": int(len(gdelt)),
        "combined_rows": int(len(combined)),
        "market_columns": list(market.columns),
        "gdelt_columns": list(gdelt.columns),
        "gdelt_source": gdelt_source,
        "warnings": warnings,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "market_rows": len(market),
        "gdelt_rows": len(gdelt),
        "combined_rows": len(combined),
        "market_path": str(market_path),
        "gdelt_path": str(gdelt_path),
        "combined_path": str(combined_path),
        "gdelt_source": gdelt_source,
        "warnings": warnings,
    }

"""
DailyPipeline: orchestrates data -> LLM -> stat -> prediction for each date.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data_loader import (
    build_context_packet,
    detect_regime,
    extract_headline_from_url,
    prepare_dataset,
)
from llm_tasks import LLMAnalyst
from market_agents import MarketSimulation, encode_simulation_features
from meta_learner import MetaLearner
from stat_engine import RegimeConditionalEngine


def _preload_gdelt_index(
    filepath: str,
    start_date=None,
    end_date=None,
    max_headlines_per_day: int = config.MAX_HEADLINES_PER_DAY,
    chunk_size: int = 200_000,
) -> Optional[Dict]:
    """Pre-load GDELT file with chunked reading and build a {date: [headlines]} dict.

    Uses chunked reading to avoid loading entire multi-GB files into RAM.
    Only processes rows within [start_date, end_date].
    """
    if not os.path.exists(filepath):
        return None

    # Precompute integer date bounds for fast row filtering
    start_int = None
    end_int = None
    if start_date is not None:
        start_int = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
    if end_date is not None:
        end_int = int(pd.to_datetime(end_date).strftime("%Y%m%d"))

    try:
        cols = ["SQLDATE", "SOURCEURL", "AvgTone", "GoldsteinScale", "NumMentions"]
        chunks_in_range = []

        reader = pd.read_csv(
            filepath, usecols=cols, dtype={"SQLDATE": str},
            low_memory=True, chunksize=chunk_size,
        )
        for chunk in reader:
            sql_int = pd.to_numeric(chunk["SQLDATE"], errors="coerce")
            chunk_min = sql_int.min()
            chunk_max = sql_int.max()

            # Early exit: file is sorted so if chunk starts after end, we're done
            if end_int is not None and chunk_min > end_int:
                break
            # Skip chunks entirely before start
            if start_int is not None and chunk_max < start_int:
                continue

            # Filter to exact range
            mask = pd.Series(True, index=chunk.index)
            if start_int is not None:
                mask &= sql_int >= start_int
            if end_int is not None:
                mask &= sql_int <= end_int

            filtered = chunk[mask].copy()
            if not filtered.empty:
                chunks_in_range.append(filtered)

        if not chunks_in_range:
            return {}

        # Combine filtered chunks and process vectorially
        df = pd.concat(chunks_in_range, ignore_index=True)
        df["_date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce").dt.date
        df.dropna(subset=["_date"], inplace=True)
        df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce").fillna(0)
        df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce").fillna(0)
        df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(1)
        df["_score"] = df["NumMentions"] * df["AvgTone"].abs()

        # Build date index using vectorized groupby (no iterrows on full dataset)
        date_index: Dict = {}
        for dt, group in df.groupby("_date"):
            # Keep only top candidates by score before the slow URL extraction loop
            top = group.nlargest(max_headlines_per_day * 3, "_score")
            headlines = []
            seen: set = set()
            for url, tone, goldstein, mentions in zip(
                top["SOURCEURL"], top["AvgTone"],
                top["GoldsteinScale"], top["NumMentions"]
            ):
                if len(headlines) >= max_headlines_per_day:
                    break
                if not isinstance(url, str):
                    continue
                headline = extract_headline_from_url(url)
                if headline and headline not in seen:
                    seen.add(headline)
                    headlines.append({
                        "headline": headline,
                        "tone": float(tone),
                        "goldstein": float(goldstein),
                        "mentions": int(mentions),
                    })
            date_index[dt] = headlines

        return date_index

    except Exception as e:
        print(f"[Pipeline] Warning: could not preload {filepath}: {e}")
        return None


def _headlines_for_date(gdelt_index: Optional[Dict], target_date, max_headlines=40) -> List[dict]:
    """Fast O(1) headline lookup from pre-built date index."""
    if not gdelt_index:
        return []

    if hasattr(target_date, "date"):
        dt = target_date.date()
    elif isinstance(target_date, str):
        dt = pd.to_datetime(target_date).date()
    else:
        dt = target_date

    return gdelt_index.get(dt, [])[:max_headlines]


# ── Legacy shim (kept for backwards compat, not used in pipeline) ──────────
def _preload_gdelt_index_raw(filepath: str) -> Optional[pd.DataFrame]:
    """Load full GDELT file as DataFrame (slow, for compatibility only)."""
    if not os.path.exists(filepath):
        return None
    try:
        cols = ["SQLDATE", "SOURCEURL", "AvgTone", "GoldsteinScale", "NumMentions"]
        df = pd.read_csv(filepath, usecols=cols, dtype={"SQLDATE": str})
        df["Date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df["DateOnly"] = df["Date"].dt.date
        return df
    except Exception as e:
        print(f"[Pipeline] Warning: could not preload {filepath}: {e}")
        return None


def _headlines_for_date_raw(gdelt_df: Optional[pd.DataFrame], target_date, max_headlines=40) -> List[dict]:
    """Slow DataFrame lookup (legacy, not used in pipeline)."""
    if gdelt_df is None:
        return []

    if hasattr(target_date, "date"):
        dt = target_date.date()
    elif isinstance(target_date, str):
        dt = pd.to_datetime(target_date).date()
    else:
        dt = target_date

    mask = gdelt_df["DateOnly"] == dt
    subset = gdelt_df[mask]
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

    headlines.sort(key=lambda h: h["mentions"] * abs(h["tone"]), reverse=True)
    return headlines[:max_headlines]


class DailyPipeline:
    """Orchestrates the full prediction pipeline for a single date."""

    def __init__(
        self,
        df: pd.DataFrame,
        llm: Optional[LLMAnalyst] = None,
        engine: Optional[RegimeConditionalEngine] = None,
        meta: Optional[MetaLearner] = None,
        sim: Optional[MarketSimulation] = None,
        use_llm: bool = True,
        use_agents: bool = True,
        agent_personas: Optional[list] = None,  # None = all 30; pass subset for quick mode
        gdelt_start=None,
        gdelt_end=None,
    ):
        self.df = df
        self.llm = llm if llm is not None else (LLMAnalyst() if use_llm else None)
        self.engine = engine if engine is not None else RegimeConditionalEngine()
        self.meta = meta if meta is not None else MetaLearner()
        self.use_llm = use_llm and (self.llm is not None and self.llm.available)

        # Market simulation (30 Ollama agents, or fewer in quick mode)
        self.sim = sim if sim is not None else (MarketSimulation() if use_agents else None)
        self.use_agents = use_agents and (self.sim is not None and self.sim.available)
        self._agent_personas = agent_personas  # None = use all AGENT_PERSONAS

        # Pre-load GDELT data (only if using LLM or agents — headlines not needed for stat-only)
        # Pass date range to avoid loading all 14M rows of USA GDELT
        self._india_gdelt = None
        self._usa_gdelt = None
        if self.use_llm or self.use_agents:
            import sys
            print("[Pipeline] Pre-loading GDELT data...", flush=True)
            self._india_gdelt = _preload_gdelt_index(
                config.INDIA_NEWS_PATH, start_date=gdelt_start, end_date=gdelt_end
            )
            self._usa_gdelt = _preload_gdelt_index(
                config.USA_NEWS_PATH, start_date=gdelt_start, end_date=gdelt_end
            )
            india_n = len(self._india_gdelt) if self._india_gdelt else 0
            usa_n = len(self._usa_gdelt) if self._usa_gdelt else 0
            print(f"[Pipeline] GDELT loaded: India={india_n} days, USA={usa_n} days", flush=True)

    def fit(self, train_end_idx: int, regime_labels: Optional[pd.Series] = None):
        """Fit the stat engine on data up to train_end_idx."""
        train_df = self.df.iloc[:train_end_idx].copy()

        if regime_labels is None:
            regime_labels = pd.Series(
                [detect_regime(train_df.iloc[i], train_df, i) for i in range(len(train_df))],
                index=train_df.index,
            )

        self.engine.fit(train_df, regime_labels)

    def predict_single(self, target_idx: int) -> Dict:
        """Generate prediction for a single date (target_idx).

        Uses data up to target_idx - 1 for features and LLM context.
        """
        if target_idx < 2 or target_idx >= len(self.df):
            return {"error": f"Invalid target_idx: {target_idx}"}

        # Step 1: Build context packet (from data up to target_idx-1)
        headlines = []
        if self.use_llm or self.use_agents:
            date_val = self.df.iloc[target_idx - 1]["Date"]
            headlines += _headlines_for_date(self._india_gdelt, date_val)
            headlines += _headlines_for_date(self._usa_gdelt, date_val)

        context = build_context_packet(self.df, target_idx, headlines if headlines else None)
        if not context:
            return {"error": "Could not build context packet"}

        # Step 2: LLM analysis — 4 analyst tasks (Gemini, if available)
        llm_results = {}
        llm_features = {}
        llm_regime = config.DEFAULT_REGIME

        if self.use_llm:
            llm_results = self.llm.run_all_tasks(context)
            llm_features = self.llm.encode_features(llm_results)
            llm_regime = self.llm.get_regime_label(llm_results)

        # Step 3: Market simulation — 30 Ollama agents (if available)
        sim_result = {}
        sim_features = {}
        if self.use_agents:
            sim_kwargs = {}
            if self._agent_personas is not None:
                sim_kwargs["personas"] = self._agent_personas
            sim_result = self.sim.run(context, **sim_kwargs)
            sim_features = encode_simulation_features(sim_result)

        # Step 4: Statistical regime detection (fallback / blend)
        stat_regime = context.get("stat_regime", config.DEFAULT_REGIME)

        # Use LLM regime if confident, else stat regime
        regime_conf = llm_features.get("regime_confidence", 0.0)
        regime = llm_regime if (self.use_llm and regime_conf > 0.6) else stat_regime

        # Step 5: Prepare feature row for stat model
        feature_row = self.df.iloc[[target_idx - 1]].copy()

        # Add LLM analyst features (weighted by meta-learner)
        llm_weight = self.meta.get_regime_weight(regime)
        for feat_name, feat_val in llm_features.items():
            feature_row[feat_name] = feat_val * llm_weight

        # Add simulation features (separately weighted — agents start at same 5%)
        agent_weight = self.meta.get_regime_weight(regime)  # reuse meta weight for now
        for feat_name, feat_val in sim_features.items():
            feature_row[feat_name] = feat_val * agent_weight

        # Step 6: Statistical prediction
        if not self.engine.fitted:
            return {"error": "Engine not fitted"}

        try:
            pred, std_err, (lower, upper) = self.engine.predict(feature_row, regime)
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

        # Assemble result
        actual = self.df.iloc[target_idx].get("target")
        if pd.isna(actual):
            actual = self.df.iloc[target_idx]["INR"]

        result = {
            "date": str(self.df.iloc[target_idx]["Date"].date())
            if hasattr(self.df.iloc[target_idx]["Date"], "date")
            else str(self.df.iloc[target_idx]["Date"]),
            "prediction": round(float(pred), 4),
            "actual": round(float(actual), 4) if pd.notna(actual) else None,
            "std_error": round(float(std_err), 6),
            "ci_lower": round(float(lower), 4),
            "ci_upper": round(float(upper), 4),
            "regime": regime,
            "regime_source": "llm" if (self.use_llm and regime_conf > 0.6) else "stat",
            "llm_weight": round(llm_weight, 4),
            "llm_features_used": len([v for v in llm_features.values() if v != 0.0]),
            "sim_consensus": sim_result.get("consensus_direction", "n/a"),
            "sim_strength": sim_result.get("consensus_strength", 0.0),
            "sim_entropy": sim_result.get("entropy", 0.0),
            "sim_n_agents": sim_result.get("n_success", 0),
        }

        # Step 8: Update meta-learner
        if actual is not None and pd.notna(actual):
            error = pred - float(actual)
            self.meta.record(regime, error, llm_features)

        return result

    def predict_range(
        self, start_idx: int, end_idx: int, refit_freq: int = config.REFIT_FREQUENCY
    ) -> List[Dict]:
        """Run predictions for a range of dates with periodic refitting."""
        results = []
        last_fit_idx = start_idx

        for idx in range(start_idx, min(end_idx, len(self.df))):
            # Refit if needed
            if idx - last_fit_idx >= refit_freq or not self.engine.fitted:
                self.fit(idx)
                last_fit_idx = idx

            result = self.predict_single(idx)
            results.append(result)

            if "error" not in result and result.get("actual") is not None:
                actual = result["actual"]
                pred = result["prediction"]
                err = abs(pred - actual)
                if idx % 50 == 0:
                    agents_str = f" agents={result.get('sim_n_agents', 0)}" if self.use_agents else ""
                    print(
                        f"  [{result['date']}] pred={pred:.4f} actual={actual:.4f} "
                        f"err={err:.4f} regime={result['regime']}{agents_str}",
                        flush=True,
                    )

        return results

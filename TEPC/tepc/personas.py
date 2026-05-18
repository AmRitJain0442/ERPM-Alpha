"""
Deterministic multi-persona market engine for TEPC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List
import math

import numpy as np
import pandas as pd


PERSONAS = [
    {"name": "Dollar Rates Trader", "weight": 1.15},
    {"name": "Oil Shock Analyst", "weight": 1.05},
    {"name": "India News Analyst", "weight": 0.95},
    {"name": "Global Risk Sentinel", "weight": 1.0},
    {"name": "Cross-FX Relative Value", "weight": 0.9},
    {"name": "Topology Synchronization Analyst", "weight": 1.1},
    {"name": "Chaos Regime Analyst", "weight": 1.15},
]


@dataclass
class MarketMemorySnapshot:
    as_of_date: str
    target_date: str
    regime: str
    macro_state: Dict[str, float]
    news_state: Dict[str, float]
    topology_state: Dict[str, float]
    chaos_state: Dict[str, float]
    cross_asset_state: Dict[str, float]
    memory_features: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PersonaVote:
    persona: str
    direction: str
    expected_return: float
    magnitude: float
    confidence: float
    thesis: str
    risk_flags: List[str]
    base_weight: float
    calibration_weight: float

    def to_dict(self) -> Dict:
        return asdict(self)


class PersonaMemoryStore:
    def __init__(self) -> None:
        self.records: Dict[str, List[Dict[str, float]]] = {}

    def calibration_weight(self, persona: str) -> float:
        history = self.records.get(persona, [])
        if len(history) < 5:
            return 1.0
        hit_rate = float(np.mean([row["direction_hit"] for row in history[-20:]]))
        mean_abs_error = float(np.mean([abs(row["error"]) for row in history[-20:]]))
        weight = 0.8 + (hit_rate - 0.5) * 1.4 - mean_abs_error * 60.0
        return float(np.clip(weight, 0.45, 1.55))

    def record_votes(self, votes: List[PersonaVote], actual_return: float, actual_label: str | None = None) -> None:
        actual_sign = int(np.sign(actual_return))
        for vote in votes:
            if isinstance(vote, dict):
                vote = PersonaVote(**vote)
            pred_sign = int(np.sign(vote.expected_return))
            if actual_label is not None:
                direction_hit = float(vote.direction == actual_label)
            else:
                direction_hit = float(pred_sign == actual_sign) if actual_sign != 0 else float(pred_sign == 0)
            self.records.setdefault(vote.persona, []).append(
                {
                    "direction_hit": direction_hit,
                    "error": float(vote.expected_return - actual_return),
                    "confidence": float(vote.confidence),
                }
            )

    def summary(self) -> Dict:
        if not self.records:
            return {
                "coverage": 0,
                "mean_directional_hit_rate": 0.5,
                "mean_abs_bias": 0.0,
                "per_persona": {},
            }

        per_persona = {}
        hit_rates = []
        abs_biases = []
        for persona, history in self.records.items():
            hit_rate = float(np.mean([row["direction_hit"] for row in history]))
            mean_bias = float(np.mean([row["error"] for row in history]))
            per_persona[persona] = {
                "n": len(history),
                "hit_rate": hit_rate,
                "mean_bias": mean_bias,
            }
            hit_rates.append(hit_rate)
            abs_biases.append(abs(mean_bias))

        return {
            "coverage": len(per_persona),
            "mean_directional_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.5,
            "mean_abs_bias": float(np.mean(abs_biases)) if abs_biases else 0.0,
            "per_persona": per_persona,
        }


def _safe(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        value = float(value)
    except Exception:
        return float(default)
    if np.isnan(value) or np.isinf(value):
        return float(default)
    return value


class MarketMemoryBuilder:
    def build(self, frame: pd.DataFrame, row_idx: int) -> MarketMemorySnapshot:
        row = frame.iloc[row_idx]
        history = frame.iloc[max(0, row_idx - 40) : row_idx + 1]
        realized_move = history["current_rate"].pct_change().abs().dropna().tail(20)
        expected_abs_move = float(realized_move.mean()) if not realized_move.empty else 0.003
        if not np.isfinite(expected_abs_move) or expected_abs_move <= 0:
            expected_abs_move = 0.003

        macro_state = {
            "dxy_shock": _safe(row, "dxy_shock_1"),
            "us10y_shock": _safe(row, "us10y_shock_1"),
            "brent_shock": _safe(row, "brent_shock_1"),
            "gold_shock": _safe(row, "gold_shock_1"),
            "brent_dxy_interaction": _safe(row, "brent_dxy_interaction"),
        }
        news_state = {
            "india_news": _safe(row, "live_india_fx_news_mean_5"),
            "usd_macro_news": _safe(row, "live_usd_macro_news_mean_5"),
            "geo_risk_news": _safe(row, "live_geo_risk_mean_5"),
            "india_goldstein": _safe(row, "india_fx_goldstein"),
            "usd_goldstein": _safe(row, "usd_macro_goldstein"),
            "geo_risk_goldstein": _safe(row, "geo_risk_goldstein"),
            "india_articles": _safe(row, "india_fx_articles"),
            "usd_articles": _safe(row, "usd_macro_articles"),
        }
        topology_state = {
            "target_degree": _safe(row, "target_degree"),
            "target_mean_corr": _safe(row, "target_mean_corr"),
            "network_tension": _safe(row, "network_tension"),
            "fiedler_value": _safe(row, "fiedler_value"),
            "spectral_entropy": _safe(row, "spectral_entropy"),
        }
        chaos_state = {
            "sync_mean": _safe(row, "chaos_sync_mean"),
            "target_sync_final": _safe(row, "chaos_target_sync_final"),
            "ftle": _safe(row, "chaos_ftle"),
            "coupling_response": _safe(row, "chaos_coupling_response"),
            "target_x_span": _safe(row, "chaos_target_x_span"),
        }
        cross_asset_state = {
            "eurusd_shock": _safe(row, "eurusd_shock_1"),
            "gbpinr_shock": _safe(row, "gbpinr_shock_1"),
            "goldstein_spread": _safe(row, "goldstein_spread_mean_5"),
        }

        regime = self._detect_regime(macro_state, news_state, topology_state, chaos_state)
        memory_features = {
            "expected_abs_move": expected_abs_move,
            "macro_pressure": macro_state["dxy_shock"] + 0.7 * macro_state["us10y_shock"] + 0.4 * macro_state["brent_shock"],
            "news_pressure": news_state["india_news"] + 0.8 * news_state["geo_risk_news"] + 0.5 * news_state["usd_macro_news"],
            "stress_score": topology_state["network_tension"] + chaos_state["ftle"] + 0.2 * abs(news_state["geo_risk_goldstein"]),
        }

        return MarketMemorySnapshot(
            as_of_date=str(pd.Timestamp(row.name).date()),
            target_date=str(pd.Timestamp(row["target_date"]).date()),
            regime=regime,
            macro_state=macro_state,
            news_state=news_state,
            topology_state=topology_state,
            chaos_state=chaos_state,
            cross_asset_state=cross_asset_state,
            memory_features=memory_features,
        )

    @staticmethod
    def _detect_regime(macro_state: Dict[str, float], news_state: Dict[str, float], topology_state: Dict[str, float], chaos_state: Dict[str, float]) -> str:
        if news_state["geo_risk_news"] > 1.5 or topology_state["network_tension"] > 0.18:
            return "stress"
        if chaos_state["target_sync_final"] > 0.78 and macro_state["dxy_shock"] > 0:
            return "synchronized_up"
        if chaos_state["target_sync_final"] > 0.78 and macro_state["dxy_shock"] < 0:
            return "synchronized_down"
        return "mixed"


def _score_to_vote(score: float, expected_abs_move: float, breakout_threshold: float) -> tuple[str, float, float]:
    magnitude = float(np.clip(abs(score), 0.0, 1.0))
    confidence = float(np.clip(0.45 + 0.45 * magnitude, 0.05, 0.98))
    if score > 0.08:
        direction = "up"
        expected_return = expected_abs_move * magnitude
    elif score < -0.08:
        direction = "down"
        expected_return = -expected_abs_move * magnitude
    else:
        direction = "range"
        expected_return = 0.0
    move_gate = breakout_threshold * 0.55
    if abs(expected_return) < move_gate or magnitude < 0.14:
        direction = "range"
        expected_return = 0.0
        confidence = float(np.clip(0.40 + 0.20 * magnitude, 0.05, 0.85))
    return direction, float(expected_return), confidence


class RulePersonaEngine:
    def __init__(self, breakout_threshold: float = 0.005):
        self.breakout_threshold = breakout_threshold

    def run(self, snapshot: MarketMemorySnapshot, store: PersonaMemoryStore) -> Dict:
        votes: List[PersonaVote] = []
        for persona in PERSONAS:
            score, thesis, risk_flags = self._persona_score(persona["name"], snapshot)
            direction, expected_return, confidence = _score_to_vote(
                score,
                snapshot.memory_features["expected_abs_move"],
                self.breakout_threshold,
            )
            votes.append(
                PersonaVote(
                    persona=persona["name"],
                    direction=direction,
                    expected_return=expected_return,
                    magnitude=abs(score),
                    confidence=confidence,
                    thesis=thesis,
                    risk_flags=risk_flags,
                    base_weight=float(persona["weight"]),
                    calibration_weight=store.calibration_weight(persona["name"]),
                )
            )
        return self._aggregate(votes)

    def _persona_score(self, name: str, snapshot: MarketMemorySnapshot) -> tuple[float, str, List[str]]:
        macro = snapshot.macro_state
        news = snapshot.news_state
        topo = snapshot.topology_state
        chaos = snapshot.chaos_state
        cross = snapshot.cross_asset_state
        mem = snapshot.memory_features

        if name == "Dollar Rates Trader":
            score = 0.45 * macro["dxy_shock"] + 0.35 * macro["us10y_shock"] - 0.15 * cross["eurusd_shock"]
            thesis = "Dollar and rates impulse dominate the next response window."
            flags = ["dxy", "rates"]
        elif name == "Oil Shock Analyst":
            score = 0.55 * macro["brent_shock"] + 0.20 * news["geo_risk_news"] - 0.10 * macro["gold_shock"]
            thesis = "Imported oil stress remains the cleanest INR vulnerability channel."
            flags = ["oil", "geo_risk"]
        elif name == "India News Analyst":
            score = 0.40 * news["india_news"] + 0.18 * abs(min(news["india_goldstein"], 0.0)) + 0.06 * np.log1p(max(news["india_articles"], 0.0))
            thesis = "India-specific news heat and event intensity skew the local stress balance."
            flags = ["india_news"]
        elif name == "Global Risk Sentinel":
            score = 0.35 * news["geo_risk_news"] + 0.20 * news["usd_macro_news"] + 0.20 * macro["dxy_shock"]
            thesis = "Global risk repricing and safe-haven demand drive the move."
            flags = ["risk", "usd_macro"]
        elif name == "Cross-FX Relative Value":
            score = -0.35 * cross["eurusd_shock"] + 0.35 * cross["gbpinr_shock"] + 0.10 * macro["dxy_shock"]
            thesis = "Cross-currency relative value suggests how isolated the INR move really is."
            flags = ["eurusd", "gbpinr"]
        elif name == "Topology Synchronization Analyst":
            score = 0.18 * topo["target_degree"] + 0.20 * topo["target_mean_corr"] + 0.16 * topo["network_tension"] - 0.14 * topo["fiedler_value"]
            thesis = "A tighter synchronized macro network raises continuation pressure."
            flags = ["topology"]
        else:
            chaos_direction = np.tanh(8.0 * chaos["coupling_response"])
            score = 0.30 * chaos_direction + 0.22 * chaos["target_sync_final"] + 0.10 * chaos["ftle"] + 0.08 * np.sign(mem["macro_pressure"]) * chaos["sync_mean"]
            thesis = "Chaotic synchronization indicates whether stress is cohering into a directional state."
            flags = ["chaos"]

        return float(np.clip(score, -1.0, 1.0)), thesis, flags

    @staticmethod
    def _aggregate(votes: List[PersonaVote]) -> Dict:
        if not votes:
            return {
                "votes": [],
                "expected_return": 0.0,
                "direction_score": 0.0,
                "confidence": 0.0,
                "entropy": 1.0,
                "consensus": "range",
                "breakout_probabilities": {"down": 0.0, "range": 1.0, "up": 0.0},
            }

        weights = np.array([vote.base_weight * vote.calibration_weight * vote.confidence for vote in votes], dtype=float)
        weights = weights / weights.sum()
        returns = np.array([vote.expected_return for vote in votes], dtype=float)
        final_return = float(np.dot(weights, returns))
        signs = np.sign(returns)
        direction_score = float(np.dot(weights, signs))

        up_prob = float(weights[signs > 0].sum())
        down_prob = float(weights[signs < 0].sum())
        range_prob = float(weights[signs == 0].sum())
        probs = np.array([down_prob, range_prob, up_prob], dtype=float)
        nonzero = probs[probs > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero + 1e-12)) / math.log(3))
        consensus = "up" if direction_score > 0.05 else ("down" if direction_score < -0.05 else "range")

        return {
            "vote_objects": votes,
            "votes": [vote.to_dict() for vote in votes],
            "expected_return": final_return,
            "direction_score": direction_score,
            "confidence": float(np.dot(weights, np.array([vote.confidence for vote in votes], dtype=float))),
            "entropy": entropy,
            "consensus": consensus,
            "breakout_probabilities": {"down": down_prob, "range": range_prob, "up": up_prob},
        }

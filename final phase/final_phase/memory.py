"""
Shared market memory structures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketMemorySnapshot:
    as_of_date: str
    target_date: str
    regime: str
    macro_state: Dict
    sentiment_state: Dict
    volatility_state: Dict
    latent_state: Dict
    persona_state: Dict
    memory_features: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

    def prompt_payload(self) -> Dict:
        return {
            "as_of_date": self.as_of_date,
            "target_date": self.target_date,
            "regime": self.regime,
            "macro_state": self.macro_state,
            "sentiment_state": self.sentiment_state,
            "volatility_state": self.volatility_state,
            "latent_state": self.latent_state,
            "persona_state": self.persona_state,
        }


class MarketMemoryBuilder:
    """
    Converts the engineered frame into a compact market-state snapshot.
    """

    def build(
        self,
        frame: pd.DataFrame,
        row_idx: int,
        persona_state: Optional[Dict] = None,
    ) -> MarketMemorySnapshot:
        history = frame.iloc[max(0, row_idx - 60): row_idx + 1]
        row = frame.iloc[row_idx]

        regime = self._detect_regime(row)
        expected_abs_move = float(
            history["raw_return_1d"].abs().dropna().tail(20).mean()
            if "raw_return_1d" in history
            else row.get("expected_abs_move_proxy", 0.003)
        )
        if np.isnan(expected_abs_move) or expected_abs_move <= 0:
            expected_abs_move = float(row.get("expected_abs_move_proxy", 0.003) or 0.003)

        macro_state = {
            "oil_pressure": round(float(row.get("macro_oil_dxy_pressure", 0.0)), 6),
            "dollar_pressure": round(float(row.get("macro_rates_dollar_pressure", 0.0)), 6),
            "gold_risk_divergence": round(float(row.get("macro_gold_risk_divergence", 0.0)), 6),
            "carry_pressure": round(float(row.get("mem_carry_pressure", 0.0)), 6),
        }
        sentiment_state = {
            "india_tone": round(float(row.get("sent_in_tone_5d_mean", 0.0)), 6),
            "india_panic": round(float(row.get("sent_in_panic_5d_mean", 0.0)), 6),
            "us_tone": round(float(row.get("sent_us_tone_5d_mean", 0.0)), 6),
            "diff_tone": round(float(row.get("sent_diff_tone_5d_mean", 0.0)), 6),
            "event_heat": round(float(row.get("mem_event_heat", 0.0)), 6),
            "goldstein_diff": round(float(row.get("gold_diff_level", 0.0)), 6),
        }
        volatility_state = {
            "vol_5": round(float(row.get("tech_vol_5", 0.0)), 6),
            "vol_20": round(float(row.get("tech_vol_20", 0.0)), 6),
            "vol_stress": round(float(row.get("mem_vol_stress", 1.0)), 6),
            "expected_abs_move": round(expected_abs_move, 6),
            "trend_5_20": round(float(row.get("tech_trend_5_20", 0.0)), 6),
            "reversal_pressure": round(float(row.get("mem_reversal_pressure", 0.0)), 6),
        }
        latent_state = {
            "macro_pressure": round(float(row.get("mem_macro_pressure", 0.0)), 6),
            "sentiment_pressure": round(float(row.get("mem_sentiment_pressure", 0.0)), 6),
            "regime_score": round(float(row.get("mem_regime_score", 0.0)), 6),
            "range_position": round(float(row.get("tech_range_pos_20", 0.5)), 6),
        }
        persona_state = persona_state or {
            "coverage": 0,
            "mean_directional_hit_rate": 0.5,
            "mean_abs_bias": 0.0,
        }
        memory_features = {
            "expected_abs_move": expected_abs_move,
            "macro_pressure": float(row.get("mem_macro_pressure", 0.0)),
            "sentiment_pressure": float(row.get("mem_sentiment_pressure", 0.0)),
            "event_heat": float(row.get("mem_event_heat", 0.0)),
            "vol_stress": float(row.get("mem_vol_stress", 1.0)),
            "carry_pressure": float(row.get("mem_carry_pressure", 0.0)),
            "reversal_pressure": float(row.get("mem_reversal_pressure", 0.0)),
        }

        return MarketMemorySnapshot(
            as_of_date=str(pd.to_datetime(row["Date"]).date()),
            target_date=str(pd.to_datetime(row["target_date"]).date()),
            regime=regime,
            macro_state=macro_state,
            sentiment_state=sentiment_state,
            volatility_state=volatility_state,
            latent_state=latent_state,
            persona_state=persona_state,
            memory_features=memory_features,
        )

    @staticmethod
    def _detect_regime(row: pd.Series) -> str:
        vol_stress = float(row.get("mem_vol_stress", 1.0) or 1.0)
        regime_score = float(row.get("mem_regime_score", 0.0) or 0.0)
        sentiment_pressure = float(row.get("mem_sentiment_pressure", 0.0) or 0.0)
        trend = float(row.get("tech_trend_5_20", 0.0) or 0.0)

        if vol_stress > 1.5 or sentiment_pressure > 1.0:
            return "stress"
        if trend > 0.0025:
            return "trend_up"
        if trend < -0.0025:
            return "trend_down"
        if abs(regime_score) < 0.25:
            return "calm"
        return "mixed"

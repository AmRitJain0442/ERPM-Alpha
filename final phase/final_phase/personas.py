"""
Persona engines and persona memory tracking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import json
import math

import numpy as np

from .config import PERSONAS, RunConfig
from .llm_clients import make_chat_client
from .memory import MarketMemorySnapshot


@dataclass
class PersonaVote:
    backend: str
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
    def __init__(self):
        self.records: Dict[str, List[Dict]] = {}

    def calibration_weight(self, backend: str, persona: str) -> float:
        key = f"{backend}:{persona}"
        history = self.records.get(key, [])
        if len(history) < 5:
            return 1.0
        hit_rate = np.mean([row["direction_hit"] for row in history[-20:]])
        mae = np.mean([abs(row["error"]) for row in history[-20:]])
        weight = 0.8 + (hit_rate - 0.5) * 1.5 - mae * 10
        return float(np.clip(weight, 0.5, 1.5))

    def record(self, vote: PersonaVote, actual_return: float) -> None:
        direction_hit = int(np.sign(vote.expected_return) == np.sign(actual_return))
        error = vote.expected_return - actual_return
        key = f"{vote.backend}:{vote.persona}"
        self.records.setdefault(key, []).append(
            {
                "direction_hit": direction_hit,
                "error": float(error),
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
        biases = []
        for key, history in self.records.items():
            hit_rate = float(np.mean([row["direction_hit"] for row in history]))
            mean_bias = float(np.mean([row["error"] for row in history]))
            per_persona[key] = {
                "n": len(history),
                "hit_rate": hit_rate,
                "mean_bias": mean_bias,
            }
            hit_rates.append(hit_rate)
            biases.append(abs(mean_bias))

        return {
            "coverage": len(per_persona),
            "mean_directional_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.5,
            "mean_abs_bias": float(np.mean(biases)) if biases else 0.0,
            "per_persona": per_persona,
        }


def _score_to_vote(score: float, expected_abs_move: float) -> tuple[str, float, float]:
    magnitude = float(np.clip(abs(score), 0.0, 1.0))
    confidence = float(np.clip(0.45 + abs(score) * 0.45, 0.05, 0.98))
    if score > 0.08:
        direction = "up"
    elif score < -0.08:
        direction = "down"
    else:
        direction = "flat"
    expected_return = 0.0
    if direction == "up":
        expected_return = expected_abs_move * magnitude
    elif direction == "down":
        expected_return = -expected_abs_move * magnitude
    return direction, expected_return, confidence


class RulePersonaEngine:
    backend = "rule"

    def __init__(self, config: RunConfig):
        self.config = config
        self.personas = PERSONAS[: config.persona_limit]

    def run(self, snapshot: MarketMemorySnapshot, store: PersonaMemoryStore) -> Dict:
        votes = []
        emove = snapshot.memory_features["expected_abs_move"]
        for persona in self.personas:
            score, thesis, risk_flags = self._persona_score(persona["name"], snapshot)
            direction, expected_return, confidence = _score_to_vote(score, emove)
            cal = store.calibration_weight(self.backend, persona["name"])
            votes.append(
                PersonaVote(
                    backend=self.backend,
                    persona=persona["name"],
                    direction=direction,
                    expected_return=expected_return,
                    magnitude=abs(score),
                    confidence=confidence,
                    thesis=thesis,
                    risk_flags=risk_flags,
                    base_weight=persona["weight"],
                    calibration_weight=cal,
                )
            )
        return self._aggregate(votes)

    def _persona_score(self, name: str, snapshot: MarketMemorySnapshot) -> tuple[float, str, List[str]]:
        macro = snapshot.macro_state
        sent = snapshot.sentiment_state
        vol = snapshot.volatility_state
        latent = snapshot.latent_state

        if name == "Macro Rates Strategist":
            score = macro["dollar_pressure"] + 0.5 * macro["carry_pressure"]
            thesis = "Dollar and rates pressure dominate the near-term FX balance."
            flags = ["dxy", "rates"]
        elif name == "Oil And Commodities Analyst":
            score = macro["oil_pressure"] - 0.2 * macro["gold_risk_divergence"]
            thesis = "Oil shock transmission is the main imported stress channel for INR."
            flags = ["oil"]
        elif name == "India Sentiment Analyst":
            score = 0.6 * sent["india_panic"] - 0.12 * sent["india_tone"] + 0.1 * sent["event_heat"]
            thesis = "India-specific tone and panic skew the market toward local stress pricing."
            flags = ["india_news"]
        elif name == "Global Risk Analyst":
            score = 0.15 * macro["dollar_pressure"] - 0.08 * sent["us_tone"] + 0.1 * sent["event_heat"]
            thesis = "Global risk balance and safe-haven demand drive the short-horizon move."
            flags = ["global_risk"]
        elif name == "Technical Trend Analyst":
            score = 40 * vol["trend_5_20"] + 0.25 * latent["regime_score"]
            thesis = "Trend pressure and regime persistence favor continuation."
            flags = ["trend"]
        elif name == "Mean Reversion Analyst":
            score = -0.3 * latent["range_position"] - 0.35 * vol["reversal_pressure"]
            thesis = "The move looks stretched enough that reversal risk matters."
            flags = ["reversion"]
        elif name == "Carry And Flow Trader":
            score = -0.35 * macro["carry_pressure"] + 0.15 * sent["diff_tone"]
            thesis = "Carry and relative stability shape how sticky the move can be."
            flags = ["carry"]
        else:
            score = 0.4 * vol["vol_stress"] + 0.25 * sent["event_heat"] + 0.2 * sent["india_panic"]
            thesis = "Tail-risk management demands extra caution under stress and event heat."
            flags = ["risk"]

        return float(np.clip(score, -1.0, 1.0)), thesis, flags

    def _aggregate(self, votes: List[PersonaVote]) -> Dict:
        if not votes:
            return {
                "backend": self.backend,
                "votes": [],
                "expected_return": 0.0,
                "direction_score": 0.0,
                "confidence": 0.0,
                "entropy": 1.0,
            }
        weights = np.array([v.base_weight * v.calibration_weight * v.confidence for v in votes], dtype=float)
        weights = weights / weights.sum()
        returns = np.array([v.expected_return for v in votes], dtype=float)
        final_return = float(np.dot(weights, returns))
        signs = np.sign(returns)
        direction_score = float(np.dot(weights, signs))
        up_w = float(weights[signs > 0].sum())
        down_w = float(weights[signs < 0].sum())
        flat_w = float(weights[signs == 0].sum())
        probs = np.array([up_w, down_w, flat_w], dtype=float)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(3))
        confidence = float(np.dot(weights, np.array([v.confidence for v in votes], dtype=float)))
        consensus = "up" if direction_score > 0.05 else ("down" if direction_score < -0.05 else "flat")
        return {
            "backend": self.backend,
            "votes": [v.to_dict() for v in votes],
            "expected_return": final_return,
            "direction_score": direction_score,
            "confidence": confidence,
            "entropy": entropy,
            "consensus": consensus,
        }


class LLMPersonaEngine:
    def __init__(self, backend_spec: str, config: RunConfig):
        self.backend = backend_spec
        self.config = config
        self.personas = PERSONAS[: config.persona_limit]
        self.client = make_chat_client(backend_spec)
        self.available = self.client is not None and self.client.available

    def run(self, snapshot: MarketMemorySnapshot, store: PersonaMemoryStore) -> Dict:
        if not self.available:
            return {
                "backend": self.backend,
                "votes": [],
                "expected_return": 0.0,
                "direction_score": 0.0,
                "confidence": 0.0,
                "entropy": 1.0,
                "consensus": "flat",
                "skipped": True,
            }

        votes = []
        payload = json.dumps(snapshot.prompt_payload(), indent=2)
        for persona in self.personas:
            system_prompt = (
                "You are one specialized FX market persona. "
                "Do not anchor on a raw spot price. Use only the supplied structured market memory. "
                "Respond in JSON only."
            )
            user_prompt = f"""
Persona: {persona["name"]}
Style: {persona["description"]}

Structured market memory:
{payload}

Return ONLY this JSON:
{{
  "direction": "up|down|flat",
  "magnitude": 0.0,
  "confidence": 0.0,
  "thesis": "one concise sentence",
  "risk_flags": ["..."]
}}

Direction convention:
- up = USD/INR higher
- down = USD/INR lower
- flat = no edge
"""
            completion = self.client.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            parsed = completion.json_payload if completion else None
            if not parsed:
                continue
            direction = str(parsed.get("direction", "flat")).strip().lower()
            if direction not in {"up", "down", "flat"}:
                direction = "flat"
            magnitude = float(np.clip(parsed.get("magnitude", 0.0), 0.0, 1.0))
            confidence = float(np.clip(parsed.get("confidence", 0.0), 0.05, 0.99))
            thesis = str(parsed.get("thesis", "")).strip()[:240]
            risk_flags = parsed.get("risk_flags", [])
            if not isinstance(risk_flags, list):
                risk_flags = [str(risk_flags)]

            emove = snapshot.memory_features["expected_abs_move"]
            expected_return = 0.0
            if direction == "up":
                expected_return = emove * magnitude
            elif direction == "down":
                expected_return = -emove * magnitude

            cal = store.calibration_weight(self.backend, persona["name"])
            votes.append(
                PersonaVote(
                    backend=self.backend,
                    persona=persona["name"],
                    direction=direction,
                    expected_return=expected_return,
                    magnitude=magnitude,
                    confidence=confidence,
                    thesis=thesis,
                    risk_flags=[str(x) for x in risk_flags[:5]],
                    base_weight=persona["weight"],
                    calibration_weight=cal,
                )
            )

        if not votes:
            return {
                "backend": self.backend,
                "votes": [],
                "expected_return": 0.0,
                "direction_score": 0.0,
                "confidence": 0.0,
                "entropy": 1.0,
                "consensus": "flat",
                "skipped": True,
            }
        result = RulePersonaEngine(self.config)._aggregate(votes)
        result["backend"] = self.backend
        return result

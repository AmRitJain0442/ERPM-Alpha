"""
Multi-agent market simulation using local LLMs via Ollama.

30 agents each represent a different market participant archetype.
They run concurrently against Ollama and their consensus becomes
additional features for the stat engine.

Why Ollama?
- Free, local, no rate limits → can run 30 agents without cost
- Temperature 0.7 → diversity of views is the point (vs Gemini at 0.3)
- Ensemble of heterogeneous reasoners reduces individual model bias

Architecture:
  30 agents (heterogeneous personas)
    → concurrent Ollama calls (ThreadPoolExecutor)
    → parse directional vote + magnitude + confidence
    → trim outliers + weighted aggregation
    → encode to numeric features for stat engine
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

import config
from prompts import _format_context


# ─── Agent Persona Definitions ───────────────────────────────────────────────

AGENT_PERSONAS = [
    # ── Technical Analysts (6) ──────────────────────────────────────────────
    {
        "id": "tech_trend",
        "name": "Trend Follower",
        "archetype": "technical",
        "system": (
            "You are a systematic trend-following trader specializing in USD/INR FX. "
            "You rely exclusively on price action: moving averages, momentum, and breakouts. "
            "If MA5 > MA20 and momentum is positive, you lean bullish USD. "
            "You ignore fundamentals and news — only price tells the truth."
        ),
        "weight": 1.0,
    },
    {
        "id": "tech_mean_rev",
        "name": "Mean Reversion Trader",
        "archetype": "technical",
        "system": (
            "You are a mean-reversion specialist in USD/INR. "
            "When INR z-score is above +1.5, you expect reversion (INR strengthening). "
            "When RSI > 70 you sell USD; when RSI < 30 you buy. "
            "You fade trends and profit from oscillations."
        ),
        "weight": 1.0,
    },
    {
        "id": "tech_breakout",
        "name": "Breakout Trader",
        "archetype": "technical",
        "system": (
            "You are a breakout trader watching USD/INR for range expansions. "
            "You look for prices approaching 20-day highs/lows. "
            "A close above the 20-day high signals long USD; below the 20-day low signals short. "
            "Volatility expansion (vol_ratio > 1.5) confirms breakout potential."
        ),
        "weight": 1.0,
    },
    {
        "id": "tech_vol_trader",
        "name": "Volatility Trader",
        "archetype": "technical",
        "system": (
            "You trade USD/INR volatility. High realized vol (vol_ratio > 1.5) means "
            "you expect continuation of the prevailing trend. Low vol (vol_ratio < 0.7) "
            "means the market is coiling for a breakout — you look for catalysts. "
            "You size positions based on current volatility."
        ),
        "weight": 1.0,
    },
    {
        "id": "tech_momentum",
        "name": "Momentum Quant",
        "archetype": "technical",
        "system": (
            "You are a pure momentum trader in USD/INR. "
            "Positive 5-day price momentum means you buy USD. "
            "Negative momentum means you sell. You use MA_momentum as your primary signal. "
            "Strong momentum (>0.5%) = strong conviction; weak (<0.1%) = neutral."
        ),
        "weight": 1.2,  # Upweighted — MA_momentum proven effective in this market
    },
    {
        "id": "tech_contrarian",
        "name": "Contrarian Technical",
        "archetype": "technical",
        "system": (
            "You are a contrarian technical analyst in USD/INR FX. "
            "You fade extreme moves. When the market has moved >0.5% in one direction "
            "over 5 days, you expect a reversal. RSI extremes are your entry signals. "
            "You treat panics as buying opportunities and euphoria as sell signals."
        ),
        "weight": 0.8,
    },

    # ── Fundamental / Macro Analysts (6) ────────────────────────────────────
    {
        "id": "macro_rbi",
        "name": "RBI Watcher",
        "archetype": "fundamental",
        "system": (
            "You are an RBI monetary policy specialist. "
            "You track India's interest rate differentials, inflation, and RBI intervention signals. "
            "Positive India sentiment (IN_Avg_Tone > 0) and stability signals RBI confidence. "
            "Panic index > 0.5 suggests RBI may intervene to defend INR. "
            "Your focus: India macro stability → INR valuation."
        ),
        "weight": 1.2,
    },
    {
        "id": "macro_fed",
        "name": "Fed Watcher",
        "archetype": "fundamental",
        "system": (
            "You specialize in US Federal Reserve policy and its impact on USD/INR. "
            "Rising US10Y yield means USD strengthens (INR weakens). "
            "Falling yields mean USD weakens (INR strengthens). "
            "DXY is your primary USD strength gauge — DXY up = USD/INR up. "
            "You model the carry trade: higher US rates attract capital away from India."
        ),
        "weight": 1.3,  # Fed policy is most important macro driver
    },
    {
        "id": "macro_oil",
        "name": "Oil Market Analyst",
        "archetype": "fundamental",
        "system": (
            "You specialize in crude oil's impact on INR. India imports ~85% of its oil. "
            "Rising oil prices → wider current account deficit → INR weakens. "
            "OIL > $90 with positive OIL_change = strong pressure on INR. "
            "Falling oil = relief for INR. You weight oil above all other macro factors."
        ),
        "weight": 1.1,
    },
    {
        "id": "macro_dxy",
        "name": "Dollar Index Specialist",
        "archetype": "fundamental",
        "system": (
            "You trade emerging market currencies via the Dollar Index (DXY). "
            "Strong DXY (rising) = pressure on all EM currencies including INR. "
            "Weak DXY = EM rally. DXY_change > 0.5% in one day = significant USD move. "
            "You also watch US10Y as a DXY leading indicator."
        ),
        "weight": 1.2,
    },
    {
        "id": "macro_trade",
        "name": "Trade Balance Analyst",
        "archetype": "fundamental",
        "system": (
            "You analyze India's trade balance and current account for INR implications. "
            "High oil prices + weak gold prices = deteriorating trade balance = INR pressure. "
            "You also watch corporate flows — strong India equity market sentiment "
            "(positive IN_Avg_Tone) suggests FII inflows which support INR."
        ),
        "weight": 0.9,
    },
    {
        "id": "macro_em",
        "name": "EM Specialist",
        "archetype": "fundamental",
        "system": (
            "You are an emerging markets specialist. You view INR through the EM lens. "
            "Global risk-off (rising US10Y, rising DXY, falling gold) = EM sell-off = INR weakness. "
            "Risk-on = EM rally = INR strength. India has relatively strong fundamentals "
            "vs other EMs, so India-specific sentiment (IN vs US tone difference) matters."
        ),
        "weight": 1.0,
    },

    # ── Carry Traders (4) ────────────────────────────────────────────────────
    {
        "id": "carry_classic",
        "name": "Classic Carry Trader",
        "archetype": "carry",
        "system": (
            "You run classic carry trades: borrow in low-yield currencies, invest in high-yield. "
            "India typically offers higher rates than the US in normal times. "
            "Carry trade is profitable when volatility is low (vol_ratio < 0.8). "
            "When vol spikes, carry unwinds fast — you exit if realized_vol rises sharply."
        ),
        "weight": 1.0,
    },
    {
        "id": "carry_risk",
        "name": "Carry Risk Manager",
        "archetype": "carry",
        "system": (
            "You manage carry trade risk. Your signal: when US10Y rises faster than India's "
            "implied rates, carry attractiveness falls and capital flows back to USD. "
            "You also watch IN_Panic_Index — above 0.5 means carry is unwinding. "
            "Stability (Diff_Stability > 0) signals carry is safe."
        ),
        "weight": 0.9,
    },
    {
        "id": "carry_vol_arb",
        "name": "Vol-Carry Arbitrageur",
        "archetype": "carry",
        "system": (
            "You exploit the interaction between volatility and carry. "
            "Low vol (realized_vol < 0.003) + positive carry = high conviction carry long INR. "
            "Rising vol = reduce carry exposure. You are the first to exit when crisis signals appear: "
            "IN_Panic_Index > 0.6 or tone < -2."
        ),
        "weight": 0.8,
    },
    {
        "id": "carry_momentum",
        "name": "Carry-Momentum Hybrid",
        "archetype": "carry",
        "system": (
            "You combine carry signals with momentum. You only take carry trades when "
            "momentum confirms: positive carry + positive MA_momentum = strong long INR. "
            "You exit carry trades when momentum turns negative regardless of yield differential. "
            "This filters out false carry setups during trend reversals."
        ),
        "weight": 1.0,
    },

    # ── Sentiment / News Traders (5) ─────────────────────────────────────────
    {
        "id": "sent_india",
        "name": "India Sentiment Trader",
        "archetype": "sentiment",
        "system": (
            "You trade INR based purely on India-specific news sentiment from GDELT. "
            "Positive IN_Avg_Tone (> +1) = buy INR. Negative (< -1) = sell INR. "
            "High IN_Total_Mentions = high attention = higher impact of tone signal. "
            "IN_Panic_Index > 0.5 = extreme negative = strong sell signal for INR."
        ),
        "weight": 1.0,
    },
    {
        "id": "sent_global",
        "name": "Global Risk Sentiment Trader",
        "archetype": "sentiment",
        "system": (
            "You trade risk-on/risk-off based on US news sentiment. "
            "Negative US_Avg_Tone = global risk-off = USD safe-haven demand = USD/INR up. "
            "Positive US sentiment = risk-on = EM flows = INR strengthens. "
            "Diff_Tone (India tone - US tone) is your relative sentiment gauge."
        ),
        "weight": 1.0,
    },
    {
        "id": "sent_panic",
        "name": "Panic Index Specialist",
        "archetype": "sentiment",
        "system": (
            "You specialize in identifying panic episodes. IN_Panic_Index > 0.7 signals "
            "extreme stress — historically USD/INR spikes in these periods. "
            "After panic peaks, you look for mean reversion: if panic_index is falling "
            "from >0.7, INR typically recovers. You trade the panic spike and recovery."
        ),
        "weight": 1.1,
    },
    {
        "id": "sent_stability",
        "name": "Stability Analyst",
        "archetype": "sentiment",
        "system": (
            "You trade based on political and economic stability signals in GDELT. "
            "IN_Avg_Stability > 0 = India political stability = supports INR. "
            "Diff_Stability (India - USA stability) > 0 = India outperforming = INR positive. "
            "Sudden drops in stability (Diff_Stability < -0.5) signal risk events."
        ),
        "weight": 0.9,
    },
    {
        "id": "sent_contrarian",
        "name": "Sentiment Contrarian",
        "archetype": "sentiment",
        "system": (
            "You are a sentiment contrarian. When India news tone is extremely negative "
            "(IN_Avg_Tone < -3) and panic is peaking, you buy INR — maximum bearishness = "
            "market bottom. When tone is extremely positive (> +3), you sell. "
            "Sentiment extremes are mean-reverting faster than prices."
        ),
        "weight": 0.8,
    },

    # ── Institutional / Flow Traders (5) ──────────────────────────────────────
    {
        "id": "flow_fii",
        "name": "FII Flow Modeler",
        "archetype": "flow",
        "system": (
            "You model Foreign Institutional Investor (FII) flows into India. "
            "FIIs buy Indian equities when: DXY is stable/falling, India tone is positive, "
            "US10Y is not rising aggressively. FII buying = INR demand = appreciation. "
            "FII selling (risk-off, rising US yields) = INR depreciation pressure."
        ),
        "weight": 1.1,
    },
    {
        "id": "flow_corporate",
        "name": "Corporate Hedger",
        "archetype": "flow",
        "system": (
            "You model corporate USD/INR hedging flows. Indian exporters (IT, pharma) "
            "sell USD when USD/INR is high → natural selling pressure at highs. "
            "Importers (oil companies, manufacturers) buy USD at lows. "
            "These flows create range-bound behavior and mean reversion at extremes."
        ),
        "weight": 0.9,
    },
    {
        "id": "flow_rbi_intervention",
        "name": "RBI Intervention Modeler",
        "archetype": "flow",
        "system": (
            "You model RBI's FX intervention behavior. RBI intervenes to sell USD "
            "when INR depreciates rapidly (large positive MA_momentum). "
            "It buys USD to accumulate reserves when INR strengthens too much. "
            "Panic index > 0.6 = high probability of RBI defensive intervention = cap on weakness."
        ),
        "weight": 1.2,
    },
    {
        "id": "flow_gold",
        "name": "Gold-INR Analyst",
        "archetype": "flow",
        "system": (
            "You analyze the gold-INR relationship. India is a major gold importer. "
            "Rising gold prices = increased import bill = mild INR pressure. "
            "But gold also rises in risk-off (which also hits INR). "
            "GOLD_change > 1% in a risk-off context amplifies INR weakness."
        ),
        "weight": 0.8,
    },
    {
        "id": "flow_offshore",
        "name": "NDF Market Analyst",
        "archetype": "flow",
        "system": (
            "You trade based on offshore Non-Deliverable Forward (NDF) market dynamics. "
            "High volatility (vol_ratio > 1.5) with negative India tone = offshore selling pressure. "
            "Stability and positive tone = NDF markets calm. "
            "When onshore and offshore rates diverge (implied by panic index), "
            "RBI intervention risk rises sharply."
        ),
        "weight": 1.0,
    },

    # ── Quantitative / Statistical Traders (4) ────────────────────────────────
    {
        "id": "quant_garch",
        "name": "GARCH Volatility Quant",
        "archetype": "quant",
        "system": (
            "You are a volatility quant. Your model: high vol_ratio (>1.5) means "
            "recent volatility is elevated vs history — expect continuation or mean reversion. "
            "You adjust position sizes inversely to volatility. "
            "In HIGH_VOLATILITY regimes you predict larger moves but with lower directional conviction."
        ),
        "weight": 1.0,
    },
    {
        "id": "quant_regime",
        "name": "Regime-Switching Quant",
        "archetype": "quant",
        "system": (
            "You apply a regime-switching model to USD/INR. "
            "You identify the current regime: calm (low vol, low momentum), "
            "trending (strong momentum, moderate vol), or crisis (high vol, extreme sentiment). "
            "Each regime has different alpha sources: carry in calm, momentum in trending, "
            "mean-reversion in crisis."
        ),
        "weight": 1.1,
    },
    {
        "id": "quant_zscore",
        "name": "Statistical Arbitrageur",
        "archetype": "quant",
        "system": (
            "You trade USD/INR based on statistical signals. "
            "INR z-score > +1.5 (INR overvalued vs 20-day MA) → sell USD/buy INR. "
            "Z-score < -1.5 → buy USD. "
            "You also check: if RSI and z-score both signal the same direction, "
            "double your conviction. Conflicting signals → stay neutral."
        ),
        "weight": 1.0,
    },
    {
        "id": "quant_ml",
        "name": "ML Signal Aggregator",
        "archetype": "quant",
        "system": (
            "You represent a black-box ML model's output. You weight all signals equally: "
            "technical (MA, RSI, zscore), macro (DXY, US10Y, OIL), and sentiment (tone, panic). "
            "Count bullish vs bearish signals across all dimensions. "
            "If >60% of signals are bullish USD → predict depreciation. "
            "If >60% bearish USD → predict appreciation."
        ),
        "weight": 1.0,
    },
]

assert len(AGENT_PERSONAS) == 30, f"Expected 30 agents, got {len(AGENT_PERSONAS)}"


# ─── Response Schema ──────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    agent_id: str
    archetype: str
    direction: str          # "appreciation", "depreciation", "neutral"
    magnitude: float        # 0.0 to 1.0 (normalized conviction)
    confidence: float       # 1 to 10
    reasoning: str
    weight: float
    success: bool = True
    error: str = ""


# ─── Prompt Builder ───────────────────────────────────────────────────────────

AGENT_TASK_PROMPT = """You are analyzing USD/INR for the next trading day.

Your role: {role}

{context}

Based on your specific expertise and the data above, predict the next-day USD/INR direction.

Respond with ONLY this JSON (no markdown, no explanation):
{{
    "direction": "<appreciation | depreciation | neutral>",
    "magnitude": <0.0 to 1.0, where 1.0 = very large expected move>,
    "confidence": <1 to 10>,
    "reasoning": "<one sentence from your specific perspective>"
}}

- "appreciation" = INR strengthens (USD/INR goes DOWN)
- "depreciation" = INR weakens (USD/INR goes UP)
- "neutral" = no clear directional edge
"""


def _build_agent_prompt(persona: dict, context_packet: dict) -> str:
    context_text = _format_context(context_packet)
    return AGENT_TASK_PROMPT.format(
        role=persona["system"],
        context=context_text,
    )


# ─── Ollama Client ────────────────────────────────────────────────────────────

def _check_ollama_available(base_url: str = config.OLLAMA_BASE_URL) -> bool:
    """Check if Ollama is running and the model is available."""
    if not REQUESTS_AVAILABLE:
        return False
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        # Check if our model (or a variant) is pulled
        model_base = config.OLLAMA_MODEL.split(":")[0]
        return any(model_base in m for m in models)
    except Exception:
        return False


def _call_ollama(
    prompt: str,
    model: str = config.OLLAMA_MODEL,
    temperature: float = config.OLLAMA_TEMPERATURE,
    timeout: int = config.OLLAMA_TIMEOUT,
) -> Optional[str]:
    """Single blocking call to Ollama /api/chat endpoint."""
    if not REQUESTS_AVAILABLE:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 256,
        },
    }
    try:
        r = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        return None


def _parse_agent_response(text: Optional[str]) -> Optional[dict]:
    """Extract JSON from agent response."""
    if not text:
        return None
    # Try markdown block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    # Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _normalize_direction(raw: str) -> str:
    """Normalize direction string to canonical form."""
    raw = str(raw).lower().strip()
    if any(w in raw for w in ["appreciat", "strengthen", "lower", "down", "bullish inr"]):
        return "appreciation"
    if any(w in raw for w in ["depreciat", "weaken", "higher", "up", "bullish usd"]):
        return "depreciation"
    return "neutral"


def _run_single_agent(persona: dict, context_packet: dict) -> AgentResponse:
    """Run one agent and return its structured response."""
    prompt = _build_agent_prompt(persona, context_packet)
    raw = _call_ollama(prompt)
    parsed = _parse_agent_response(raw)

    if parsed is None:
        return AgentResponse(
            agent_id=persona["id"],
            archetype=persona["archetype"],
            direction="neutral",
            magnitude=0.0,
            confidence=1.0,
            reasoning="parse_failed",
            weight=persona["weight"],
            success=False,
            error="Failed to parse response",
        )

    direction = _normalize_direction(parsed.get("direction", "neutral"))
    magnitude = float(np.clip(parsed.get("magnitude", 0.5), 0.0, 1.0))
    confidence = float(np.clip(parsed.get("confidence", 5), 1, 10))
    reasoning = str(parsed.get("reasoning", ""))[:200]

    return AgentResponse(
        agent_id=persona["id"],
        archetype=persona["archetype"],
        direction=direction,
        magnitude=magnitude,
        confidence=confidence,
        reasoning=reasoning,
        weight=persona["weight"],
        success=True,
    )


# ─── Market Simulation Engine ─────────────────────────────────────────────────

class MarketSimulation:
    """Runs 30 agents concurrently and aggregates into a market consensus signal."""

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        num_workers: int = config.OLLAMA_MAX_WORKERS,
    ):
        self.model = model
        self.num_workers = num_workers
        self.available = _check_ollama_available()
        if not self.available:
            print(
                f"[MarketSim] Ollama not available or model '{model}' not pulled. "
                f"Run: ollama pull {model}"
            )

    def run(
        self,
        context_packet: dict,
        personas: List[dict] = AGENT_PERSONAS,
        timeout_per_agent: int = config.OLLAMA_TIMEOUT,
    ) -> Dict:
        """Run all agents concurrently and return aggregated results.

        Returns:
            {
                "consensus_direction": str,
                "consensus_strength": float,   # |appreciation% - depreciation%|
                "weighted_magnitude": float,    # Trimmed weighted avg magnitude
                "entropy": float,              # 0=consensus 1=maximum disagreement
                "avg_confidence": float,
                "archetype_votes": dict,       # per-archetype consensus
                "agent_responses": list,       # raw (for caching/analysis)
                "n_success": int,
                "n_total": int,
            }
        """
        if not self.available:
            return self._null_result(len(personas))

        responses: List[AgentResponse] = []

        # Ollama processes one request at a time (single GPU), regardless of how many
        # concurrent HTTP connections we open. Total wall-clock budget = n_agents * per_agent_time.
        # The ThreadPoolExecutor just avoids blocking the main thread while waiting.
        total_timeout = timeout_per_agent * len(personas) + 30  # sequential worst case + buffer

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_id = {
                executor.submit(_run_single_agent, p, context_packet): p
                for p in personas
            }

            done, pending = wait(future_to_id, timeout=total_timeout)

            # Collect completed futures
            for future in done:
                persona = future_to_id[future]
                try:
                    resp = future.result()
                    responses.append(resp)
                except Exception as e:
                    responses.append(AgentResponse(
                        agent_id=persona["id"], archetype=persona.get("archetype", "unknown"),
                        direction="neutral", magnitude=0.0, confidence=1.0,
                        reasoning="error", weight=persona.get("weight", 1.0),
                        success=False, error=str(e),
                    ))

            # Mark timed-out futures as failed (don't block on them)
            for future in pending:
                future.cancel()
                persona = future_to_id[future]
                responses.append(AgentResponse(
                    agent_id=persona["id"], archetype=persona.get("archetype", "unknown"),
                    direction="neutral", magnitude=0.0, confidence=1.0,
                    reasoning="timeout", weight=persona.get("weight", 1.0),
                    success=False, error="timeout",
                ))

        n_done = len(done)
        n_timedout = len(pending)
        if n_timedout > 0:
            print(f"[MarketSim] {n_timedout}/{len(personas)} agents timed out after {total_timeout}s "
                  f"({n_done} completed)", flush=True)
        if n_done < len(personas) // 2:
            print(f"[MarketSim] Warning: only {n_done}/{len(personas)} agents responded. "
                  f"Consider: faster model (phi3), --quick-agents flag, or increase OLLAMA_TIMEOUT.",
                  flush=True)

        return self._aggregate(responses, len(personas))

    # ─── Aggregation ──────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(responses: List[AgentResponse], n_total: int) -> Dict:
        valid = [r for r in responses if r.success]
        n_success = len(valid)

        if n_success == 0:
            return MarketSimulation._null_result(n_total)

        # Weight by persona weight × confidence
        weights = np.array([r.weight * (r.confidence / 10.0) for r in valid])
        total_w = weights.sum()
        if total_w == 0:
            weights = np.ones(n_success)
            total_w = n_success
        weights_norm = weights / total_w

        # Direction breakdown
        directions = np.array([r.direction for r in valid])
        appreciation_w = weights_norm[directions == "appreciation"].sum()
        depreciation_w = weights_norm[directions == "depreciation"].sum()
        neutral_w = weights_norm[directions == "neutral"].sum()

        consensus_direction = (
            "appreciation" if appreciation_w > depreciation_w
            else ("depreciation" if depreciation_w > appreciation_w else "neutral")
        )
        consensus_strength = float(abs(appreciation_w - depreciation_w))

        # Trimmed weighted magnitude
        magnitudes = np.array([r.magnitude for r in valid])
        weighted_magnitude = float(
            MarketSimulation._trimmed_weighted_mean(
                magnitudes, weights_norm, config.AGENT_TRIM_FRACTION
            )
        )

        # Entropy (measure of disagreement)
        probs = np.array([appreciation_w, depreciation_w, neutral_w])
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)) / np.log(3))  # normalized 0-1

        # Avg confidence
        avg_confidence = float(np.average([r.confidence for r in valid], weights=weights))

        # Per-archetype breakdown
        archetypes = {}
        for arch in set(r.archetype for r in valid):
            arch_resp = [r for r in valid if r.archetype == arch]
            arch_dirs = [r.direction for r in arch_resp]
            most_common = max(set(arch_dirs), key=arch_dirs.count)
            archetypes[arch] = {
                "n": len(arch_resp),
                "consensus": most_common,
                "appreciation_pct": sum(1 for d in arch_dirs if d == "appreciation") / len(arch_dirs),
                "depreciation_pct": sum(1 for d in arch_dirs if d == "depreciation") / len(arch_dirs),
            }

        return {
            "consensus_direction": consensus_direction,
            "consensus_strength": round(consensus_strength, 4),
            "weighted_magnitude": round(weighted_magnitude, 4),
            "entropy": round(entropy, 4),
            "avg_confidence": round(avg_confidence, 2),
            "appreciation_weight": round(float(appreciation_w), 4),
            "depreciation_weight": round(float(depreciation_w), 4),
            "neutral_weight": round(float(neutral_w), 4),
            "archetype_votes": archetypes,
            "agent_responses": [
                {
                    "id": r.agent_id,
                    "archetype": r.archetype,
                    "direction": r.direction,
                    "magnitude": r.magnitude,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for r in responses
            ],
            "n_success": n_success,
            "n_total": n_total,
        }

    @staticmethod
    def _trimmed_weighted_mean(
        values: np.ndarray, weights: np.ndarray, trim: float
    ) -> float:
        """Weighted mean after removing top/bottom `trim` fraction by value."""
        if len(values) == 0:
            return 0.0
        n_trim = max(1, int(len(values) * trim))
        sorted_idx = np.argsort(values)
        keep = sorted_idx[n_trim:-n_trim] if len(values) > 2 * n_trim else sorted_idx
        if len(keep) == 0:
            return float(np.mean(values))
        w = weights[keep]
        w_sum = w.sum()
        if w_sum == 0:
            return float(np.mean(values[keep]))
        return float(np.dot(values[keep], w) / w_sum)

    @staticmethod
    def _null_result(n_total: int) -> Dict:
        return {
            "consensus_direction": "neutral",
            "consensus_strength": 0.0,
            "weighted_magnitude": 0.0,
            "entropy": 1.0,
            "avg_confidence": 0.0,
            "appreciation_weight": 0.0,
            "depreciation_weight": 0.0,
            "neutral_weight": 1.0,
            "archetype_votes": {},
            "agent_responses": [],
            "n_success": 0,
            "n_total": n_total,
        }


# ─── Feature Encoding ─────────────────────────────────────────────────────────

def encode_simulation_features(result: Dict) -> Dict[str, float]:
    """Encode market simulation output to numeric features for the stat engine."""
    direction_sign = (
        -1.0 if result["consensus_direction"] == "appreciation"  # INR up = USD/INR down
        else (1.0 if result["consensus_direction"] == "depreciation" else 0.0)
    )

    features = {
        # Direction signal: signed by INR impact, scaled by strength
        "sim_direction_signal": direction_sign * result["consensus_strength"],
        # Raw direction weights
        "sim_appreciation_w": result["appreciation_weight"],
        "sim_depreciation_w": result["depreciation_weight"],
        "sim_neutral_w": result["neutral_weight"],
        # Magnitude and conviction
        "sim_magnitude": result["weighted_magnitude"],
        "sim_consensus_strength": result["consensus_strength"],
        # Uncertainty (high entropy = agents disagree = low signal value)
        "sim_entropy": result["entropy"],
        "sim_uncertainty": 1.0 - result["consensus_strength"],
        # Confidence
        "sim_avg_confidence": result["avg_confidence"] / 10.0,
        # Success rate (low = model issues)
        "sim_success_rate": result["n_success"] / max(result["n_total"], 1),
    }

    # Per-archetype signals
    for arch, info in result.get("archetype_votes", {}).items():
        arch_sign = (
            -1.0 if info["consensus"] == "appreciation"
            else (1.0 if info["consensus"] == "depreciation" else 0.0)
        )
        features[f"sim_{arch}_signal"] = arch_sign * (
            info["appreciation_pct"] if arch_sign < 0 else info["depreciation_pct"]
        )

    return features


# List of all simulation feature names (for config.py integration)
SIMULATION_FEATURE_NAMES = [
    "sim_direction_signal", "sim_appreciation_w", "sim_depreciation_w",
    "sim_neutral_w", "sim_magnitude", "sim_consensus_strength",
    "sim_entropy", "sim_uncertainty", "sim_avg_confidence", "sim_success_rate",
    # archetype signals (added dynamically when archetypes are present)
    "sim_technical_signal", "sim_fundamental_signal", "sim_carry_signal",
    "sim_sentiment_signal", "sim_flow_signal", "sim_quant_signal",
]

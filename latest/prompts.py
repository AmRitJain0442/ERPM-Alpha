"""
Prompt templates for the 4 LLM analyst tasks.
Each task gets the same context packet and produces structured JSON.
"""

import json


def _format_context(packet: dict) -> str:
    """Format a context packet into a readable text block for the LLM."""
    lines = []

    lines.append("=== USD/INR MARKET CONTEXT ===")
    lines.append(f"Date: {packet.get('date', 'N/A')}")

    ph = packet.get("price_history", {})
    if ph:
        lines.append(f"\n--- Price ---")
        lines.append(f"Current USD/INR: {ph.get('current_inr', 'N/A')}")
        lines.append(f"Previous close: {ph.get('prev_inr', 'N/A')}")
        lines.append(f"20-day range: {ph.get('inr_20d_low', 'N/A')} - {ph.get('inr_20d_high', 'N/A')}")
        lines.append(f"20-day mean: {ph.get('inr_20d_mean', 'N/A')}")
        lines.append(f"5-day trend offset: {ph.get('inr_5d_trend', 'N/A')}")

    tech = packet.get("technicals", {})
    if tech:
        lines.append(f"\n--- Technicals ---")
        for k, v in tech.items():
            lines.append(f"{k}: {v}")

    macro = packet.get("macro", {})
    if macro:
        lines.append(f"\n--- Macro ---")
        for k, v in macro.items():
            lines.append(f"{k}: {v}")

    sent = packet.get("sentiment", {})
    if sent:
        lines.append(f"\n--- GDELT Sentiment ---")
        for k, v in sent.items():
            lines.append(f"{k}: {v}")

    sr = packet.get("stat_regime", "")
    if sr:
        lines.append(f"\nStatistical regime estimate: {sr}")

    headlines = packet.get("headlines", [])
    if headlines:
        lines.append(f"\n--- Recent Headlines ({len(headlines)}) ---")
        for i, h in enumerate(headlines[:15], 1):
            tone_tag = "positive" if h["tone"] > 1 else ("negative" if h["tone"] < -1 else "neutral")
            lines.append(f"{i}. [{tone_tag}] {h['headline']} (tone={h['tone']:.1f}, mentions={h['mentions']})")

    return "\n".join(lines)


# ─── Task A: Regime Classifier ───────────────────────────────────────────────

REGIME_CLASSIFIER_PROMPT = """You are a forex market regime analyst specializing in the USD/INR pair.

Given the market context below, classify the CURRENT market regime into exactly one of these categories:

1. CALM_CARRY — Low volatility, range-bound, carry trade conditions dominate
2. TRENDING_APPRECIATION — INR is strengthening (USD/INR falling), sustained trend
3. TRENDING_DEPRECIATION — INR is weakening (USD/INR rising), sustained trend
4. HIGH_VOLATILITY — Elevated uncertainty, no clear direction, wide price swings
5. CRISIS_STRESS — Extreme stress, potential tail events, panic selling/buying

Consider ALL available data: price action, technicals, macro conditions, GDELT sentiment, and news.

{context}

Respond with ONLY this JSON (no markdown, no explanation):
{{
    "regime": "<one of: CALM_CARRY, TRENDING_APPRECIATION, TRENDING_DEPRECIATION, HIGH_VOLATILITY, CRISIS_STRESS>",
    "confidence": <0.0 to 1.0>,
    "reasoning": "<one sentence explaining your classification>",
    "secondary_regime": "<second most likely regime or null>",
    "secondary_confidence": <0.0 to 1.0 or 0>
}}"""


# ─── Task B: Event Impact Scorer ─────────────────────────────────────────────

EVENT_IMPACT_PROMPT = """You are a geopolitical event analyst assessing how current events affect the USD/INR exchange rate.

Score each significant event/factor in the current context on a scale from -5 to +5:
- Negative scores: event pushes USD/INR LOWER (INR appreciation / USD weakening)
- Positive scores: event pushes USD/INR HIGHER (INR depreciation / USD strengthening)
- Zero: no meaningful impact

Consider: RBI policy, Fed actions, oil prices, capital flows, trade data, geopolitical tensions, fiscal policy, global risk appetite.

{context}

Respond with ONLY this JSON (no markdown, no explanation):
{{
    "events": [
        {{
            "event": "<brief description>",
            "impact_score": <-5 to +5>,
            "confidence": <0.0 to 1.0>,
            "channel": "<how it affects INR: e.g., capital_flows, trade_balance, risk_sentiment, monetary_policy, oil_channel>"
        }}
    ],
    "net_direction": "<appreciation or depreciation or neutral>",
    "net_magnitude": <0.0 to 5.0>
}}"""


# ─── Task C: Causal Chain Extractor ──────────────────────────────────────────

CAUSAL_CHAIN_PROMPT = """You are a macroeconomic analyst mapping causal chains that affect the USD/INR rate.

Identify cause-effect chains currently active in the market. Each chain should trace from a root cause through intermediate effects to the final USD/INR impact.

Example chain: "Fed rate hike → USD strengthens → DXY rises → capital outflows from EM → INR depreciates"

{context}

Respond with ONLY this JSON (no markdown, no explanation):
{{
    "chains": [
        {{
            "chain": ["<cause>", "<intermediate1>", "...", "<INR effect>"],
            "strength": <0.0 to 1.0>,
            "direction": "<appreciation or depreciation>",
            "timeframe": "<immediate, short_term, medium_term>"
        }}
    ],
    "dominant_chain_index": <0-based index of strongest chain>
}}"""


# ─── Task D: Risk Signal Detector ────────────────────────────────────────────

RISK_SIGNAL_PROMPT = """You are a risk analyst monitoring threats and opportunities for the USD/INR rate.

Identify active risk signals — events, conditions, or emerging patterns that could cause significant USD/INR moves. Focus on:
- Tail risks (low probability, high impact)
- Emerging risks (building but not yet priced in)
- Risk reversals (existing risks that are fading)

{context}

Respond with ONLY this JSON (no markdown, no explanation):
{{
    "signals": [
        {{
            "signal": "<brief description>",
            "severity": <1 to 5, where 5 is extreme>,
            "probability": <0.0 to 1.0>,
            "direction": "<appreciation or depreciation or either>",
            "type": "<tail_risk, emerging, fading, priced_in>"
        }}
    ],
    "overall_risk_level": "<low, moderate, elevated, high, extreme>",
    "risk_bias": "<appreciation or depreciation or balanced>"
}}"""


def build_prompt(task: str, context_packet: dict) -> str:
    """Build the full prompt for a given task and context packet."""
    context_text = _format_context(context_packet)

    templates = {
        "regime_classifier": REGIME_CLASSIFIER_PROMPT,
        "event_impact": EVENT_IMPACT_PROMPT,
        "causal_chain": CAUSAL_CHAIN_PROMPT,
        "risk_signal": RISK_SIGNAL_PROMPT,
    }

    template = templates.get(task)
    if template is None:
        raise ValueError(f"Unknown task: {task}. Must be one of {list(templates.keys())}")

    return template.format(context=context_text)

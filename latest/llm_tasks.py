"""
LLMAnalyst: calls Gemini for each of the 4 analyst tasks,
parses JSON responses, encodes to numeric features, uses cache.
"""

import json
import re
import time
from typing import Dict, List, Optional

import numpy as np

# Try new google.genai SDK first, fall back to deprecated google.generativeai
_genai_new = False
try:
    from google import genai as _genai_module
    from google.genai import types as _genai_types
    _genai_new = True
except ImportError:
    _genai_module = None
    _genai_types = None

if not _genai_new:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as _genai_old
    except ImportError:
        _genai_old = None
else:
    _genai_old = None

import config
from cache import LLMCache
from prompts import build_prompt


def initialize_gemini(api_key: str = config.GEMINI_API_KEY):
    """Initialize Gemini client/model. Returns (client_or_model, bool_success)."""
    if not api_key:
        print("[LLM] GEMINI_API_KEY not set")
        return None, False

    # New google.genai SDK (preferred)
    if _genai_new and _genai_module is not None:
        try:
            client = _genai_module.Client(api_key=api_key)
            return client, True
        except Exception as e:
            print(f"[LLM] google.genai init failed: {e}")
            return None, False

    # Legacy google.generativeai SDK
    if _genai_old is not None:
        try:
            _genai_old.configure(api_key=api_key)
            model = _genai_old.GenerativeModel(
                model_name=config.GEMINI_MODEL,
                generation_config={
                    "temperature": config.GEMINI_TEMPERATURE,
                    "top_p": config.GEMINI_TOP_P,
                    "max_output_tokens": config.GEMINI_MAX_TOKENS,
                },
            )
            return model, True
        except Exception as e:
            print(f"[LLM] google.generativeai init failed: {e}")
            return None, False

    print("[LLM] Neither google.genai nor google.generativeai is installed")
    return None, False


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    # Try markdown block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try finding JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _call_gemini(client_or_model, prompt: str, retries: int = config.MAX_RETRIES) -> Optional[str]:
    """Call Gemini with retry logic. Handles both new google.genai and legacy SDK."""
    for attempt in range(retries):
        try:
            # New google.genai SDK: client.models.generate_content(...)
            if _genai_new and hasattr(client_or_model, "models"):
                response = client_or_model.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=_genai_types.GenerateContentConfig(
                        temperature=config.GEMINI_TEMPERATURE,
                        top_p=config.GEMINI_TOP_P,
                        max_output_tokens=config.GEMINI_MAX_TOKENS,
                    ) if _genai_types else None,
                )
                if hasattr(response, "text"):
                    return response.text
                # Iterate parts
                for part in getattr(response, "candidates", [{}])[0:1]:
                    content = getattr(part, "content", None)
                    if content:
                        for p in getattr(content, "parts", []):
                            if hasattr(p, "text"):
                                return p.text
                return None

            # Legacy google.generativeai SDK: model.generate_content(...)
            response = client_or_model.generate_content(prompt)
            if hasattr(response, "text"):
                return response.text
            for candidate in getattr(response, "candidates", []):
                for part in getattr(candidate.content, "parts", []):
                    if hasattr(part, "text") and getattr(part, "role", None) != "thought":
                        return part.text
            return None

        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "quota" in err or "429" in err:
                wait = config.API_DELAY_SECONDS * (3 ** attempt)
                print(f"[LLM] Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif attempt < retries - 1:
                wait = config.INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"[LLM] Error: {e}, retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"[LLM] Failed after {retries} attempts: {e}")
                return None

    return None


class LLMAnalyst:
    """Runs all 4 LLM analyst tasks and encodes results to numeric features."""

    TASKS = ["regime_classifier", "event_impact", "causal_chain", "risk_signal"]

    def __init__(self, api_key: str = config.GEMINI_API_KEY):
        self.model, self.available = initialize_gemini(api_key)
        self.cache = LLMCache()

    def run_task(self, task: str, context_packet: dict) -> Optional[dict]:
        """Run a single LLM task. Returns parsed JSON or None."""
        if not self.available:
            return None

        prompt = build_prompt(task, context_packet)
        dt = context_packet.get("date", "unknown")

        # Check cache
        cached = self.cache.get(task, dt, prompt)
        if cached is not None:
            return cached

        # Call Gemini
        time.sleep(config.API_DELAY_SECONDS)  # Rate limit spacing
        raw = _call_gemini(self.model, prompt)
        if raw is None:
            return None

        parsed = _extract_json(raw)
        if parsed is None:
            print(f"[LLM] Could not parse JSON for task={task}, date={dt}")
            return None

        # Cache result
        self.cache.put(task, dt, prompt, parsed)
        return parsed

    def run_all_tasks(self, context_packet: dict) -> Dict[str, Optional[dict]]:
        """Run all 4 tasks and return results dict."""
        results = {}
        for task in self.TASKS:
            results[task] = self.run_task(task, context_packet)
        return results

    def encode_features(self, results: Dict[str, Optional[dict]]) -> Dict[str, float]:
        """Encode all LLM task results into numeric features."""
        features = {}
        features.update(self._encode_regime(results.get("regime_classifier")))
        features.update(self._encode_events(results.get("event_impact")))
        features.update(self._encode_chains(results.get("causal_chain")))
        features.update(self._encode_risk(results.get("risk_signal")))
        return features

    # ─── Encoders ─────────────────────────────────────────────────────────

    @staticmethod
    def _encode_regime(result: Optional[dict]) -> dict:
        """One-hot encode regime + confidence."""
        features = {f"regime_{r}": 0.0 for r in config.REGIMES}
        features["regime_confidence"] = 0.5  # neutral default

        if result is None:
            features[f"regime_{config.DEFAULT_REGIME}"] = 1.0
            return features

        regime = result.get("regime", config.DEFAULT_REGIME)
        if regime in config.REGIMES:
            features[f"regime_{regime}"] = 1.0
        else:
            features[f"regime_{config.DEFAULT_REGIME}"] = 1.0

        features["regime_confidence"] = float(result.get("confidence", 0.5))
        return features

    @staticmethod
    def _encode_events(result: Optional[dict]) -> dict:
        """Encode event impact scores to aggregated features."""
        defaults = {
            "event_impact_mean": 0.0,
            "event_impact_max": 0.0,
            "event_impact_min": 0.0,
            "event_count_positive": 0.0,
            "event_count_negative": 0.0,
        }
        if result is None:
            return defaults

        events = result.get("events", [])
        if not events:
            return defaults

        scores = []
        for e in events:
            score = e.get("impact_score", 0)
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue

        if not scores:
            return defaults

        return {
            "event_impact_mean": np.mean(scores),
            "event_impact_max": max(scores),
            "event_impact_min": min(scores),
            "event_count_positive": sum(1 for s in scores if s > 0),
            "event_count_negative": sum(1 for s in scores if s < 0),
        }

    @staticmethod
    def _encode_chains(result: Optional[dict]) -> dict:
        """Encode causal chains to count/strength features."""
        defaults = {
            "chain_count": 0.0,
            "chain_avg_strength": 0.0,
            "chain_max_strength": 0.0,
        }
        if result is None:
            return defaults

        chains = result.get("chains", [])
        if not chains:
            return defaults

        strengths = []
        for c in chains:
            s = c.get("strength", 0)
            try:
                strengths.append(float(s))
            except (TypeError, ValueError):
                continue

        if not strengths:
            return defaults

        return {
            "chain_count": float(len(strengths)),
            "chain_avg_strength": np.mean(strengths),
            "chain_max_strength": max(strengths),
        }

    @staticmethod
    def _encode_risk(result: Optional[dict]) -> dict:
        """Encode risk signals to flag/severity features."""
        defaults = {
            "risk_flag_count": 0.0,
            "risk_max_severity": 0.0,
            "risk_avg_severity": 0.0,
        }
        if result is None:
            return defaults

        signals = result.get("signals", [])
        if not signals:
            return defaults

        severities = []
        for s in signals:
            sev = s.get("severity", 0)
            try:
                severities.append(float(sev))
            except (TypeError, ValueError):
                continue

        if not severities:
            return defaults

        return {
            "risk_flag_count": float(len(severities)),
            "risk_max_severity": max(severities),
            "risk_avg_severity": np.mean(severities),
        }

    def get_regime_label(self, results: Dict[str, Optional[dict]]) -> str:
        """Extract the regime label from LLM results, with fallback."""
        rc = results.get("regime_classifier")
        if rc and rc.get("regime") in config.REGIMES:
            return rc["regime"]
        return config.DEFAULT_REGIME

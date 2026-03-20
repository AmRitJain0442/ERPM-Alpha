"""
MetaLearner: tracks per-regime LLM accuracy and dynamically adjusts
LLM feature weights. Starts at 5% weight, lets LLM prove itself.
"""

import json
import os
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

import config


class MetaLearner:
    """Tracks LLM feature value per regime and adapts weights."""

    def __init__(self, state_path: str = None):
        self.state_path = state_path or os.path.join(
            config.CACHE_DIR, "meta_learner_state.json"
        )
        # Per-regime tracking
        self.regime_errors_with_llm = defaultdict(list)    # regime → [errors]
        self.regime_errors_without_llm = defaultdict(list)  # baseline errors
        self.regime_weights = {r: config.INITIAL_LLM_WEIGHT for r in config.REGIMES}
        self.regime_sample_counts = defaultdict(int)

        # Per-feature importance tracking
        self.feature_contributions = defaultdict(lambda: defaultdict(list))

        self._load_state()

    def record(self, regime: str, error: float, llm_features: Dict[str, float]):
        """Record a prediction outcome for weight adjustment."""
        self.regime_errors_with_llm[regime].append(error)
        self.regime_sample_counts[regime] += 1

        # Track which LLM features were active
        for feat, val in llm_features.items():
            if val != 0.0:
                self.feature_contributions[regime][feat].append(abs(error))

        # Periodically update weights
        if self.regime_sample_counts[regime] % 20 == 0:
            self._update_weight(regime)

        # Periodically save state
        total = sum(self.regime_sample_counts.values())
        if total % 50 == 0:
            self._save_state()

    def record_baseline(self, regime: str, error: float):
        """Record a stat-only baseline error for comparison."""
        self.regime_errors_without_llm[regime].append(error)

    def get_regime_weight(self, regime: str) -> float:
        """Get the current LLM weight for a regime."""
        return self.regime_weights.get(regime, config.INITIAL_LLM_WEIGHT)

    def _update_weight(self, regime: str):
        """Adjust LLM weight based on whether it helps vs hurts."""
        with_llm = self.regime_errors_with_llm.get(regime, [])
        without_llm = self.regime_errors_without_llm.get(regime, [])

        if len(with_llm) < 20:
            return  # Not enough data

        current = self.regime_weights[regime]
        recent_mae = np.mean(np.abs(with_llm[-50:]))

        if len(without_llm) >= 20:
            baseline_mae = np.mean(np.abs(without_llm[-50:]))
            # LLM helps → increase weight; LLM hurts → decrease
            if recent_mae < baseline_mae * 0.95:
                # LLM clearly helping (>5% improvement)
                delta = config.LLM_WEIGHT_ADJUSTMENT_RATE * 2
            elif recent_mae < baseline_mae:
                # LLM marginally helping
                delta = config.LLM_WEIGHT_ADJUSTMENT_RATE
            elif recent_mae > baseline_mae * 1.05:
                # LLM clearly hurting
                delta = -config.LLM_WEIGHT_ADJUSTMENT_RATE * 2
            else:
                # No clear signal
                delta = 0
        else:
            # No baseline comparison yet — use trend
            if len(with_llm) >= 40:
                first_half = np.mean(np.abs(with_llm[-40:-20]))
                second_half = np.mean(np.abs(with_llm[-20:]))
                if second_half < first_half:
                    delta = config.LLM_WEIGHT_ADJUSTMENT_RATE
                else:
                    delta = -config.LLM_WEIGHT_ADJUSTMENT_RATE
            else:
                delta = 0

        new_weight = np.clip(
            current + delta, config.MIN_LLM_WEIGHT, config.MAX_LLM_WEIGHT
        )
        self.regime_weights[regime] = float(new_weight)

    def get_feature_importance(self, regime: str) -> Dict[str, float]:
        """Get per-feature importance for a regime.
        Lower error when feature is active = higher importance.
        """
        contributions = self.feature_contributions.get(regime, {})
        if not contributions:
            return {}

        overall_mae = np.mean(
            np.abs(self.regime_errors_with_llm.get(regime, [0]))
        )
        if overall_mae == 0:
            return {}

        importance = {}
        for feat, errors in contributions.items():
            if len(errors) >= 5:
                feat_mae = np.mean(errors)
                # Lower error = more important (inverted ratio)
                importance[feat] = max(0, 1 - feat_mae / overall_mae)

        return importance

    def summary(self) -> Dict:
        """Return summary of meta-learner state."""
        regime_info = {}
        for regime in config.REGIMES:
            errors = self.regime_errors_with_llm.get(regime, [])
            regime_info[regime] = {
                "samples": self.regime_sample_counts[regime],
                "weight": round(self.regime_weights[regime], 4),
                "mae": round(float(np.mean(np.abs(errors))), 6) if errors else None,
            }
        return {
            "total_samples": sum(self.regime_sample_counts.values()),
            "regimes": regime_info,
        }

    def _save_state(self):
        """Persist state to disk."""
        state = {
            "regime_weights": self.regime_weights,
            "regime_sample_counts": dict(self.regime_sample_counts),
            "regime_errors_with_llm": {
                k: v[-200:] for k, v in self.regime_errors_with_llm.items()
            },
            "regime_errors_without_llm": {
                k: v[-200:] for k, v in self.regime_errors_without_llm.items()
            },
        }
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)
        except IOError:
            pass

    def _load_state(self):
        """Load state from disk if available."""
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r") as f:
                state = json.load(f)
            self.regime_weights.update(state.get("regime_weights", {}))
            for k, v in state.get("regime_sample_counts", {}).items():
                self.regime_sample_counts[k] = v
            for k, v in state.get("regime_errors_with_llm", {}).items():
                self.regime_errors_with_llm[k] = v
            for k, v in state.get("regime_errors_without_llm", {}).items():
                self.regime_errors_without_llm[k] = v
        except (json.JSONDecodeError, IOError):
            pass

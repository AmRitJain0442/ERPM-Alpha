"""
Statistical and ML model suite plus blending logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

from .config import RunConfig


@dataclass
class ModelOutput:
    base_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    ensemble_return: float
    direction_probabilities: Dict[str, float]
    direction_score: float
    volatility_estimate: float

    def to_dict(self) -> Dict:
        return asdict(self)


class HybridModelSuite:
    def __init__(self, config: RunConfig):
        self.config = config
        self.feature_cols: List[str] = []
        self.regressors: Dict[str, object] = {}
        self.model_weights: Dict[str, float] = {}
        self.direction_model: Optional[Pipeline] = None
        self.direction_classes_: Optional[np.ndarray] = None
        self.egarch = None
        self.fitted = False

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> None:
        if len(train_df) < self.config.train_min_days:
            raise ValueError("Not enough training rows")

        self.feature_cols = list(feature_cols)
        split = max(self.config.train_min_days, len(train_df) - self.config.validation_days)
        fit_df = train_df.iloc[:split].copy()
        val_df = train_df.iloc[split:].copy() if split < len(train_df) else train_df.iloc[-self.config.validation_days:].copy()

        X_fit = fit_df[self.feature_cols].copy().ffill().bfill().fillna(0.0)
        y_fit = fit_df["target_return"].astype(float).values
        X_val = val_df[self.feature_cols].copy().ffill().bfill().fillna(0.0)
        y_val = val_df["target_return"].astype(float).values

        regressors: Dict[str, object] = {
            "ridge": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=250,
                max_depth=5,
                min_samples_leaf=4,
                random_state=self.config.random_seed,
                n_jobs=-1,
            ),
        }

        if XGB_AVAILABLE:
            regressors["xgboost"] = xgb.XGBRegressor(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=self.config.random_seed,
                verbosity=0,
            )
        else:
            regressors["grad_boost"] = GradientBoostingRegressor(
                random_state=self.config.random_seed,
                n_estimators=250,
                learning_rate=0.04,
                max_depth=3,
            )

        errors = {}
        fitted = {}
        for name, model in regressors.items():
            model.fit(X_fit, y_fit)
            pred = model.predict(X_val)
            errors[name] = mean_absolute_error(y_val, pred)
            fitted[name] = model

        inv = {name: 1.0 / max(err, 1e-6) for name, err in errors.items()}
        total = sum(inv.values())
        self.model_weights = {name: val / total for name, val in inv.items()}
        self.regressors = fitted

        self.direction_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )
        self.direction_model.fit(X_fit, fit_df["target_direction"].astype(int).values)
        self.direction_classes_ = self.direction_model.named_steps["model"].classes_

        self._fit_egarch(train_df["raw_return_1d"].dropna().astype(float).values)
        self.fitted = True

    def _fit_egarch(self, returns: np.ndarray) -> None:
        if not ARCH_AVAILABLE or len(returns) < 80:
            self.egarch = None
            return
        try:
            self.egarch = arch_model(returns * 1000, vol="EGARCH", p=1, o=1, q=1, dist="skewt").fit(
                disp="off",
                show_warning=False,
            )
        except Exception:
            self.egarch = None

    def predict(self, row: pd.Series) -> ModelOutput:
        if not self.fitted:
            raise RuntimeError("Model suite is not fitted")
        X = pd.DataFrame([row[self.feature_cols].to_dict()]).ffill().bfill().fillna(0.0)
        base_predictions = {name: float(model.predict(X)[0]) for name, model in self.regressors.items()}
        ensemble_return = float(sum(self.model_weights[name] * pred for name, pred in base_predictions.items()))

        probs = self.direction_model.predict_proba(X)[0] if self.direction_model is not None else np.array([1 / 3, 1 / 3, 1 / 3])
        prob_map = {"down": 0.0, "flat": 0.0, "up": 0.0}
        if self.direction_classes_ is not None:
            for cls, prob in zip(self.direction_classes_, probs):
                if cls == -1:
                    prob_map["down"] = float(prob)
                elif cls == 0:
                    prob_map["flat"] = float(prob)
                elif cls == 1:
                    prob_map["up"] = float(prob)
        direction_score = prob_map["up"] - prob_map["down"]
        volatility_estimate = self._volatility_estimate(row)

        return ModelOutput(
            base_predictions=base_predictions,
            model_weights=self.model_weights,
            ensemble_return=ensemble_return,
            direction_probabilities=prob_map,
            direction_score=direction_score,
            volatility_estimate=volatility_estimate,
        )

    def _volatility_estimate(self, row: pd.Series) -> float:
        if self.egarch is not None:
            try:
                forecast = self.egarch.forecast(horizon=1)
                var = float(forecast.variance.iloc[-1, 0])
                return float(max(np.sqrt(var) / 1000.0, 0.0005))
            except Exception:
                pass
        return float(max(row.get("expected_abs_move_proxy", 0.003), 0.0005))


def blend_prediction(
    row: pd.Series,
    stat_output: ModelOutput,
    persona_output: Optional[Dict],
    config: RunConfig,
) -> Dict:
    """
    Blend numeric and persona outputs into a final forecast.
    """
    stat_return = stat_output.ensemble_return
    direction_score = stat_output.direction_score

    # Direction-consistency adjustment.
    adjusted_return = stat_return
    if abs(direction_score) > 0.2:
        sign = 1.0 if direction_score > 0 else -1.0
        adjusted_return = 0.75 * stat_return + 0.25 * sign * max(abs(stat_return), stat_output.volatility_estimate * 0.5)

    persona_weight = 0.0
    persona_return = 0.0
    persona_backend = None
    if persona_output and persona_output.get("votes"):
        persona_backend = persona_output["backend"]
        persona_return = float(persona_output["expected_return"])
        strength = min(abs(persona_output.get("direction_score", 0.0)), 1.0)
        confidence = float(persona_output.get("confidence", 0.0))
        entropy = float(persona_output.get("entropy", 1.0))
        cap = config.rule_persona_weight_cap if persona_backend == "rule" else config.llm_persona_weight_cap
        persona_weight = cap * (0.35 + 0.45 * strength + 0.20 * confidence) * (1.0 - 0.35 * entropy)
        persona_weight = float(np.clip(persona_weight, 0.0, cap))

    final_return = (1.0 - persona_weight) * adjusted_return + persona_weight * persona_return
    current_price = float(row["INR"])
    predicted_price = current_price * (1.0 + final_return)
    interval = 1.96 * max(stat_output.volatility_estimate, abs(final_return) * 0.5)

    return {
        "current_price": current_price,
        "stat_return": stat_return,
        "adjusted_stat_return": adjusted_return,
        "persona_return": persona_return,
        "persona_weight": persona_weight,
        "final_return": final_return,
        "predicted_price": predicted_price,
        "ci_lower": current_price * (1.0 + final_return - interval),
        "ci_upper": current_price * (1.0 + final_return + interval),
        "direction_probabilities": stat_output.direction_probabilities,
        "direction_score": stat_output.direction_score,
        "volatility_estimate": stat_output.volatility_estimate,
        "base_predictions": stat_output.base_predictions,
        "model_weights": stat_output.model_weights,
        "persona_backend": persona_backend,
    }

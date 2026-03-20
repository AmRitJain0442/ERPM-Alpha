"""
RegimeConditionalEngine: fit/predict with regime-appropriate models.
- CALM_CARRY → Ridge Regression
- TRENDING_* → XGBoost
- HIGH_VOLATILITY / CRISIS_STRESS → EGARCH + XGBoost

Features: MA_Momentum baseline + LLM-derived + GDELT numeric + macro.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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

import config

warnings.filterwarnings("ignore", category=FutureWarning)


class RegimeConditionalEngine:
    """Fits and predicts with regime-specific models."""

    def __init__(self):
        self.models = {}       # regime → fitted model
        self.scalers = {}      # regime → fitted scaler
        self.egarch_models = {}  # regime → fitted EGARCH
        self.feature_names = None
        self.fitted = False
        self._egarch_fitted = False  # Only fit EGARCH once (slow)

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select available feature columns from the dataframe."""
        candidates = (
            config.TECHNICAL_FEATURES
            + config.GDELT_FEATURES
            + config.LLM_FEATURE_NAMES
            + config.SIMULATION_FEATURE_NAMES
            + [f"{c}_change" for c in config.MACRO_FEATURES]
        )
        available = [c for c in candidates if c in df.columns]
        return available

    def _prepare_features(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract and clean feature matrix."""
        X = df[feature_cols].copy()
        # Fill NaN with 0 for LLM features (missing = no signal)
        for col in config.LLM_FEATURE_NAMES:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
        # Forward-fill then backfill for other features
        X = X.ffill().bfill().fillna(0.0)
        return X.values, list(X.columns)

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray, regime: str) -> object:
        """Fit Ridge regression for calm/carry regime."""
        alpha = config.MODEL_CONFIG.get(regime, {}).get("ridge_alpha", 1.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y)
        self.scalers[regime] = scaler
        return model

    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray, regime: str) -> object:
        """Fit XGBoost for trending/volatile regimes."""
        if not XGB_AVAILABLE:
            # Fallback to Ridge
            return self._fit_ridge(X, y, regime)

        cfg = config.MODEL_CONFIG[regime]
        model = xgb.XGBRegressor(
            n_estimators=cfg.get("xgb_n_estimators", 200),
            max_depth=cfg.get("xgb_max_depth", 4),
            learning_rate=cfg.get("xgb_learning_rate", 0.05),
            subsample=cfg.get("xgb_subsample", 0.8),
            colsample_bytree=cfg.get("xgb_colsample_bytree", 0.8),
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        self.scalers[regime] = scaler
        return model

    def _fit_egarch(self, returns: np.ndarray, regime: str) -> Optional[object]:
        """Fit EGARCH volatility model for high-vol/crisis regimes."""
        if not ARCH_AVAILABLE:
            return None

        cfg = config.MODEL_CONFIG[regime]
        try:
            scaled_returns = returns * 100  # GARCH prefers scaled returns
            am = arch_model(
                scaled_returns,
                vol="EGARCH",
                p=cfg.get("egarch_p", 1),
                q=cfg.get("egarch_q", 1),
                o=cfg.get("egarch_o", 1),
                dist=cfg.get("egarch_dist", "skewt"),
            )
            result = am.fit(disp="off", show_warning=False)
            return result
        except Exception as e:
            print(f"[StatEngine] EGARCH fit failed for {regime}: {e}")
            return None

    def fit(self, df: pd.DataFrame, regime_labels: pd.Series):
        """Fit all regime-specific models on training data.

        Args:
            df: DataFrame with features + 'target' column
            regime_labels: Series of regime labels aligned with df index
        """
        self.feature_names = self._get_feature_columns(df)
        if not self.feature_names:
            print("[StatEngine] No features available, cannot fit")
            return

        X_all, feat_names = self._prepare_features(df, self.feature_names)
        y_all = df["target"].values

        # Filter out rows with NaN target
        valid = ~np.isnan(y_all)
        X_all = X_all[valid]
        y_all = y_all[valid]
        regimes_valid = regime_labels[valid].values if hasattr(regime_labels, 'values') else np.array(regime_labels)[valid]

        # Always fit a global fallback model on all data (Ridge is robust)
        self.models["_global"] = self._fit_ridge(X_all, y_all, "_global")

        for regime in config.REGIMES:
            mask = regimes_valid == regime
            n_samples = mask.sum()

            if n_samples < 50:
                # Not enough samples — use global fallback
                self.models[regime] = self.models["_global"]
                self.scalers[regime] = self.scalers.get("_global")
                continue

            X_regime = X_all[mask]
            y_regime = y_all[mask]
            model_type = config.MODEL_CONFIG[regime]["primary"]

            if model_type == "ridge":
                self.models[regime] = self._fit_ridge(X_regime, y_regime, regime)
            elif model_type in ("xgboost", "xgboost_egarch"):
                self.models[regime] = self._fit_xgboost(X_regime, y_regime, regime)
                if model_type == "xgboost_egarch" and not self._egarch_fitted:
                    inr_returns = df["INR_return"].dropna().values
                    if len(inr_returns) > 100:
                        self.egarch_models[regime] = self._fit_egarch(inr_returns, regime)

        self._egarch_fitted = True
        self.fitted = True

    def predict(
        self, features: pd.DataFrame, regime: str
    ) -> Tuple[float, float, Tuple[float, float]]:
        """Predict next-day INR given features and regime.

        Returns: (prediction, std_error, (lower_95, upper_95))
        """
        if not self.fitted:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        # Fallback if regime model not available
        if regime not in self.models:
            regime = config.DEFAULT_REGIME
        if regime not in self.models:
            # Ultimate fallback: use any available model
            regime = next(iter(self.models))

        model = self.models[regime]
        scaler = self.scalers.get(regime)

        X, _ = self._prepare_features(features, self.feature_names)

        if scaler is not None:
            X = scaler.transform(X)

        pred = float(model.predict(X)[0])

        # Estimate uncertainty
        std_err = self._estimate_uncertainty(regime, pred, features)

        ci_mult = config.MODEL_CONFIG.get(regime, {}).get("ci_multiplier", 1.0)
        lower = pred - 1.96 * std_err * ci_mult
        upper = pred + 1.96 * std_err * ci_mult

        return pred, std_err, (lower, upper)

    def _estimate_uncertainty(
        self, regime: str, prediction: float, features: pd.DataFrame
    ) -> float:
        """Estimate prediction uncertainty using EGARCH or heuristic."""
        # Try EGARCH forecast
        egarch = self.egarch_models.get(regime)
        if egarch is not None:
            try:
                fcast = egarch.forecast(horizon=1)
                # Variance is in scaled returns (×100), convert back
                var = fcast.variance.iloc[-1, 0]
                std_pct = np.sqrt(var) / 100
                return std_pct * prediction  # Convert pct to absolute
            except Exception:
                pass

        # Heuristic: base uncertainty on realized vol
        vol = features.get("realized_vol")
        if vol is not None and hasattr(vol, 'iloc'):
            vol_val = vol.iloc[0]
        elif vol is not None:
            vol_val = float(vol)
        else:
            vol_val = 0.003  # ~0.3% default daily vol

        if np.isnan(vol_val) or vol_val == 0:
            vol_val = 0.003

        return abs(vol_val * prediction)

    def get_feature_importance(self, regime: str) -> Optional[Dict[str, float]]:
        """Get feature importance for a regime's model."""
        if regime not in self.models or self.feature_names is None:
            return None

        model = self.models[regime]

        # XGBoost
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return dict(zip(self.feature_names, importances))

        # Ridge
        if hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            return dict(zip(self.feature_names, coefs))

        return None

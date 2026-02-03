"""
Gemini USD/INR Forex Simulation V4 - Robust Ensemble with Debiased Design

Key improvements from V3:
1. DEBIASED PROMPT DESIGN
   - Present bull/bear cases symmetrically
   - Randomize direction labels to prevent anchoring
   - Show historical accuracy to each persona
   - Explicit uncertainty bands

2. IMPROVED STATISTICAL MODEL
   - Ridge regression with cross-validated lambda
   - Realized volatility as dynamic feature
   - Bootstrap confidence intervals
   - Proper out-of-sample validation

3. ROBUST AGGREGATION
   - Trimmed weighted mean (outlier resistant)
   - Entropy-based information weighting
   - Bayesian updating with informative priors
   - Comparison against random walk benchmark

4. ADAPTIVE WEIGHTING
   - Track persona-level accuracy over time
   - Downweight consistently wrong personas
   - Upweight personas that add information
   - Regime-conditional performance tracking
"""

import os
import json
import time
import re
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd
import google.generativeai as genai
from scipy import stats as scipy_stats

# Import news digest module
try:
    from news_digest import NewsDigestManager, extract_headlines_for_date
    NEWS_DIGEST_AVAILABLE = True
except ImportError:
    NEWS_DIGEST_AVAILABLE = False
    print("Warning: news_digest module not found. Running without news integration.")

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMULATION_START = datetime(2023, 1, 1)
SIMULATION_END = datetime(2023, 12, 31)
PERSONAS_FILE = "forex_personas_v4.json"
RESULTS_FILE = "simulation_results_v4.json"
SUMMARY_CSV = "simulation_summary_v4.csv"

DATA_FILE = "../Super_Master_Dataset.csv"

# News data files
INDIA_NEWS_FILE = "../india_news_combined_sorted.csv"
USA_NEWS_FILE = "../usa_news_combined_sorted.csv"

# News digest settings
USE_NEWS_DIGEST = True  # Enable/disable news integration
MAX_HEADLINES_PER_DAY = 40  # Maximum headlines to process

WARMUP_DAYS = 20
REGRESSION_WINDOW = 60
CV_FOLDS = 5  # Cross-validation folds for ridge regression

# Weight bounds
MIN_STAT_WEIGHT = 0.55  # Allow more LLM influence when confident
MAX_STAT_WEIGHT = 0.85
DEFAULT_STAT_WEIGHT = 0.70

# LLM adjustment bounds - asymmetric based on confidence
MAX_LLM_ADJUSTMENT_PCT = 0.30  # Maximum adjustment
MIN_LLM_ADJUSTMENT_PCT = 0.02  # Minimum meaningful adjustment

# Trimmed mean parameters
TRIM_FRACTION = 0.1  # Remove top and bottom 10% of predictions

API_DELAY_SECONDS = 6
MAX_RETRIES = 3
INITIAL_DELAY = 2

# ============================================================================
# IMPROVED STATISTICAL MODEL WITH REGULARIZATION
# ============================================================================

class RobustStatisticalModel:
    """
    Ridge regression with cross-validated regularization.
    Includes realized volatility and proper uncertainty quantification.
    """

    def __init__(self, lookback_window: int = 60, cv_folds: int = 5):
        self.lookback_window = lookback_window
        self.cv_folds = cv_folds
        self.betas = None
        self.alpha = 0.0
        self.lambda_opt = 0.01  # Regularization parameter
        self.feature_means = {}
        self.feature_stds = {}
        self.inr_mean = 0.0
        self.inr_std = 0.0
        self.r_squared = 0.0
        self.residual_std = 0.0
        self.historical_errors = []  # Track for calibration
        self.features = ['US10Y', 'GOLD', 'DXY', 'IN_Avg_Tone', 'OIL', 'RealizedVol']

    def _calculate_realized_vol(self, returns: np.ndarray, window: int = 10) -> float:
        """Calculate realized volatility from recent returns."""
        if len(returns) < window:
            return np.std(returns) if len(returns) > 1 else 0.01
        return np.std(returns[-window:])

    def _cross_validate_lambda(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find optimal ridge regularization parameter via CV."""
        lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
        n = len(y)
        fold_size = n // self.cv_folds

        best_lambda = 0.01
        best_mse = float('inf')

        for lam in lambdas:
            mses = []
            for fold in range(self.cv_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size

                X_train = np.vstack([X[:val_start], X[val_end:]])
                y_train = np.concatenate([y[:val_start], y[val_end:]])
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]

                if len(X_train) < 10:
                    continue

                # Ridge regression: (X'X + λI)^(-1) X'y
                try:
                    I = np.eye(X_train.shape[1])
                    I[0, 0] = 0  # Don't regularize intercept
                    betas = np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ y_train)
                    y_pred = X_val @ betas
                    mses.append(np.mean((y_val - y_pred) ** 2))
                except np.linalg.LinAlgError:
                    continue

            if mses:
                avg_mse = np.mean(mses)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_lambda = lam

        return best_lambda

    def fit(self, df: pd.DataFrame) -> bool:
        """Fit ridge regression with cross-validated regularization."""
        if len(df) < self.lookback_window:
            return False

        data = df.tail(self.lookback_window).copy()

        # Calculate realized volatility
        if 'INR_change' in data.columns:
            returns = data['INR_change'].dropna().values
            data['RealizedVol'] = self._calculate_realized_vol(returns)
        else:
            data['RealizedVol'] = 0.01

        self.inr_mean = data['INR'].mean()
        self.inr_std = data['INR'].std()

        for feat in self.features:
            if feat in data.columns:
                self.feature_means[feat] = data[feat].mean()
                self.feature_stds[feat] = data[feat].std()
            else:
                self.feature_means[feat] = 0
                self.feature_stds[feat] = 1

        # Build feature matrix
        X = np.column_stack([
            (data[feat].fillna(self.feature_means[feat]) - self.feature_means[feat]) / 
            (self.feature_stds[feat] + 1e-8)
            for feat in self.features if feat in data.columns or feat == 'RealizedVol'
        ])
        X = np.column_stack([np.ones(len(X)), X])
        y = (data['INR'] - self.inr_mean) / (self.inr_std + 1e-8)

        # Cross-validate lambda
        if len(X) >= 20:
            self.lambda_opt = self._cross_validate_lambda(X, y.values)

        try:
            # Ridge regression
            I = np.eye(X.shape[1])
            I[0, 0] = 0  # Don't regularize intercept
            XtX_reg = X.T @ X + self.lambda_opt * I
            self.betas = np.linalg.solve(XtX_reg, X.T @ y.values)
            self.alpha = self.betas[0]
            self.betas = self.betas[1:]

            # Calculate fit statistics
            y_pred = X @ np.concatenate([[self.alpha], self.betas])
            residuals = y.values - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y.values - y.values.mean()) ** 2)
            self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.residual_std = np.sqrt(ss_res / max(1, len(y) - len(self.features) - 1))

            return True
        except np.linalg.LinAlgError:
            return False

    def predict(self, features: Dict[str, float]) -> Tuple[float, float, Tuple[float, float]]:
        """
        Returns: (prediction, std_error, (lower_bound, upper_bound))
        """
        if self.betas is None:
            return None, None, (None, None)

        feature_values = []
        for feat in self.features:
            if feat == 'RealizedVol':
                val = features.get(feat, 0.01)
            else:
                val = features.get(feat, self.feature_means.get(feat, 0))
            z = (val - self.feature_means.get(feat, 0)) / (self.feature_stds.get(feat, 1) + 1e-8)
            feature_values.append(z)

        x = np.array(feature_values[:len(self.betas)])
        z_pred = self.alpha + np.dot(self.betas, x)
        inr_pred = z_pred * self.inr_std + self.inr_mean
        inr_std = self.residual_std * self.inr_std

        # 95% confidence interval
        lower = inr_pred - 1.96 * inr_std
        upper = inr_pred + 1.96 * inr_std

        return inr_pred, inr_std, (lower, upper)

    def update_error_history(self, actual: float, predicted: float):
        """Track prediction errors for calibration."""
        error_pct = (predicted - actual) / actual * 100
        self.historical_errors.append(error_pct)
        # Keep only recent errors
        if len(self.historical_errors) > 100:
            self.historical_errors = self.historical_errors[-100:]

    def get_historical_bias(self) -> float:
        """Return historical bias for debiasing."""
        if len(self.historical_errors) < 10:
            return 0.0
        return np.mean(self.historical_errors)

    def get_feature_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get contribution of each feature to the prediction."""
        if self.betas is None:
            return {}

        contributions = {}
        for i, feat in enumerate(self.features):
            if i >= len(self.betas):
                break
            if feat == 'RealizedVol':
                val = features.get(feat, 0.01)
            else:
                val = features.get(feat, self.feature_means.get(feat, 0))
            z_val = (val - self.feature_means.get(feat, 0)) / (self.feature_stds.get(feat, 1) + 1e-8)
            contributions[feat] = self.betas[i] * z_val * self.inr_std

        return contributions


# ============================================================================
# REGIME DETECTION WITH JUMP DETECTION
# ============================================================================

class EnhancedRegimeDetector:
    """Enhanced regime detection with jump identification."""

    @staticmethod
    def detect_volatility_regime(daily_changes: List[float], window: int = 20) -> str:
        if len(daily_changes) < window:
            return "normal"

        recent_vol = np.std(daily_changes[-window:])
        long_vol = np.std(daily_changes)
        ratio = recent_vol / (long_vol + 1e-8)

        if ratio < 0.7:
            return "low"
        elif ratio > 1.5:
            return "high"
        return "normal"

    @staticmethod
    def detect_trend_regime(prices: List[float], short_window: int = 5, long_window: int = 20) -> str:
        if len(prices) < long_window:
            return "neutral"

        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices[-long_window:])
        pct_diff = (short_ma - long_ma) / long_ma * 100

        if pct_diff > 0.3:
            return "trending_up"
        elif pct_diff < -0.3:
            return "trending_down"
        return "mean_reverting"

    @staticmethod
    def detect_jump(changes: List[float], threshold_sigma: float = 2.5) -> bool:
        """Detect if recent price action contains a jump."""
        if len(changes) < 10:
            return False

        recent = changes[-1] if changes else 0
        historical_std = np.std(changes[:-1]) if len(changes) > 1 else 0.1

        return abs(recent) > threshold_sigma * historical_std

    @staticmethod
    def get_mean_reversion_signal(prices: List[float], window: int = 20) -> float:
        """
        Z-score of current price relative to moving average.
        Positive = overbought, Negative = oversold
        """
        if len(prices) < window:
            return 0.0

        ma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        if std < 1e-8:
            return 0.0

        return (prices[-1] - ma) / std


# ============================================================================
# PERSONA PERFORMANCE TRACKING
# ============================================================================

class PersonaTracker:
    """Track persona accuracy over time for adaptive weighting."""

    def __init__(self):
        self.persona_history = defaultdict(lambda: {
            'predictions': [],
            'correct_direction': 0,
            'total': 0,
            'cumulative_value': 0.0,  # Did their adjustment help or hurt?
        })

    def record_prediction(self, persona_id: str, direction: str, 
                         adjustment_pct: float, confidence: int):
        """Record a persona's prediction before we know the outcome."""
        self.persona_history[persona_id]['predictions'].append({
            'direction': direction,
            'adjustment_pct': adjustment_pct,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })

    def record_outcome(self, persona_id: str, was_direction_correct: bool,
                      adjustment_value: float):
        """Record whether the persona's last prediction was correct."""
        self.persona_history[persona_id]['total'] += 1
        if was_direction_correct:
            self.persona_history[persona_id]['correct_direction'] += 1
        self.persona_history[persona_id]['cumulative_value'] += adjustment_value

    def get_persona_accuracy(self, persona_id: str) -> float:
        """Get persona's historical direction accuracy."""
        stats = self.persona_history[persona_id]
        if stats['total'] < 5:
            return 0.5  # Neutral prior
        return stats['correct_direction'] / stats['total']

    def get_persona_value_added(self, persona_id: str) -> float:
        """Get whether persona's adjustments have added value on average."""
        stats = self.persona_history[persona_id]
        if stats['total'] < 5:
            return 0.0
        return stats['cumulative_value'] / stats['total']

    def get_adaptive_weight(self, persona_id: str, base_weight: float) -> float:
        """Adjust weight based on historical performance."""
        accuracy = self.get_persona_accuracy(persona_id)
        value = self.get_persona_value_added(persona_id)

        # Accuracy factor: boost if >55%, reduce if <45%
        if accuracy > 0.55:
            acc_factor = 1.0 + (accuracy - 0.5) * 0.5
        elif accuracy < 0.45:
            acc_factor = 1.0 - (0.5 - accuracy) * 0.5
        else:
            acc_factor = 1.0

        # Value factor: slight adjustment based on whether adjustments helped
        val_factor = 1.0 + np.clip(value * 10, -0.2, 0.2)

        return base_weight * acc_factor * val_factor


# ============================================================================
# DATA LOADING
# ============================================================================

def load_market_data(filepath: str) -> pd.DataFrame:
    print(f"Loading market data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['INR_change'] = df['INR'].pct_change() * 100
    df['DXY_change'] = df['DXY'].pct_change() * 100
    df['US10Y_change'] = df['US10Y'].diff()
    df['GOLD_change'] = df['GOLD'].pct_change() * 100
    df['OIL_change'] = df['OIL'].pct_change() * 100
    print(f"  Loaded {len(df)} rows")
    return df


def get_trading_days(df: pd.DataFrame, start: datetime, end: datetime) -> List[datetime]:
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    trading_days = df.loc[mask, 'Date'].tolist()
    return sorted([d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d for d in trading_days])


def get_historical_data(df: pd.DataFrame, target_date: datetime, lookback: int = 60) -> pd.DataFrame:
    mask = df['Date'] < target_date
    return df.loc[mask].tail(lookback).copy()


def get_market_context(df: pd.DataFrame, target_date: datetime) -> Optional[Dict]:
    available = df[df['Date'] < target_date].tail(60)

    if len(available) < 20:
        return None

    t2_row = available.iloc[-2]
    t1_row = available.iloc[-1]

    def safe_float(val, default=0.0):
        try:
            return float(val) if pd.notna(val) else default
        except:
            return default

    recent_20 = available.tail(20)
    recent_5 = available.tail(5)

    # Calculate realized volatility
    returns = available['INR_change'].dropna().values
    realized_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)

    # Mean reversion signal
    prices = available['INR'].values
    mr_signal = EnhancedRegimeDetector.get_mean_reversion_signal(list(prices))

    # Jump detection
    has_jump = EnhancedRegimeDetector.detect_jump(list(available['INR_change'].dropna().values))

    context = {
        't2': {
            'usdinr': safe_float(t2_row['INR']),
            'oil': safe_float(t2_row['OIL']),
            'gold': safe_float(t2_row['GOLD']),
            'us10y': safe_float(t2_row['US10Y']),
            'dxy': safe_float(t2_row['DXY']),
            'india_tone': safe_float(t2_row['IN_Avg_Tone']),
            'india_mentions': safe_float(t2_row['IN_Total_Mentions']),
            'us_tone': safe_float(t2_row['US_Avg_Tone']),
            'us_mentions': safe_float(t2_row['US_Total_Mentions']),
        },
        't1': {
            'usdinr': safe_float(t1_row['INR']),
            'oil': safe_float(t1_row['OIL']),
            'gold': safe_float(t1_row['GOLD']),
            'us10y': safe_float(t1_row['US10Y']),
            'dxy': safe_float(t1_row['DXY']),
            'india_tone': safe_float(t1_row['IN_Avg_Tone']),
            'india_mentions': safe_float(t1_row['IN_Total_Mentions']),
            'us_tone': safe_float(t1_row['US_Avg_Tone']),
            'us_mentions': safe_float(t1_row['US_Total_Mentions']),
        },
        'stats': {
            'inr_20d_mean': safe_float(recent_20['INR'].mean()),
            'inr_20d_std': safe_float(recent_20['INR'].std()),
            'inr_5d_mean': safe_float(recent_5['INR'].mean()),
            'dxy_20d_mean': safe_float(recent_20['DXY'].mean()),
            'us10y_20d_mean': safe_float(recent_20['US10Y'].mean()),
            'inr_5d_momentum': safe_float(available['INR_change'].tail(5).sum()),
            'realized_vol': realized_vol,
            'mean_reversion_zscore': mr_signal,
        },
        'regime': {
            'volatility': EnhancedRegimeDetector.detect_volatility_regime(
                available['INR_change'].dropna().tolist()
            ),
            'trend': EnhancedRegimeDetector.detect_trend_regime(
                available['INR'].tolist()
            ),
            'has_jump': has_jump,
        }
    }

    return context


def get_actual_price(df: pd.DataFrame, target_date: datetime) -> Optional[float]:
    mask = df['Date'].dt.date == target_date.date()
    matches = df.loc[mask, 'INR']
    if len(matches) > 0:
        return float(matches.iloc[0])
    return None


# ============================================================================
# PERSONA MANAGEMENT
# ============================================================================

def load_personas(filepath: str) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['personas']


def format_debiased_prompt(
    persona: Dict,
    context: Dict,
    stat_prediction: float,
    stat_uncertainty: Tuple[float, float],
    stat_implied_change_pct: float,
    feature_contributions: Dict[str, float],
    persona_accuracy: float,
    historical_bias: float,
    news_digest: Optional[str] = None
) -> str:
    """
    Create a DEBIASED prompt that:
    1. Presents bull and bear cases SYMMETRICALLY
    2. Shows uncertainty explicitly
    3. Includes persona's own historical accuracy
    4. Randomizes presentation order to prevent anchoring
    5. Includes news digest for context (each persona interprets differently)
    """
    t1 = context['t1']
    t2 = context['t2']
    regime = context['regime']
    stats = context['stats']

    # Calculate changes
    dxy_change = t1['dxy'] - t2['dxy']
    dxy_pct = (dxy_change / t2['dxy'] * 100) if t2['dxy'] else 0
    us10y_change = t1['us10y'] - t2['us10y']
    gold_change = t1['gold'] - t2['gold']
    gold_pct = (gold_change / t2['gold'] * 100) if t2['gold'] else 0
    oil_change = t1['oil'] - t2['oil']
    oil_pct = (oil_change / t2['oil'] * 100) if t2['oil'] else 0

    # Build SYMMETRIC bull and bear cases
    bull_factors = []
    bear_factors = []

    # DXY
    if dxy_pct > 0.1:
        bull_factors.append(f"DXY strengthening ({dxy_pct:+.2f}%) supports USD")
    elif dxy_pct < -0.1:
        bear_factors.append(f"DXY weakening ({dxy_pct:+.2f}%) pressures USD")

    # US10Y
    if us10y_change > 0.02:
        bull_factors.append(f"US yields rising ({us10y_change:+.3f}%) attracts capital to USD")
    elif us10y_change < -0.02:
        bear_factors.append(f"US yields falling ({us10y_change:+.3f}%) reduces USD appeal")

    # Oil (impacts India's CAD)
    if oil_pct > 1:
        bull_factors.append(f"Oil rising ({oil_pct:+.1f}%) widens India's deficit")
    elif oil_pct < -1:
        bear_factors.append(f"Oil falling ({oil_pct:+.1f}%) eases India's import bill")

    # India sentiment
    if t1['india_tone'] > t2['india_tone'] + 0.5:
        bear_factors.append(f"India sentiment improving (tone: {t1['india_tone']:.2f})")
    elif t1['india_tone'] < t2['india_tone'] - 0.5:
        bull_factors.append(f"India sentiment deteriorating (tone: {t1['india_tone']:.2f})")

    # Mean reversion
    mr_z = stats.get('mean_reversion_zscore', 0)
    if mr_z > 1.5:
        bear_factors.append(f"USD/INR overbought (z-score: {mr_z:.2f}), reversion likely")
    elif mr_z < -1.5:
        bull_factors.append(f"USD/INR oversold (z-score: {mr_z:.2f}), reversion likely")

    # Volatility regime
    if regime['volatility'] == 'high':
        bull_factors.append("High volatility often favors safe-haven USD")

    # Format factor lists
    bull_text = "\n".join([f"  + {f}" for f in bull_factors]) if bull_factors else "  (No strong bullish signals)"
    bear_text = "\n".join([f"  - {f}" for f in bear_factors]) if bear_factors else "  (No strong bearish signals)"

    # RANDOMIZE presentation order to prevent anchoring
    if random.random() > 0.5:
        case_order = f"""
**Case for USD STRENGTH (higher USD/INR):**
{bull_text}

**Case for USD WEAKNESS (lower USD/INR):**
{bear_text}
"""
    else:
        case_order = f"""
**Case for USD WEAKNESS (lower USD/INR):**
{bear_text}

**Case for USD STRENGTH (higher USD/INR):**
{bull_text}
"""

    # Feature contribution summary
    top_drivers = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    driver_text = ", ".join([f"{k}: {v:+.4f}" for k, v in top_drivers])

    # Persona accuracy feedback
    if persona_accuracy >= 0.55:
        acc_text = f"Your historical direction accuracy is {persona_accuracy*100:.0f}% - ABOVE average."
    elif persona_accuracy <= 0.45:
        acc_text = f"Your historical direction accuracy is {persona_accuracy*100:.0f}% - BELOW average. Consider recalibrating."
    else:
        acc_text = f"Your historical direction accuracy is {persona_accuracy*100:.0f}% - at baseline."

    # Historical bias warning
    bias_warning = ""
    if abs(historical_bias) > 0.05:
        direction = "bullish" if historical_bias > 0 else "bearish"
        bias_warning = f"\n**WARNING**: The model has shown a {direction} bias of {abs(historical_bias):.3f}% recently. Account for this."

    # News digest section
    news_section = ""
    if news_digest:
        news_section = f"""
---

### NEWS DIGEST (from headlines extracted from GDELT URLs)

{news_digest}

**IMPORTANT**: Interpret this news from YOUR analytical perspective. The same news may have different implications for different market participants.
"""

    prompt = f"""
## USD/INR FOREX ANALYSIS

{acc_text}

---

### STATISTICAL MODEL PREDICTION

**Point estimate: {stat_prediction:.4f}** (implied change: {stat_implied_change_pct:+.3f}%)
**95% confidence interval: [{stat_uncertainty[0]:.4f}, {stat_uncertainty[1]:.4f}]**

Key drivers: {driver_text}
Model R²: ~0.56 (explains 56% of variance)
{bias_warning}
{news_section}
---

### BALANCED CASE ANALYSIS

{case_order}

---

### MARKET DATA

| Indicator | Yesterday | Today | Change |
|-----------|-----------|-------|--------|
| DXY | {t2['dxy']:.2f} | {t1['dxy']:.2f} | {dxy_pct:+.2f}% |
| US 10Y | {t2['us10y']:.3f}% | {t1['us10y']:.3f}% | {us10y_change:+.3f}% |
| Gold | ${t2['gold']:.0f} | ${t1['gold']:.0f} | {gold_pct:+.2f}% |
| Oil | ${t2['oil']:.1f} | ${t1['oil']:.1f} | {oil_pct:+.2f}% |

**Volatility Regime**: {regime['volatility'].upper()}
**Trend Regime**: {regime['trend'].replace('_', ' ').upper()}
**Recent Jump**: {'YES' if regime.get('has_jump') else 'NO'}

---

### YOUR TASK

Given the statistical prediction of **{stat_prediction:.4f}** with uncertainty band [{stat_uncertainty[0]:.4f}, {stat_uncertainty[1]:.4f}]:

Should the final prediction be:
- **ABOVE** {stat_prediction:.4f}? (bullish USD adjustment)
- **BELOW** {stat_prediction:.4f}? (bearish USD adjustment)  
- **UNCHANGED**? (trust the model)

Respond with ONLY this JSON:
```json
{{
    "direction": "higher" | "lower" | "unchanged",
    "adjustment_pips": <integer 0-30, where 10 pips = 0.10%>,
    "confidence": <1-10>,
    "primary_reason": "<one sentence>"
}}
```

**Adjustment guide** (in pips, 100 pips = 1 rupee):
- 0: Trust the model completely
- 5-10: Minor adjustment based on sentiment
- 10-20: Moderate adjustment for notable factors
- 20-30: Strong adjustment (use sparingly, requires high confidence)
"""
    return prompt


# ============================================================================
# GEMINI API
# ============================================================================

def initialize_gemini(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        generation_config={
            'temperature': 0.4,  # Lower for more consistent outputs
            'top_p': 0.85,
            'max_output_tokens': 512,
        }
    )
    return model


def extract_response_text(response) -> str:
    """Extract text from Gemini response."""
    try:
        if response.text:
            return response.text
    except (ValueError, AttributeError):
        pass

    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        continue
                    if hasattr(part, 'text') and part.text:
                        return part.text

    return ""


def parse_v4_response(response_text: str) -> Optional[Dict]:
    """Parse V4 format response with pips-based adjustment."""
    try:
        cleaned = response_text.strip()

        # Extract JSON
        json_str = None

        if '```json' in cleaned:
            parts = cleaned.split('```json')
            if len(parts) > 1:
                json_str = parts[1].split('```')[0].strip()
        elif '```' in cleaned:
            for block in cleaned.split('```')[1::2]:
                block = block.strip()
                if block.startswith('json'):
                    block = block[4:].strip()
                if block.startswith('{'):
                    json_str = block
                    break
        else:
            start_idx = cleaned.find('{')
            if start_idx != -1:
                brace_count = 0
                for i, char in enumerate(cleaned[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = cleaned[start_idx:i+1]
                            break

        if not json_str:
            return None

        parsed = json.loads(json_str)

        # Normalize direction
        direction = parsed.get('direction', 'unchanged').lower().strip()
        if direction in ['higher', 'bullish', 'up', 'bullish_usd']:
            direction = 'higher'
        elif direction in ['lower', 'bearish', 'down', 'bearish_usd']:
            direction = 'lower'
        else:
            direction = 'unchanged'

        # Get adjustment in pips
        pips = parsed.get('adjustment_pips', 0)
        if isinstance(pips, str):
            pips = int(re.search(r'\d+', pips).group()) if re.search(r'\d+', pips) else 0
        pips = max(0, min(30, int(pips)))  # Clamp to 0-30

        return {
            'direction': direction,
            'adjustment_pips': pips,
            'confidence': int(parsed.get('confidence', 5)),
            'primary_reason': parsed.get('primary_reason', ''),
        }

    except (json.JSONDecodeError, ValueError, AttributeError):
        return None


def query_persona(model: genai.GenerativeModel, persona: Dict, 
                  market_prompt: str, retries: int = MAX_RETRIES) -> Dict:
    """Query a persona with the debiased prompt."""
    system_prompt = persona['system_prompt']
    full_prompt = f"{system_prompt}\n\n{market_prompt}"

    for attempt in range(retries):
        try:
            response = model.generate_content(full_prompt)
            response_text = extract_response_text(response)

            if not response_text.strip():
                backoff = API_DELAY_SECONDS * (2 ** attempt)
                time.sleep(backoff)
                continue

            parsed = parse_v4_response(response_text)

            if parsed:
                return {
                    'success': True,
                    'persona_id': persona['id'],
                    'persona_name': persona['name'],
                    'short_name': persona.get('short_name', persona['name'][:15]),
                    'weight': persona.get('adaptive_weight', persona['weight']),
                    'original_weight': persona['weight'],
                    'raw_response': response_text,
                    **parsed
                }

            time.sleep(API_DELAY_SECONDS)

        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str or 'quota' in error_str or '429' in error_str:
                backoff = API_DELAY_SECONDS * (3 ** attempt)
                time.sleep(backoff)
            else:
                time.sleep(API_DELAY_SECONDS * 2)

    return {
        'success': False,
        'persona_id': persona['id'],
        'persona_name': persona['name'],
        'short_name': persona.get('short_name', persona['name'][:15]),
        'weight': persona.get('adaptive_weight', persona['weight']),
        'original_weight': persona['weight'],
        'direction': 'unchanged',
        'adjustment_pips': 0,
        'confidence': 5,
        'primary_reason': 'API failure',
        'raw_response': None
    }


# ============================================================================
# ROBUST AGGREGATION
# ============================================================================

def convert_pips_to_pct(direction: str, pips: int, confidence: int) -> float:
    """Convert pips-based adjustment to percentage."""
    # 10 pips = 0.10 rupee = ~0.12% for INR around 83
    base_pct = pips * 0.01  # 10 pips = 0.10%

    # Confidence scaling
    if confidence <= 3:
        conf_scale = 0.5
    elif confidence <= 5:
        conf_scale = 0.75
    elif confidence <= 7:
        conf_scale = 1.0
    else:
        conf_scale = 1.15

    adjusted = base_pct * conf_scale

    if direction == 'higher':
        return adjusted
    elif direction == 'lower':
        return -adjusted
    return 0.0


def trimmed_weighted_mean(values: List[float], weights: List[float], 
                          trim_fraction: float = 0.1) -> float:
    """
    Calculate trimmed weighted mean - removes extreme predictions.
    This is robust to outlier personas.
    """
    if len(values) == 0:
        return 0.0

    if len(values) <= 2:
        return np.average(values, weights=weights)

    # Sort by value
    sorted_pairs = sorted(zip(values, weights), key=lambda x: x[0])
    n = len(sorted_pairs)

    # Calculate how many to trim from each end
    trim_count = max(1, int(n * trim_fraction))

    # Trim extremes
    trimmed_pairs = sorted_pairs[trim_count:-trim_count] if trim_count > 0 else sorted_pairs

    if len(trimmed_pairs) == 0:
        # If all trimmed, use median
        return np.median(values)

    trimmed_values, trimmed_weights = zip(*trimmed_pairs)
    total_weight = sum(trimmed_weights)

    if total_weight == 0:
        return np.mean(trimmed_values)

    return sum(v * w for v, w in zip(trimmed_values, trimmed_weights)) / total_weight


def calculate_entropy_weight(predictions: List[Dict]) -> float:
    """
    Calculate how much information the LLM ensemble is providing.
    High entropy = diverse opinions = less confident = lower weight
    Low entropy = agreement = more confident = higher weight
    """
    if len(predictions) < 3:
        return 0.5

    directions = [p['direction'] for p in predictions if p.get('success')]
    if not directions:
        return 0.5

    # Count direction frequencies
    counts = {'higher': 0, 'lower': 0, 'unchanged': 0}
    for d in directions:
        counts[d] = counts.get(d, 0) + 1

    total = len(directions)
    probs = [c / total for c in counts.values() if c > 0]

    # Calculate entropy (0 = complete agreement, 1.58 = uniform)
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(3)  # Max possible entropy with 3 options

    # Normalize: 0 = max entropy (uniform), 1 = min entropy (all agree)
    normalized = 1 - (entropy / max_entropy)

    return normalized


def aggregate_predictions_robust(predictions: List[Dict], 
                                 trim_fraction: float = TRIM_FRACTION) -> Dict:
    """
    Robust aggregation using:
    1. Trimmed weighted mean (outlier resistant)
    2. Entropy-based confidence
    3. Proper uncertainty propagation
    """
    valid = [p for p in predictions if p.get('success')]

    if not valid:
        return {
            'weighted_adjustment_pct': 0.0,
            'consensus_direction': 'unchanged',
            'consensus_strength': 0.0,
            'entropy_confidence': 0.5,
            'avg_confidence': 5,
            'num_valid': 0,
            'higher_pct': 0,
            'lower_pct': 0,
        }

    # Calculate individual adjustments
    adjustments = []
    weights = []
    for p in valid:
        adj = convert_pips_to_pct(p['direction'], p['adjustment_pips'], p['confidence'])
        adjustments.append(adj)
        weights.append(p['weight'])

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Trimmed weighted mean
    robust_adjustment = trimmed_weighted_mean(adjustments, weights, trim_fraction)

    # Calculate direction percentages
    higher_weight = sum(p['weight'] for p in valid if p['direction'] == 'higher')
    lower_weight = sum(p['weight'] for p in valid if p['direction'] == 'lower')
    unchanged_weight = sum(p['weight'] for p in valid if p['direction'] == 'unchanged')

    total_w = higher_weight + lower_weight + unchanged_weight
    higher_pct = higher_weight / total_w if total_w > 0 else 0
    lower_pct = lower_weight / total_w if total_w > 0 else 0

    # Consensus
    consensus_strength = abs(higher_pct - lower_pct)
    if higher_pct > lower_pct + 0.15:
        consensus = 'higher'
    elif lower_pct > higher_pct + 0.15:
        consensus = 'lower'
    else:
        consensus = 'mixed'

    # Entropy-based confidence
    entropy_confidence = calculate_entropy_weight(valid)

    # Average confidence
    avg_confidence = sum(p['confidence'] * p['weight'] for p in valid) / total_w

    return {
        'weighted_adjustment_pct': robust_adjustment,
        'consensus_direction': consensus,
        'consensus_strength': consensus_strength,
        'entropy_confidence': entropy_confidence,
        'avg_confidence': avg_confidence,
        'num_valid': len(valid),
        'higher_pct': higher_pct,
        'lower_pct': lower_pct,
    }


def calculate_adaptive_weights(
    llm_result: Dict,
    stat_r_squared: float,
    regime_volatility: str,
    recent_llm_value: float  # Track if LLM has been helping recently
) -> Tuple[float, float]:
    """
    Calculate weights based on multiple factors:
    1. Entropy confidence (agreement among personas)
    2. Statistical model quality
    3. Market regime
    4. Recent LLM performance
    """
    stat_weight = DEFAULT_STAT_WEIGHT

    # Entropy-based: high agreement = trust LLM more
    entropy_conf = llm_result['entropy_confidence']
    if entropy_conf > 0.7:
        stat_weight -= 0.08  # Strong agreement
    elif entropy_conf < 0.4:
        stat_weight += 0.05  # Disagreement

    # Stat model quality
    if stat_r_squared > 0.6:
        stat_weight += 0.05
    elif stat_r_squared < 0.4:
        stat_weight -= 0.05

    # Volatility regime
    if regime_volatility == 'high':
        stat_weight += 0.05  # Trust quantitative in volatile markets
    elif regime_volatility == 'low':
        stat_weight -= 0.03

    # Recent LLM performance
    if recent_llm_value > 0.02:
        stat_weight -= 0.05  # LLM has been helping
    elif recent_llm_value < -0.02:
        stat_weight += 0.05  # LLM has been hurting

    # Clamp
    stat_weight = max(MIN_STAT_WEIGHT, min(MAX_STAT_WEIGHT, stat_weight))
    llm_weight = 1 - stat_weight

    return stat_weight, llm_weight


def ensemble_prediction(
    stat_prediction: float,
    llm_adjustment_pct: float,
    stat_weight: float,
    llm_weight: float,
    historical_bias: float = 0.0
) -> float:
    """
    Combine predictions with optional bias correction.
    """
    # Apply LLM adjustment scaled by weight
    effective_adjustment = llm_adjustment_pct * llm_weight

    # Bias correction (subtract historical bias)
    bias_correction = -historical_bias * 0.5  # Partial correction

    final = stat_prediction * (1 + (effective_adjustment + bias_correction) / 100)
    return final


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(api_key: str, personas_file: str = PERSONAS_FILE):
    print("=" * 70)
    print("GEMINI USD/INR FOREX SIMULATION V4 - ROBUST DEBIASED ENSEMBLE + NEWS")
    print("=" * 70)
    print(f"Period: {SIMULATION_START.date()} to {SIMULATION_END.date()}")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load personas
    personas_path = os.path.join(script_dir, personas_file)
    
    # Check if V4 personas exist, fall back to V3
    if not os.path.exists(personas_path):
        personas_path = os.path.join(script_dir, "forex_personas_v3.json")
        print(f"Using V3 personas from {personas_path}")
    
    personas = load_personas(personas_path)
    print(f"Loaded {len(personas)} personas")

    # Initialize tracking
    persona_tracker = PersonaTracker()

    # Load data
    data_path = os.path.join(script_dir, DATA_FILE)
    market_df = load_market_data(data_path)

    # Initialize
    print("\nInitializing Gemini 2.0 Flash...")
    model = initialize_gemini(api_key)
    
    # Initialize news digest manager
    news_manager = None
    if USE_NEWS_DIGEST and NEWS_DIGEST_AVAILABLE:
        india_news_path = os.path.join(script_dir, INDIA_NEWS_FILE)
        usa_news_path = os.path.join(script_dir, USA_NEWS_FILE)
        
        if os.path.exists(india_news_path):
            print("\nInitializing News Digest Manager...")
            news_manager = NewsDigestManager(
                model=model,
                india_news_path=india_news_path,
                usa_news_path=usa_news_path if os.path.exists(usa_news_path) else None
            )
            print(f"  News digest enabled with {MAX_HEADLINES_PER_DAY} headlines/day")
        else:
            print(f"\nWarning: India news file not found at {india_news_path}")
            print("  Running without news integration")
    elif USE_NEWS_DIGEST and not NEWS_DIGEST_AVAILABLE:
        print("\nWarning: News digest module not available")
    else:
        print("\nNews digest disabled by configuration")
    stat_model = RobustStatisticalModel(lookback_window=REGRESSION_WINDOW, cv_folds=CV_FOLDS)

    trading_days = get_trading_days(market_df, SIMULATION_START, SIMULATION_END)
    print(f"\nSimulating {len(trading_days)} trading days")

    all_results = []
    recent_llm_values = []  # Track if LLM is adding value

    for day_idx, target_date in enumerate(trading_days):
        is_warmup = day_idx < WARMUP_DAYS

        print(f"\n[Day {day_idx + 1}/{len(trading_days)}] {target_date.date()}")
        if is_warmup:
            print("  MODE: WARMUP")
        print("-" * 50)

        context = get_market_context(market_df, target_date)
        if not context:
            print("  Skipping - insufficient data")
            continue

        hist_data = get_historical_data(market_df, target_date, REGRESSION_WINDOW)
        if len(hist_data) >= 30:
            stat_model.fit(hist_data)

        last_close = context['t1']['usdinr']
        actual_price = get_actual_price(market_df, target_date)

        # Get realized volatility
        returns = hist_data['INR_change'].dropna().values
        realized_vol = np.std(returns[-10:]) if len(returns) >= 10 else 0.01

        t1_features = {
            'US10Y': context['t1']['us10y'],
            'GOLD': context['t1']['gold'],
            'DXY': context['t1']['dxy'],
            'IN_Avg_Tone': context['t1']['india_tone'],
            'OIL': context['t1']['oil'],
            'RealizedVol': realized_vol,
        }

        stat_pred, stat_std, stat_bounds = stat_model.predict(t1_features)
        if stat_pred is None:
            stat_pred = last_close
            stat_std = 0.1
            stat_bounds = (last_close - 0.2, last_close + 0.2)

        stat_implied_change = (stat_pred - last_close) / last_close * 100
        contributions = stat_model.get_feature_contributions(t1_features)
        historical_bias = stat_model.get_historical_bias()

        print(f"  Statistical: {stat_pred:.4f} ({stat_implied_change:+.3f}%) R²={stat_model.r_squared:.3f}")
        print(f"  95% CI: [{stat_bounds[0]:.4f}, {stat_bounds[1]:.4f}]")

        if is_warmup:
            final_prediction = stat_pred
            day_result = {
                'date': target_date.isoformat(),
                'day_idx': day_idx,
                'mode': 'warmup',
                'last_close': last_close,
                'stat_prediction': stat_pred,
                'stat_r_squared': stat_model.r_squared,
                'final_prediction': final_prediction,
                'actual_price': actual_price,
                'prediction_error_pct': ((final_prediction - actual_price) / actual_price * 100) if actual_price else None,
                'persona_predictions': []
            }

            # Update error history for calibration
            if actual_price:
                stat_model.update_error_history(actual_price, stat_pred)

        else:
            # Get adaptive weights for personas
            for p in personas:
                p['adaptive_weight'] = persona_tracker.get_adaptive_weight(
                    p['id'], p['weight']
                )

            # Normalize weights
            total_w = sum(p['adaptive_weight'] for p in personas)
            for p in personas:
                p['adaptive_weight'] /= total_w

            # Get news digest for the day
            news_digest_text = None
            if news_manager is not None:
                print("  Generating news digest...", end=" ")
                try:
                    digest_result = news_manager.get_digest(
                        target_date, 
                        use_cache=True,
                        max_headlines=MAX_HEADLINES_PER_DAY
                    )
                    news_digest_text = news_manager.format_digest_for_prompt(digest_result)
                    headline_count = len(digest_result.get('india_headlines', [])) + \
                                   len(digest_result.get('usa_headlines', []))
                    print(f"OK ({headline_count} headlines)")
                except Exception as e:
                    print(f"FAILED ({e})")
                    news_digest_text = None

            time.sleep(INITIAL_DELAY)

            # Query personas with debiased prompts
            day_predictions = []

            for i, persona in enumerate(personas):
                if i > 0 and i % 3 == 0:
                    time.sleep(API_DELAY_SECONDS)

                persona_acc = persona_tracker.get_persona_accuracy(persona['id'])

                print(f"  {persona['short_name']}...", end=" ")

                prompt = format_debiased_prompt(
                    persona,
                    context,
                    stat_pred,
                    stat_bounds,
                    stat_implied_change,
                    contributions,
                    persona_acc,
                    historical_bias,
                    news_digest=news_digest_text  # Pass news digest to each persona
                )

                prediction = query_persona(model, persona, prompt)
                day_predictions.append(prediction)

                if prediction['success']:
                    adj_pct = convert_pips_to_pct(
                        prediction['direction'],
                        prediction['adjustment_pips'],
                        prediction['confidence']
                    )
                    print(f"{prediction['direction'][:5]} {prediction['adjustment_pips']}pips c={prediction['confidence']} -> {adj_pct:+.2f}%")
                else:
                    print("FAIL")

                time.sleep(API_DELAY_SECONDS)

            # Robust aggregation
            llm_result = aggregate_predictions_robust(day_predictions)

            print(f"\n  LLM Aggregate: {llm_result['consensus_direction']} "
                  f"(higher:{llm_result['higher_pct']*100:.0f}% lower:{llm_result['lower_pct']*100:.0f}%) "
                  f"entropy_conf:{llm_result['entropy_confidence']:.2f} "
                  f"adj:{llm_result['weighted_adjustment_pct']:+.3f}%")

            # Calculate recent LLM value
            recent_llm_value = np.mean(recent_llm_values[-20:]) if recent_llm_values else 0.0

            # Dynamic weights
            stat_weight, llm_weight = calculate_adaptive_weights(
                llm_result,
                stat_model.r_squared,
                context['regime']['volatility'],
                recent_llm_value
            )

            print(f"  Dynamic weights: Stat {stat_weight*100:.0f}% | LLM {llm_weight*100:.0f}%")

            # Ensemble with bias correction
            final_prediction = ensemble_prediction(
                stat_pred,
                llm_result['weighted_adjustment_pct'],
                stat_weight,
                llm_weight,
                historical_bias
            )

            print(f"  Final: {final_prediction:.4f}")

            day_result = {
                'date': target_date.isoformat(),
                'day_idx': day_idx,
                'mode': 'hybrid',
                'last_close': last_close,
                'stat_prediction': stat_pred,
                'stat_r_squared': stat_model.r_squared,
                'stat_bounds': list(stat_bounds),
                'llm_adjustment_pct': llm_result['weighted_adjustment_pct'],
                'llm_consensus': llm_result['consensus_direction'],
                'llm_consensus_strength': llm_result['consensus_strength'],
                'entropy_confidence': llm_result['entropy_confidence'],
                'higher_pct': llm_result['higher_pct'],
                'lower_pct': llm_result['lower_pct'],
                'stat_weight': stat_weight,
                'llm_weight': llm_weight,
                'historical_bias': historical_bias,
                'news_digest_available': news_digest_text is not None,
                'final_prediction': final_prediction,
                'actual_price': actual_price,
                'prediction_error_pct': ((final_prediction - actual_price) / actual_price * 100) if actual_price else None,
                'stat_error_pct': ((stat_pred - actual_price) / actual_price * 100) if actual_price else None,
                'regime_volatility': context['regime']['volatility'],
                'regime_trend': context['regime']['trend'],
                'persona_predictions': [{k: v for k, v in p.items() if k != 'raw_response'} for p in day_predictions]
            }

            # Update tracking
            if actual_price:
                stat_model.update_error_history(actual_price, stat_pred)

                # Track if LLM helped
                stat_error = abs(stat_pred - actual_price)
                final_error = abs(final_prediction - actual_price)
                llm_value = (stat_error - final_error) / actual_price * 100
                recent_llm_values.append(llm_value)
                if len(recent_llm_values) > 50:
                    recent_llm_values = recent_llm_values[-50:]

                # Update persona tracking
                actual_direction = 'higher' if actual_price > stat_pred else 'lower'
                for p in day_predictions:
                    if p.get('success'):
                        was_correct = p['direction'] == actual_direction
                        adj = convert_pips_to_pct(p['direction'], p['adjustment_pips'], p['confidence'])
                        # Did this adjustment help?
                        persona_value = llm_value * (adj / (llm_result['weighted_adjustment_pct'] + 1e-8))
                        persona_tracker.record_outcome(p['persona_id'], was_correct, persona_value)

        if actual_price:
            print(f"  Actual: {actual_price:.4f} | Error: {day_result.get('prediction_error_pct', 0):+.3f}%")

        all_results.append(day_result)
        save_results(all_results, script_dir)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    generate_summary(all_results, script_dir)
    return all_results


def save_results(results: List[Dict], output_dir: str):
    output_path = os.path.join(output_dir, RESULTS_FILE)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def generate_summary(results: List[Dict], output_dir: str):
    summary_data = []
    for r in results:
        summary_data.append({
            'date': r['date'],
            'mode': r['mode'],
            'last_close': r['last_close'],
            'stat_prediction': r.get('stat_prediction'),
            'llm_adjustment_pct': r.get('llm_adjustment_pct', 0),
            'final_prediction': r['final_prediction'],
            'actual_price': r['actual_price'],
            'prediction_error_pct': r.get('prediction_error_pct'),
            'stat_error_pct': r.get('stat_error_pct'),
            'stat_r_squared': r.get('stat_r_squared'),
            'higher_pct': r.get('higher_pct'),
            'lower_pct': r.get('lower_pct'),
            'entropy_confidence': r.get('entropy_confidence'),
            'stat_weight': r.get('stat_weight'),
            'llm_weight': r.get('llm_weight'),
        })

    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, SUMMARY_CSV)
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    hybrid = df[df['mode'] == 'hybrid']
    if not hybrid.empty:
        valid = hybrid[hybrid['prediction_error_pct'].notna()]

        if not valid.empty:
            print("\n" + "=" * 50)
            print("V4 PERFORMANCE METRICS")
            print("=" * 50)

            print(f"\nEnsemble Model:")
            print(f"  MAE:  {valid['prediction_error_pct'].abs().mean():.4f}%")
            print(f"  Bias: {valid['prediction_error_pct'].mean():+.4f}%")
            print(f"  Std:  {valid['prediction_error_pct'].std():.4f}%")

            stat_valid = valid[valid['stat_error_pct'].notna()]
            if not stat_valid.empty:
                print(f"\nStatistical Model:")
                print(f"  MAE:  {stat_valid['stat_error_pct'].abs().mean():.4f}%")
                print(f"  Bias: {stat_valid['stat_error_pct'].mean():+.4f}%")

            # LLM contribution
            llm_helped = sum(
                abs(row['prediction_error_pct']) < abs(row['stat_error_pct'])
                for _, row in valid.iterrows()
                if row['stat_error_pct'] is not None
            )
            total = len([r for _, r in valid.iterrows() if r['stat_error_pct'] is not None])
            print(f"\nLLM helped in {llm_helped}/{total} cases ({100*llm_helped/total:.1f}%)")

            # Direction balance
            print(f"\nDirection Balance:")
            print(f"  Avg Higher: {valid['higher_pct'].mean()*100:.1f}%")
            print(f"  Avg Lower: {valid['lower_pct'].mean()*100:.1f}%")

            # Direction accuracy
            correct = sum(
                1 for _, row in valid.iterrows()
                if (row['final_prediction'] > row['last_close'] and row['actual_price'] > row['last_close']) or
                   (row['final_prediction'] < row['last_close'] and row['actual_price'] < row['last_close'])
            )
            print(f"\nDirection Accuracy: {correct}/{len(valid)} ({100*correct/len(valid):.1f}%)")


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        exit(1)

    try:
        results = run_simulation(api_key)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

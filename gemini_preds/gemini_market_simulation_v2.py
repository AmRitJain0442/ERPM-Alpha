"""
Gemini USD/INR Forex Simulation V2 - Hybrid Statistical + LLM Architecture

Key improvements over V1:
1. Statistical baseline model (OLS regression) does the numerical heavy lifting
2. LLM provides qualitative directional adjustment, not exact prices
3. Rolling anchor system prevents drift
4. Z-scores and momentum indicators for better feature engineering
5. Calibrated ensemble combining statistical + LLM predictions
6. Regime-aware adjustments
7. Historical correlation context in prompts

The core insight: LLMs are good at qualitative reasoning, statistical models
are good at numerical relationships. Use each for what they're best at.
"""

import os
import json
import time
import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import google.generativeai as genai

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMULATION_START = datetime(2023, 1, 1)
SIMULATION_END = datetime(2023, 12, 31)
PERSONAS_FILE = "forex_personas.json"
RESULTS_FILE = "simulation_results_v2.json"
SUMMARY_CSV = "simulation_summary_v2.csv"

# Data file (relative to project root)
DATA_FILE = "../Super_Master_Dataset.csv"

# Warmup period for statistical model calibration
WARMUP_DAYS = 20  # Need more days to estimate regression coefficients

# Rolling window for statistical model
REGRESSION_WINDOW = 60  # Use last 60 days to estimate betas

# Ensemble weights (will be dynamically adjusted based on performance)
INITIAL_STAT_WEIGHT = 0.6  # Statistical model gets 60% weight initially
INITIAL_LLM_WEIGHT = 0.4   # LLM adjustment gets 40% weight initially

# LLM adjustment bounds (prevents wild predictions)
MAX_LLM_ADJUSTMENT_PCT = 0.5  # LLM can adjust by at most ±0.5%

# Rate limiting
API_DELAY_SECONDS = 2
MAX_RETRIES = 3

# ============================================================================
# STATISTICAL MODEL
# ============================================================================

class StatisticalModel:
    """
    OLS regression model for USD/INR prediction based on correlated features.

    Uses the strong correlations discovered in data analysis:
    - US10Y: 0.84 correlation (strongest)
    - GOLD: 0.80 correlation
    - DXY: 0.62 correlation
    - IN_Avg_Tone: 0.44 correlation
    """

    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window
        self.betas = None
        self.alpha = 0.0
        self.feature_means = {}
        self.feature_stds = {}
        self.inr_mean = 0.0
        self.inr_std = 0.0
        self.r_squared = 0.0
        self.residual_std = 0.0

        # Features to use (ordered by correlation strength)
        self.features = ['US10Y', 'GOLD', 'DXY', 'IN_Avg_Tone', 'OIL']

    def fit(self, df: pd.DataFrame) -> bool:
        """
        Fit the regression model on historical data.
        Predicts INR level from feature levels (not changes).
        """
        if len(df) < self.lookback_window:
            return False

        # Use most recent data
        data = df.tail(self.lookback_window).copy()

        # Store normalization parameters
        self.inr_mean = data['INR'].mean()
        self.inr_std = data['INR'].std()

        for feat in self.features:
            self.feature_means[feat] = data[feat].mean()
            self.feature_stds[feat] = data[feat].std()

        # Prepare features (z-scored for numerical stability)
        X = np.column_stack([
            (data[feat] - self.feature_means[feat]) / (self.feature_stds[feat] + 1e-8)
            for feat in self.features
        ])

        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])

        # Target: z-scored INR
        y = (data['INR'] - self.inr_mean) / (self.inr_std + 1e-8)

        # OLS regression: beta = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            self.betas = XtX_inv @ X.T @ y
            self.alpha = self.betas[0]
            self.betas = self.betas[1:]

            # Calculate R-squared and residual std
            y_pred = X @ np.concatenate([[self.alpha], self.betas])
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.residual_std = np.sqrt(ss_res / (len(y) - len(self.features) - 1))

            return True
        except np.linalg.LinAlgError:
            return False

    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict INR value and confidence interval.
        Returns (predicted_inr, prediction_std)
        """
        if self.betas is None:
            return None, None

        # Z-score the input features
        x = np.array([
            (features.get(feat, self.feature_means[feat]) - self.feature_means[feat])
            / (self.feature_stds[feat] + 1e-8)
            for feat in self.features
        ])

        # Predict z-scored INR
        z_pred = self.alpha + np.dot(self.betas, x)

        # Convert back to INR scale
        inr_pred = z_pred * self.inr_std + self.inr_mean
        inr_std = self.residual_std * self.inr_std  # Approximate prediction std

        return inr_pred, inr_std

    def get_feature_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Get contribution of each feature to the prediction.
        Useful for explaining to the LLM what's driving the statistical forecast.
        """
        if self.betas is None:
            return {}

        contributions = {}
        for i, feat in enumerate(self.features):
            z_val = (features.get(feat, self.feature_means[feat]) - self.feature_means[feat]) / (self.feature_stds[feat] + 1e-8)
            # Contribution in INR terms
            contributions[feat] = self.betas[i] * z_val * self.inr_std

        return contributions


class MomentumIndicators:
    """
    Calculate momentum and mean-reversion indicators for features.
    """

    @staticmethod
    def calculate_z_score(current: float, mean: float, std: float) -> float:
        """How many std devs from mean."""
        if std < 1e-8:
            return 0.0
        return (current - mean) / std

    @staticmethod
    def calculate_momentum(values: List[float], window: int = 5) -> float:
        """Rate of change over window."""
        if len(values) < window + 1:
            return 0.0
        return (values[-1] - values[-window-1]) / (values[-window-1] + 1e-8) * 100

    @staticmethod
    def calculate_rsi(changes: List[float], window: int = 14) -> float:
        """Relative Strength Index (0-100)."""
        if len(changes) < window:
            return 50.0  # Neutral

        recent = changes[-window:]
        gains = [c for c in recent if c > 0]
        losses = [-c for c in recent if c < 0]

        avg_gain = sum(gains) / window if gains else 0
        avg_loss = sum(losses) / window if losses else 0

        if avg_loss < 1e-8:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class RegimeDetector:
    """
    Detect market regime for adaptive predictions.
    """

    @staticmethod
    def detect_volatility_regime(daily_changes: List[float], window: int = 20) -> str:
        """
        Classify volatility as low/normal/high.
        """
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
        """
        Classify as trending up/down or mean-reverting.
        """
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


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_market_data(filepath: str) -> pd.DataFrame:
    """Load market data from CSV."""
    print(f"Loading market data from {filepath}...")

    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Calculate daily changes for momentum indicators
    df['INR_change'] = df['INR'].pct_change() * 100
    df['DXY_change'] = df['DXY'].pct_change() * 100
    df['US10Y_change'] = df['US10Y'].diff()
    df['GOLD_change'] = df['GOLD'].pct_change() * 100
    df['OIL_change'] = df['OIL'].pct_change() * 100

    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def get_trading_days(df: pd.DataFrame, start: datetime, end: datetime) -> List[datetime]:
    """Get list of trading days within the simulation period."""
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    trading_days = df.loc[mask, 'Date'].tolist()
    return sorted([d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d for d in trading_days])


def get_historical_data(df: pd.DataFrame, target_date: datetime, lookback: int = 60) -> pd.DataFrame:
    """Get historical data up to (but not including) target_date."""
    mask = df['Date'] < target_date
    return df.loc[mask].tail(lookback).copy()


def get_market_context(df: pd.DataFrame, target_date: datetime) -> Optional[Dict]:
    """
    Get enriched market context for T-2 and T-1.
    Includes z-scores, momentum, and regime information.
    """
    available = df[df['Date'] < target_date].tail(60)  # Get 60 days for calculations

    if len(available) < 20:  # Need minimum data for meaningful stats
        return None

    t2_row = available.iloc[-2]
    t1_row = available.iloc[-1]

    def safe_float(val, default=0.0):
        try:
            return float(val) if pd.notna(val) else default
        except:
            return default

    # Calculate 20-day statistics for z-scores
    recent_20 = available.tail(20)

    context = {
        't2': {
            'date': t2_row['Date'].strftime('%Y-%m-%d') if hasattr(t2_row['Date'], 'strftime') else str(t2_row['Date'])[:10],
            'usdinr': safe_float(t2_row['INR']),
            'oil': safe_float(t2_row['OIL']),
            'gold': safe_float(t2_row['GOLD']),
            'us10y': safe_float(t2_row['US10Y']),
            'dxy': safe_float(t2_row['DXY']),
            'india_tone': safe_float(t2_row['IN_Avg_Tone']),
            'india_stability': safe_float(t2_row['IN_Avg_Stability']),
            'india_mentions': safe_float(t2_row['IN_Total_Mentions']),
            'india_panic': safe_float(t2_row['IN_Panic_Index']),
            'us_tone': safe_float(t2_row['US_Avg_Tone']),
            'us_stability': safe_float(t2_row['US_Avg_Stability']),
            'us_mentions': safe_float(t2_row['US_Total_Mentions']),
            'us_panic': safe_float(t2_row['US_Panic_Index']),
        },
        't1': {
            'date': t1_row['Date'].strftime('%Y-%m-%d') if hasattr(t1_row['Date'], 'strftime') else str(t1_row['Date'])[:10],
            'usdinr': safe_float(t1_row['INR']),
            'oil': safe_float(t1_row['OIL']),
            'gold': safe_float(t1_row['GOLD']),
            'us10y': safe_float(t1_row['US10Y']),
            'dxy': safe_float(t1_row['DXY']),
            'india_tone': safe_float(t1_row['IN_Avg_Tone']),
            'india_stability': safe_float(t1_row['IN_Avg_Stability']),
            'india_mentions': safe_float(t1_row['IN_Total_Mentions']),
            'india_panic': safe_float(t1_row['IN_Panic_Index']),
            'us_tone': safe_float(t1_row['US_Avg_Tone']),
            'us_stability': safe_float(t1_row['US_Avg_Stability']),
            'us_mentions': safe_float(t1_row['US_Total_Mentions']),
            'us_panic': safe_float(t1_row['US_Panic_Index']),
        },
        # Statistical context
        'stats': {
            'inr_20d_mean': safe_float(recent_20['INR'].mean()),
            'inr_20d_std': safe_float(recent_20['INR'].std()),
            'dxy_20d_mean': safe_float(recent_20['DXY'].mean()),
            'dxy_20d_std': safe_float(recent_20['DXY'].std()),
            'us10y_20d_mean': safe_float(recent_20['US10Y'].mean()),
            'us10y_20d_std': safe_float(recent_20['US10Y'].std()),
            'gold_20d_mean': safe_float(recent_20['GOLD'].mean()),
            'gold_20d_std': safe_float(recent_20['GOLD'].std()),
            'inr_5d_momentum': safe_float(available['INR_change'].tail(5).sum()),
            'dxy_5d_momentum': safe_float(available['DXY_change'].tail(5).sum()),
        },
        # Regime information
        'regime': {
            'volatility': RegimeDetector.detect_volatility_regime(
                available['INR_change'].dropna().tolist()
            ),
            'trend': RegimeDetector.detect_trend_regime(
                available['INR'].tolist()
            ),
        }
    }

    # Add z-scores
    context['z_scores'] = {
        'dxy': MomentumIndicators.calculate_z_score(
            context['t1']['dxy'],
            context['stats']['dxy_20d_mean'],
            context['stats']['dxy_20d_std']
        ),
        'us10y': MomentumIndicators.calculate_z_score(
            context['t1']['us10y'],
            context['stats']['us10y_20d_mean'],
            context['stats']['us10y_20d_std']
        ),
        'gold': MomentumIndicators.calculate_z_score(
            context['t1']['gold'],
            context['stats']['gold_20d_mean'],
            context['stats']['gold_20d_std']
        ),
    }

    return context


def get_actual_price(df: pd.DataFrame, target_date: datetime) -> Optional[float]:
    """Get actual USD/INR closing price for a given date."""
    mask = df['Date'].dt.date == target_date.date()
    matches = df.loc[mask, 'INR']

    if len(matches) > 0:
        return float(matches.iloc[0])
    return None


# ============================================================================
# PERSONA MANAGEMENT
# ============================================================================

def load_personas(filepath: str) -> List[Dict]:
    """Load personas from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['personas']


def format_llm_prompt(
    context: Dict,
    stat_prediction: float,
    stat_confidence: float,
    feature_contributions: Dict[str, float],
    historical_correlations: Dict[str, float]
) -> str:
    """
    Format prompt for LLM that asks for DIRECTIONAL ADJUSTMENT, not exact price.

    Key change from V1: We give the LLM the statistical model's prediction and
    ask it to provide a qualitative adjustment based on sentiment analysis.
    """
    t2 = context['t2']
    t1 = context['t1']
    stats = context['stats']
    z_scores = context['z_scores']
    regime = context['regime']

    # Format feature changes
    dxy_change = t1['dxy'] - t2['dxy']
    dxy_pct = (dxy_change / t2['dxy'] * 100) if t2['dxy'] else 0

    us10y_change = t1['us10y'] - t2['us10y']

    gold_change = t1['gold'] - t2['gold']
    gold_pct = (gold_change / t2['gold'] * 100) if t2['gold'] else 0

    oil_change = t1['oil'] - t2['oil']
    oil_pct = (oil_change / t2['oil'] * 100) if t2['oil'] else 0

    # Format contributions from statistical model
    contrib_text = "\n".join([
        f"  - {feat}: {contrib:+.4f} INR impact"
        for feat, contrib in sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    ])

    # Z-score interpretation
    def interpret_zscore(z: float, name: str) -> str:
        if abs(z) < 0.5:
            return f"{name} is near its 20-day average"
        elif z > 1.5:
            return f"{name} is SIGNIFICANTLY ABOVE average (+{z:.1f}σ) - potentially overbought"
        elif z > 0.5:
            return f"{name} is above average (+{z:.1f}σ)"
        elif z < -1.5:
            return f"{name} is SIGNIFICANTLY BELOW average ({z:.1f}σ) - potentially oversold"
        else:
            return f"{name} is below average ({z:.1f}σ)"

    prompt = f"""
## QUANTITATIVE FOREX ANALYSIS - USD/INR

**IMPORTANT: You are providing a QUALITATIVE ADJUSTMENT to a statistical model's prediction.
The statistical model handles numerical relationships. Your job is to assess sentiment
and news factors that the model might miss.**

---

### STATISTICAL MODEL BASELINE

The quantitative model predicts: **{stat_prediction:.4f}** (confidence: {stat_confidence:.1f}%)

**What's driving this prediction:**
{contrib_text}

**Historical correlations with USD/INR:**
- US10Y: +0.84 (STRONG - rising yields = weaker INR)
- GOLD: +0.80 (STRONG - rising gold = weaker INR)
- DXY: +0.62 (MODERATE - stronger dollar = weaker INR)
- IN_Avg_Tone: +0.44 (MODERATE - better India sentiment = stronger INR)

---

### MARKET DATA (Past 2 Days)

**Dollar Index (DXY):**
- Day T-2: {t2['dxy']:.2f} → Day T-1: {t1['dxy']:.2f} ({dxy_pct:+.2f}%)
- {interpret_zscore(z_scores['dxy'], 'DXY')}

**US 10-Year Treasury:**
- Day T-2: {t2['us10y']:.3f}% → Day T-1: {t1['us10y']:.3f}% ({us10y_change:+.3f}%)
- {interpret_zscore(z_scores['us10y'], 'US10Y')}

**Gold (USD/oz):**
- Day T-2: ${t2['gold']:.2f} → Day T-1: ${t1['gold']:.2f} ({gold_pct:+.2f}%)
- {interpret_zscore(z_scores['gold'], 'Gold')}

**Crude Oil (USD/barrel):**
- Day T-2: ${t2['oil']:.2f} → Day T-1: ${t1['oil']:.2f} ({oil_pct:+.2f}%)

---

### NEWS SENTIMENT ANALYSIS (GDELT)

**India News:**
| Metric | T-2 | T-1 | Change | Interpretation |
|--------|-----|-----|--------|----------------|
| Avg Tone | {t2['india_tone']:.2f} | {t1['india_tone']:.2f} | {t1['india_tone']-t2['india_tone']:+.2f} | {'Improving' if t1['india_tone'] > t2['india_tone'] else 'Worsening'} sentiment |
| Mentions | {int(t2['india_mentions']):,} | {int(t1['india_mentions']):,} | {((t1['india_mentions']-t2['india_mentions'])/t2['india_mentions']*100) if t2['india_mentions'] else 0:+.1f}% | {'Spike!' if abs(t1['india_mentions']-t2['india_mentions'])/t2['india_mentions'] > 0.2 else 'Normal'} |

**US News:**
| Metric | T-2 | T-1 | Change | Interpretation |
|--------|-----|-----|--------|----------------|
| Avg Tone | {t2['us_tone']:.2f} | {t1['us_tone']:.2f} | {t1['us_tone']-t2['us_tone']:+.2f} | {'Improving' if t1['us_tone'] > t2['us_tone'] else 'Worsening'} sentiment |
| Mentions | {int(t2['us_mentions']):,} | {int(t1['us_mentions']):,} | {((t1['us_mentions']-t2['us_mentions'])/t2['us_mentions']*100) if t2['us_mentions'] else 0:+.1f}% | {'Spike!' if abs(t1['us_mentions']-t2['us_mentions'])/t2['us_mentions'] > 0.2 else 'Normal'} |

---

### MARKET REGIME

- **Volatility Regime:** {regime['volatility'].upper()}
- **Trend Regime:** {regime['trend'].replace('_', ' ').upper()}
- **5-Day INR Momentum:** {stats['inr_5d_momentum']:+.2f}%

---

## YOUR TASK

Based on the SENTIMENT and NEWS factors above, should the statistical model's prediction be adjusted?

**Think about:**
1. Is there sentiment information suggesting INR will be stronger or weaker than the model predicts?
2. Are there news spikes that indicate market-moving events?
3. Does the market regime suggest the model might over/under-estimate?

**Respond with ONLY this JSON (no other text):**
```json
{{
    "direction": "bullish_usd" | "bearish_usd" | "neutral",
    "adjustment_magnitude": "none" | "small" | "medium" | "large",
    "confidence": <1-10>,
    "reasoning": "<2-3 sentences explaining your sentiment assessment>"
}}
```

**Adjustment guide:**
- "none": Statistical model is probably right, no adjustment needed
- "small": Minor sentiment factor, adjust by ~0.05-0.1%
- "medium": Notable sentiment factor, adjust by ~0.1-0.2%
- "large": Strong sentiment signal, adjust by ~0.2-0.3%

Note: "bullish_usd" means USD strengthens (INR weakens, rate goes UP)
"""
    return prompt


# ============================================================================
# GEMINI API INTERACTION
# ============================================================================

def initialize_gemini(api_key: str) -> genai.GenerativeModel:
    """Initialize Gemini model."""
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        generation_config={
            'temperature': 0.5,  # Lower temperature for more consistent outputs
            'top_p': 0.9,
            'max_output_tokens': 512,
        }
    )

    return model


def query_persona(model: genai.GenerativeModel, persona: Dict, market_prompt: str, retries: int = MAX_RETRIES) -> Dict:
    """Query Gemini with a specific persona."""

    system_prompt = persona['system_prompt']
    full_prompt = f"{system_prompt}\n\n{market_prompt}"

    for attempt in range(retries):
        try:
            response = model.generate_content(full_prompt)

            # Extract text from response
            response_text = ""
            try:
                response_text = response.text
            except (ValueError, AttributeError):
                pass

            if not response_text and hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text

            if not response_text and hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text

            if not response_text.strip():
                continue

            parsed = parse_adjustment_response(response_text)

            if parsed:
                return {
                    'success': True,
                    'persona_id': persona['id'],
                    'persona_name': persona['name'],
                    'short_name': persona.get('short_name', persona['name'][:15]),
                    'weight': persona['weight'],
                    'raw_response': response_text,
                    **parsed
                }
            else:
                print(f"    Failed to parse response for {persona.get('short_name', persona['name'][:15])}, attempt {attempt + 1}")

        except Exception as e:
            print(f"    API error for {persona.get('short_name', persona['name'][:15])}: {e}, attempt {attempt + 1}")
            time.sleep(API_DELAY_SECONDS * 2)

    # Return neutral adjustment on failure
    return {
        'success': False,
        'persona_id': persona['id'],
        'persona_name': persona['name'],
        'short_name': persona.get('short_name', persona['name'][:15]),
        'weight': persona['weight'],
        'direction': 'neutral',
        'adjustment_magnitude': 'none',
        'confidence': 5,
        'reasoning': 'Failed to get response from API',
        'raw_response': None
    }


def parse_adjustment_response(response_text: str) -> Optional[Dict]:
    """Parse the directional adjustment response from LLM."""
    try:
        cleaned = response_text.strip()

        if '```json' in cleaned:
            cleaned = cleaned.split('```json')[1]
        if '```' in cleaned:
            cleaned = cleaned.split('```')[0]
        cleaned = cleaned.strip()

        start_idx = cleaned.find('{')
        if start_idx == -1:
            return None

        brace_count = 0
        end_idx = -1
        for i, char in enumerate(cleaned[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if end_idx == -1:
            return None

        json_str = cleaned[start_idx:end_idx]
        parsed = json.loads(json_str)

        # Validate required fields
        if 'direction' in parsed:
            return {
                'direction': parsed.get('direction', 'neutral'),
                'adjustment_magnitude': parsed.get('adjustment_magnitude', 'none'),
                'confidence': int(parsed.get('confidence', 5)),
                'reasoning': parsed.get('reasoning', '')
            }

    except (json.JSONDecodeError, ValueError) as e:
        print(f"      Parse error: {e}")

    return None


# ============================================================================
# ENSEMBLE AND AGGREGATION
# ============================================================================

def convert_llm_adjustment_to_pct(direction: str, magnitude: str, confidence: int) -> float:
    """
    Convert LLM's qualitative adjustment to a percentage change.

    Bounded by MAX_LLM_ADJUSTMENT_PCT to prevent wild predictions.
    """
    # Base adjustment by magnitude
    magnitude_map = {
        'none': 0.0,
        'small': 0.08,
        'medium': 0.15,
        'large': 0.25,
    }

    base_adj = magnitude_map.get(magnitude, 0.0)

    # Scale by confidence (5 is neutral, below = less confident, above = more confident)
    confidence_scale = (confidence - 5) / 10 + 1  # 0.5 to 1.5
    base_adj *= confidence_scale

    # Apply direction
    if direction == 'bullish_usd':
        adj = base_adj  # USD strengthens = INR weakens = rate goes UP
    elif direction == 'bearish_usd':
        adj = -base_adj  # USD weakens = INR strengthens = rate goes DOWN
    else:
        adj = 0.0

    # Clamp to bounds
    return max(-MAX_LLM_ADJUSTMENT_PCT, min(MAX_LLM_ADJUSTMENT_PCT, adj))


def aggregate_persona_adjustments(predictions: List[Dict]) -> Dict:
    """
    Aggregate adjustments from all personas using weighted voting.
    """
    valid_predictions = [p for p in predictions if p.get('success')]

    if not valid_predictions:
        return {
            'weighted_adjustment_pct': 0.0,
            'consensus_direction': 'neutral',
            'avg_confidence': 5,
            'num_valid': 0
        }

    total_weight = sum(p['weight'] for p in valid_predictions)

    # Calculate weighted adjustment
    weighted_adj = sum(
        convert_llm_adjustment_to_pct(p['direction'], p['adjustment_magnitude'], p['confidence'])
        * (p['weight'] / total_weight)
        for p in valid_predictions
    )

    # Determine consensus direction
    bullish_weight = sum(p['weight'] for p in valid_predictions if p['direction'] == 'bullish_usd')
    bearish_weight = sum(p['weight'] for p in valid_predictions if p['direction'] == 'bearish_usd')

    if bullish_weight > bearish_weight + 0.1 * total_weight:
        consensus = 'bullish_usd'
    elif bearish_weight > bullish_weight + 0.1 * total_weight:
        consensus = 'bearish_usd'
    else:
        consensus = 'neutral'

    avg_confidence = sum(p['confidence'] * p['weight'] for p in valid_predictions) / total_weight

    return {
        'weighted_adjustment_pct': weighted_adj,
        'consensus_direction': consensus,
        'avg_confidence': avg_confidence,
        'num_valid': len(valid_predictions)
    }


def ensemble_prediction(
    stat_prediction: float,
    stat_confidence: float,
    llm_adjustment_pct: float,
    stat_weight: float = INITIAL_STAT_WEIGHT
) -> float:
    """
    Combine statistical prediction with LLM adjustment.

    The statistical model provides the base prediction.
    The LLM provides a percentage adjustment based on sentiment.
    """
    # LLM adjustment is applied as a percentage change to the statistical prediction
    llm_weight = 1 - stat_weight

    # Apply LLM adjustment
    adjusted_prediction = stat_prediction * (1 + llm_adjustment_pct / 100 * llm_weight / stat_weight)

    return adjusted_prediction


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(api_key: str, personas_file: str = PERSONAS_FILE):
    """Run the full market simulation with hybrid statistical + LLM approach."""

    print("=" * 70)
    print("GEMINI USD/INR FOREX SIMULATION V2 - HYBRID STATISTICAL + LLM")
    print("=" * 70)
    print(f"Period: {SIMULATION_START.date()} to {SIMULATION_END.date()}")
    print(f"Warmup: First {WARMUP_DAYS} days for statistical model calibration")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load personas
    personas_path = os.path.join(script_dir, personas_file)
    personas = load_personas(personas_path)
    print(f"Loaded {len(personas)} personas")

    # Load market data
    data_path = os.path.join(script_dir, DATA_FILE)
    market_df = load_market_data(data_path)

    # Initialize Gemini
    print("\nInitializing Gemini 2.0 Flash...")
    model = initialize_gemini(api_key)

    # Initialize statistical model
    stat_model = StatisticalModel(lookback_window=REGRESSION_WINDOW)

    # Get trading days
    trading_days = get_trading_days(market_df, SIMULATION_START, SIMULATION_END)
    print(f"\nSimulating {len(trading_days)} trading days")
    print()

    # Historical correlations (from our analysis)
    historical_correlations = {
        'US10Y': 0.84,
        'GOLD': 0.80,
        'DXY': 0.62,
        'IN_Avg_Tone': 0.44,
        'OIL': 0.29
    }

    all_results = []

    # Performance tracking for adaptive weights
    stat_errors = []
    llm_errors = []

    for day_idx, target_date in enumerate(trading_days):
        is_warmup = day_idx < WARMUP_DAYS

        print(f"\n[Day {day_idx + 1}/{len(trading_days)}] {target_date.date()}")
        if is_warmup:
            print("  MODE: WARMUP (calibrating statistical model)")
        else:
            print("  MODE: HYBRID PREDICTION (statistical + LLM)")
        print("-" * 50)

        # Get market context
        context = get_market_context(market_df, target_date)

        if not context:
            print("  Skipping - insufficient historical data")
            continue

        # Get historical data for statistical model
        hist_data = get_historical_data(market_df, target_date, REGRESSION_WINDOW)

        # Fit/update statistical model
        if len(hist_data) >= 30:
            stat_model.fit(hist_data)

        # Get previous day's close
        last_close = context['t1']['usdinr']

        # Get actual price (for evaluation)
        actual_price = get_actual_price(market_df, target_date)

        if is_warmup:
            # During warmup, just use statistical model
            t1_features = {
                'US10Y': context['t1']['us10y'],
                'GOLD': context['t1']['gold'],
                'DXY': context['t1']['dxy'],
                'IN_Avg_Tone': context['t1']['india_tone'],
                'OIL': context['t1']['oil']
            }

            stat_pred, stat_std = stat_model.predict(t1_features)

            if stat_pred is None:
                stat_pred = last_close
                stat_std = 0.1

            final_prediction = stat_pred
            stat_confidence = max(0, min(100, 100 - stat_std / last_close * 1000))

            print(f"  Statistical Prediction: {stat_pred:.4f} (R²: {stat_model.r_squared:.3f})")
            print(f"  Final Prediction: {final_prediction:.4f}")

            day_result = {
                'date': target_date.isoformat(),
                'day_idx': day_idx,
                'mode': 'warmup',
                'last_close': last_close,
                'stat_prediction': stat_pred,
                'stat_r_squared': stat_model.r_squared,
                'llm_adjustment_pct': 0.0,
                'final_prediction': final_prediction,
                'actual_price': actual_price,
                'prediction_error_pct': ((final_prediction - actual_price) / actual_price * 100) if actual_price else None,
                'persona_predictions': []
            }

        else:
            # Full hybrid mode
            t1_features = {
                'US10Y': context['t1']['us10y'],
                'GOLD': context['t1']['gold'],
                'DXY': context['t1']['dxy'],
                'IN_Avg_Tone': context['t1']['india_tone'],
                'OIL': context['t1']['oil']
            }

            stat_pred, stat_std = stat_model.predict(t1_features)

            if stat_pred is None:
                stat_pred = last_close
                stat_std = 0.1

            stat_confidence = max(0, min(100, 100 - stat_std / last_close * 1000))

            # Get feature contributions for LLM context
            contributions = stat_model.get_feature_contributions(t1_features)

            print(f"  Statistical Model: {stat_pred:.4f} (R²: {stat_model.r_squared:.3f})")

            # Format prompt for LLM
            market_prompt = format_llm_prompt(
                context,
                stat_pred,
                stat_confidence,
                contributions,
                historical_correlations
            )

            # Query each persona
            day_predictions = []

            for persona in personas:
                print(f"  Querying {persona.get('short_name', persona['name'][:15])}...", end=" ")

                prediction = query_persona(model, persona, market_prompt)
                day_predictions.append(prediction)

                if prediction['success']:
                    adj = convert_llm_adjustment_to_pct(
                        prediction['direction'],
                        prediction['adjustment_magnitude'],
                        prediction['confidence']
                    )
                    print(f"{prediction['direction']} ({prediction['adjustment_magnitude']}) -> {adj:+.2f}%")
                else:
                    print("FAILED")

                time.sleep(API_DELAY_SECONDS)

            # Aggregate LLM adjustments
            llm_result = aggregate_persona_adjustments(day_predictions)

            print(f"\n  LLM Consensus: {llm_result['consensus_direction']} (adj: {llm_result['weighted_adjustment_pct']:+.3f}%)")

            # Ensemble prediction
            final_prediction = ensemble_prediction(
                stat_pred,
                stat_confidence,
                llm_result['weighted_adjustment_pct']
            )

            print(f"  Final Ensemble: {final_prediction:.4f}")

            day_result = {
                'date': target_date.isoformat(),
                'day_idx': day_idx,
                'mode': 'hybrid',
                'last_close': last_close,
                'stat_prediction': stat_pred,
                'stat_r_squared': stat_model.r_squared,
                'stat_confidence': stat_confidence,
                'llm_adjustment_pct': llm_result['weighted_adjustment_pct'],
                'llm_consensus': llm_result['consensus_direction'],
                'llm_avg_confidence': llm_result['avg_confidence'],
                'final_prediction': final_prediction,
                'actual_price': actual_price,
                'prediction_error_pct': ((final_prediction - actual_price) / actual_price * 100) if actual_price else None,
                'stat_error_pct': ((stat_pred - actual_price) / actual_price * 100) if actual_price else None,
                'persona_predictions': day_predictions
            }

            # Track errors for adaptive weighting
            if actual_price:
                stat_errors.append(abs((stat_pred - actual_price) / actual_price * 100))
                llm_errors.append(abs((final_prediction - actual_price) / actual_price * 100))

        # Show actual vs predicted
        if actual_price:
            print(f"  Actual: {actual_price:.4f}")
            if day_result.get('prediction_error_pct') is not None:
                print(f"  Error: {day_result['prediction_error_pct']:+.3f}%")

        all_results.append(day_result)

        # Save intermediate results
        save_results(all_results, script_dir)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    generate_summary(all_results, script_dir)

    return all_results


def save_results(results: List[Dict], output_dir: str):
    """Save results to JSON file."""
    output_path = os.path.join(output_dir, RESULTS_FILE)

    # Make a copy without raw_response to keep file size manageable
    clean_results = []
    for r in results:
        clean_r = {k: v for k, v in r.items() if k != 'persona_predictions'}
        if 'persona_predictions' in r:
            clean_r['persona_predictions'] = [
                {k: v for k, v in p.items() if k != 'raw_response'}
                for p in r['persona_predictions']
            ]
        clean_results.append(clean_r)

    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)


def generate_summary(results: List[Dict], output_dir: str):
    """Generate summary statistics and CSV."""

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
        })

    df = pd.DataFrame(summary_data)

    csv_path = os.path.join(output_dir, SUMMARY_CSV)
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    # Print statistics
    hybrid_results = df[df['mode'] == 'hybrid']

    if not hybrid_results.empty:
        valid = hybrid_results[hybrid_results['prediction_error_pct'].notna()]

        if not valid.empty:
            print("\n" + "=" * 50)
            print("HYBRID MODEL PERFORMANCE (Post-Warmup)")
            print("=" * 50)

            print(f"\nFinal Ensemble:")
            print(f"  Mean Absolute Error:  {valid['prediction_error_pct'].abs().mean():.4f}%")
            print(f"  Mean Error (Bias):    {valid['prediction_error_pct'].mean():+.4f}%")
            print(f"  Std Dev of Error:     {valid['prediction_error_pct'].std():.4f}%")
            print(f"  Max Error:            {valid['prediction_error_pct'].abs().max():.4f}%")

            if 'stat_error_pct' in valid.columns:
                stat_valid = valid[valid['stat_error_pct'].notna()]
                if not stat_valid.empty:
                    print(f"\nStatistical Model Only:")
                    print(f"  Mean Absolute Error:  {stat_valid['stat_error_pct'].abs().mean():.4f}%")
                    print(f"  Mean Error (Bias):    {stat_valid['stat_error_pct'].mean():+.4f}%")

            # Direction accuracy
            correct_direction = sum(
                1 for _, row in valid.iterrows()
                if (row['final_prediction'] > row['last_close'] and row['actual_price'] > row['last_close']) or
                   (row['final_prediction'] < row['last_close'] and row['actual_price'] < row['last_close']) or
                   (abs(row['final_prediction'] - row['last_close']) < 0.01 and abs(row['actual_price'] - row['last_close']) < 0.01)
            )
            print(f"\nDirection Accuracy: {correct_direction}/{len(valid)} ({100*correct_direction/len(valid):.1f}%)")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set it using: set GEMINI_API_KEY=your_api_key_here")
        exit(1)

    try:
        results = run_simulation(api_key)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()

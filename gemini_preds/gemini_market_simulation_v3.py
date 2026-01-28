"""
Gemini USD/INR Forex Simulation V3 - Fixed LLM Integration

Key fixes from V2 analysis:
1. Personas now SEE the statistical baseline prediction
2. Adversarial persona design (50/50 bull/bear split)
3. Contrarian instructions to prevent herd behavior
4. Dynamic ensemble weights based on consensus strength
5. Regime-specific persona weight adjustments
6. Explicit bias acknowledgment in prompts

V2 Problems Solved:
- 66.7% bullish bias → Now 50/50 adversarial design
- LLM degraded by 0.8% → Dynamic weighting reduces impact when uncertain
- Low confidence variance → Better prompts encourage conviction
- Personas blind to stats → Now see statistical baseline
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
PERSONAS_FILE = "forex_personas_v3.json"
RESULTS_FILE = "simulation_results_v3.json"
SUMMARY_CSV = "simulation_summary_v3.csv"

DATA_FILE = "../Super_Master_Dataset.csv"

WARMUP_DAYS = 20
REGRESSION_WINDOW = 60

# Dynamic weight bounds
MIN_STAT_WEIGHT = 0.65  # Statistical model always gets at least 65%
MAX_STAT_WEIGHT = 0.90  # Cap at 90% so LLM always has some influence
DEFAULT_STAT_WEIGHT = 0.75  # Higher default - LLM must earn trust

# LLM adjustment bounds (tighter than V2)
MAX_LLM_ADJUSTMENT_PCT = 0.35  # Reduced from 0.5% - less room for damage

API_DELAY_SECONDS = 6  # Increased significantly for rate limiting
MAX_RETRIES = 3  # Back to 3, with longer delays
INITIAL_DELAY = 2  # Delay before first request of each day

# ============================================================================
# STATISTICAL MODEL (Same as V2)
# ============================================================================

class StatisticalModel:
    """OLS regression model for USD/INR prediction."""

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
        self.features = ['US10Y', 'GOLD', 'DXY', 'IN_Avg_Tone', 'OIL']

    def fit(self, df: pd.DataFrame) -> bool:
        if len(df) < self.lookback_window:
            return False

        data = df.tail(self.lookback_window).copy()
        self.inr_mean = data['INR'].mean()
        self.inr_std = data['INR'].std()

        for feat in self.features:
            self.feature_means[feat] = data[feat].mean()
            self.feature_stds[feat] = data[feat].std()

        X = np.column_stack([
            (data[feat] - self.feature_means[feat]) / (self.feature_stds[feat] + 1e-8)
            for feat in self.features
        ])
        X = np.column_stack([np.ones(len(X)), X])
        y = (data['INR'] - self.inr_mean) / (self.inr_std + 1e-8)

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            self.betas = XtX_inv @ X.T @ y
            self.alpha = self.betas[0]
            self.betas = self.betas[1:]

            y_pred = X @ np.concatenate([[self.alpha], self.betas])
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.residual_std = np.sqrt(ss_res / (len(y) - len(self.features) - 1))
            return True
        except np.linalg.LinAlgError:
            return False

    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        if self.betas is None:
            return None, None

        x = np.array([
            (features.get(feat, self.feature_means[feat]) - self.feature_means[feat])
            / (self.feature_stds[feat] + 1e-8)
            for feat in self.features
        ])

        z_pred = self.alpha + np.dot(self.betas, x)
        inr_pred = z_pred * self.inr_std + self.inr_mean
        inr_std = self.residual_std * self.inr_std

        return inr_pred, inr_std

    def get_feature_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        if self.betas is None:
            return {}

        contributions = {}
        for i, feat in enumerate(self.features):
            z_val = (features.get(feat, self.feature_means[feat]) - self.feature_means[feat]) / (self.feature_stds[feat] + 1e-8)
            contributions[feat] = self.betas[i] * z_val * self.inr_std

        return contributions


class RegimeDetector:
    """Detect market regime for adaptive persona weighting."""

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
            'dxy_20d_mean': safe_float(recent_20['DXY'].mean()),
            'us10y_20d_mean': safe_float(recent_20['US10Y'].mean()),
            'inr_5d_momentum': safe_float(available['INR_change'].tail(5).sum()),
        },
        'regime': {
            'volatility': RegimeDetector.detect_volatility_regime(
                available['INR_change'].dropna().tolist()
            ),
            'trend': RegimeDetector.detect_trend_regime(
                available['INR'].tolist()
            ),
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


def adjust_persona_weights_for_regime(personas: List[Dict], regime: Dict) -> List[Dict]:
    """
    Dynamically adjust persona weights based on market regime.

    - High volatility: Increase weight of defensive/institutional personas
    - Trending: Increase weight of momentum personas
    - Mean reverting: Increase weight of mean-reversion personas
    """
    adjusted = []

    for p in personas:
        new_weight = p['weight']
        bias = p.get('bias', 'neutral')
        short_name = p.get('short_name', '')

        # Volatility adjustments
        if regime['volatility'] == 'high':
            # In high vol, reduce speculative personas, increase institutional
            if 'Speculator' in short_name or 'Retail' in short_name:
                new_weight *= 0.6
            elif 'RBI' in short_name or 'Importer' in short_name:
                new_weight *= 1.3

        elif regime['volatility'] == 'low':
            # In low vol, technical/algo traders are more relevant
            if 'Tech' in short_name or 'Algo' in short_name:
                new_weight *= 1.2

        # Trend adjustments
        if regime['trend'] == 'trending_up':
            # Trending up means INR weakening - bullish USD personas more relevant
            if bias == 'bullish_usd':
                new_weight *= 1.15
            elif bias == 'bearish_usd':
                new_weight *= 0.9

        elif regime['trend'] == 'trending_down':
            # Trending down means INR strengthening - bearish USD personas more relevant
            if bias == 'bearish_usd':
                new_weight *= 1.15
            elif bias == 'bullish_usd':
                new_weight *= 0.9

        adjusted.append({**p, 'adjusted_weight': new_weight})

    # Normalize weights to sum to 1
    total_weight = sum(p['adjusted_weight'] for p in adjusted)
    for p in adjusted:
        p['adjusted_weight'] = p['adjusted_weight'] / total_weight

    return adjusted


def format_persona_prompt(
    persona: Dict,
    context: Dict,
    stat_prediction: float,
    stat_implied_change_pct: float,
    feature_contributions: Dict[str, float],
    current_consensus: str = None
) -> str:
    """
    Format prompt that:
    1. Shows the statistical baseline
    2. Asks for adjustment relative to baseline
    3. Includes contrarian instruction
    4. Acknowledges persona's bias
    """
    t1 = context['t1']
    t2 = context['t2']
    regime = context['regime']

    # Format key indicators
    dxy_change = t1['dxy'] - t2['dxy']
    dxy_pct = (dxy_change / t2['dxy'] * 100) if t2['dxy'] else 0
    us10y_change = t1['us10y'] - t2['us10y']
    gold_change = t1['gold'] - t2['gold']
    gold_pct = (gold_change / t2['gold'] * 100) if t2['gold'] else 0
    oil_change = t1['oil'] - t2['oil']
    oil_pct = (oil_change / t2['oil'] * 100) if t2['oil'] else 0

    # Feature contribution summary
    top_drivers = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    driver_text = ", ".join([f"{k}: {v:+.3f}" for k, v in top_drivers])

    # Contrarian instruction based on current consensus
    contrarian_text = ""
    if current_consensus:
        contrarian_text = f"""
**CONTRARIAN CHECK**: Current consensus among other personas is {current_consensus.upper()}.
As a {persona.get('bias_description', 'market participant')}, CHALLENGE this consensus if your analysis disagrees.
Don't follow the herd - think independently based on YOUR perspective."""

    # Bias acknowledgment
    persona_bias = persona.get('bias', 'neutral')
    if persona_bias == 'bullish_usd':
        bias_text = "Your natural bias is BULLISH on USD/INR (expecting rupee weakness)."
    elif persona_bias == 'bearish_usd':
        bias_text = "Your natural bias is BEARISH on USD/INR (expecting rupee strength)."
    else:
        bias_text = "You have NO directional bias - follow the data objectively."

    prompt = f"""
## FOREX ANALYSIS: USD/INR

{bias_text}
{contrarian_text}

---

### STATISTICAL MODEL BASELINE

**The quantitative model predicts: {stat_prediction:.4f}**
- This implies a {stat_implied_change_pct:+.3f}% move from yesterday's close
- Key drivers: {driver_text}
- Model R-squared: ~0.56 (explains 56% of variance)

**Your job**: Should sentiment/news push the rate HIGHER or LOWER than {stat_prediction:.4f}?

---

### MARKET DATA (Past 2 Days)

| Indicator | T-2 | T-1 | Change |
|-----------|-----|-----|--------|
| DXY | {t2['dxy']:.2f} | {t1['dxy']:.2f} | {dxy_pct:+.2f}% |
| US 10Y Yield | {t2['us10y']:.3f}% | {t1['us10y']:.3f}% | {us10y_change:+.3f}% |
| Gold | ${t2['gold']:.0f} | ${t1['gold']:.0f} | {gold_pct:+.2f}% |
| Oil | ${t2['oil']:.1f} | ${t1['oil']:.1f} | {oil_pct:+.2f}% |

**Historical Correlations with USD/INR:**
- US10Y: +0.84 (rising yields → weaker INR)
- GOLD: +0.80 (rising gold → weaker INR)
- DXY: +0.62 (stronger dollar → weaker INR)

---

### SENTIMENT DATA

| Region | Avg Tone | Mentions |
|--------|----------|----------|
| India | {t1['india_tone']:.2f} (prev: {t2['india_tone']:.2f}) | {int(t1['india_mentions']):,} |
| US | {t1['us_tone']:.2f} (prev: {t2['us_tone']:.2f}) | {int(t1['us_mentions']):,} |

*Tone: -10 (very negative) to +10 (very positive)*

---

### MARKET REGIME

- Volatility: **{regime['volatility'].upper()}**
- Trend: **{regime['trend'].replace('_', ' ').upper()}**

---

## YOUR RESPONSE

Given the statistical baseline of **{stat_prediction:.4f}**, should USD/INR be:
- **HIGHER** (bullish USD / bearish INR)?
- **LOWER** (bearish USD / bullish INR)?
- **UNCHANGED** (trust the model)?

Respond with ONLY this JSON:
```json
{{
    "direction": "bullish_usd" | "bearish_usd" | "neutral",
    "adjustment_magnitude": "none" | "small" | "medium" | "large",
    "confidence": <1-10>,
    "reasoning": "<2-3 sentences from your persona's perspective>"
}}
```

**Magnitude guide** (adjustment to statistical prediction):
- none: Trust the model (0%)
- small: Minor sentiment factor (~0.05-0.08%)
- medium: Notable factor (~0.10-0.15%)
- large: Strong signal (~0.20-0.30%)

**Confidence guide**:
- 1-3: Low conviction, uncertain
- 4-6: Moderate conviction
- 7-10: High conviction, strong signal
"""
    return prompt


# ============================================================================
# GEMINI API
# ============================================================================

def initialize_gemini(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    # Use gemini-2.0-flash for reliable JSON responses
    # gemini-2.5-flash is a "thinking" model with different output format
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        generation_config={
            'temperature': 0.6,
            'top_p': 0.9,
            'max_output_tokens': 1024,
        }
    )
    return model


def extract_response_text(response) -> str:
    """Extract text from Gemini response, handling all formats including thinking mode."""
    response_text = ""

    # Method 1: Simple .text accessor
    try:
        if response.text:
            return response.text
    except (ValueError, AttributeError):
        pass

    # Method 2: Extract from candidates (Gemini 2.5 format)
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    # Skip thinking parts
                    if hasattr(part, 'thought') and part.thought:
                        continue
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text

    # Method 3: Direct parts access
    if not response_text and hasattr(response, 'parts'):
        for part in response.parts:
            if hasattr(part, 'thought') and part.thought:
                continue
            if hasattr(part, 'text') and part.text:
                response_text += part.text

    # Method 4: Try to get ANY text, even from thinking parts (last resort)
    if not response_text and hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        # Look for JSON in any part
                        if '{' in part.text and 'direction' in part.text:
                            response_text = part.text
                            break

    return response_text


def query_persona(model: genai.GenerativeModel, persona: Dict, market_prompt: str, retries: int = MAX_RETRIES) -> Dict:
    system_prompt = persona['system_prompt']
    full_prompt = f"{system_prompt}\n\n{market_prompt}"

    for attempt in range(retries):
        try:
            response = model.generate_content(full_prompt)
            response_text = extract_response_text(response)

            if not response_text.strip():
                # Exponential backoff for empty responses (likely rate limit)
                backoff = API_DELAY_SECONDS * (2 ** attempt)
                if attempt == retries - 1:
                    if hasattr(response, 'candidates') and response.candidates:
                        cand = response.candidates[0]
                        reason = getattr(cand, 'finish_reason', 'UNKNOWN')
                        print(f"(empty:{reason})", end=" ")
                time.sleep(backoff)
                continue

            parsed = parse_response(response_text)

            if parsed:
                return {
                    'success': True,
                    'persona_id': persona['id'],
                    'persona_name': persona['name'],
                    'short_name': persona.get('short_name', persona['name'][:15]),
                    'weight': persona.get('adjusted_weight', persona['weight']),
                    'original_weight': persona['weight'],
                    'bias': persona.get('bias', 'neutral'),
                    'raw_response': response_text,
                    **parsed
                }

            # Parse failed - already have text but can't parse
            if attempt == retries - 1:
                snippet = response_text[:150].replace('\n', ' ')
                print(f"\n    Parse failed. Response: {snippet}...")
            time.sleep(API_DELAY_SECONDS)

        except Exception as e:
            error_str = str(e).lower()
            # Detect rate limiting
            if 'rate' in error_str or 'quota' in error_str or '429' in error_str:
                backoff = API_DELAY_SECONDS * (3 ** attempt)  # Aggressive backoff for rate limits
                if attempt == retries - 1:
                    print(f"(rate-limit)", end=" ")
                time.sleep(backoff)
            else:
                if attempt == retries - 1:
                    print(f"(err:{str(e)[:30]})", end=" ")
                time.sleep(API_DELAY_SECONDS * 2)

    # All retries failed - return failure with neutral stance
    return {
        'success': False,
        'persona_id': persona['id'],
        'persona_name': persona['name'],
        'short_name': persona.get('short_name', persona['name'][:15]),
        'weight': persona.get('adjusted_weight', persona['weight']),
        'original_weight': persona['weight'],
        'bias': persona.get('bias', 'neutral'),
        'direction': 'neutral',
        'adjustment_magnitude': 'none',
        'confidence': 5,
        'reasoning': 'API failure after retries',
        'raw_response': None
    }


def normalize_parsed(parsed: Dict) -> Dict:
    """Normalize parsed JSON to expected format."""
    # Handle various direction formats
    direction = parsed.get('direction', 'neutral').lower().strip()
    if direction in ['bullish', 'bullish_usd', 'bull', 'up']:
        direction = 'bullish_usd'
    elif direction in ['bearish', 'bearish_usd', 'bear', 'down']:
        direction = 'bearish_usd'
    else:
        direction = 'neutral'

    # Handle magnitude variations
    magnitude = parsed.get('adjustment_magnitude', 'none').lower().strip()
    if magnitude not in ['none', 'small', 'medium', 'large']:
        magnitude = 'small' if magnitude in ['minor', 'slight', 'low'] else 'none'

    return {
        'direction': direction,
        'adjustment_magnitude': magnitude,
        'confidence': int(parsed.get('confidence', 5)),
        'reasoning': parsed.get('reasoning', '')
    }


def parse_response(response_text: str) -> Optional[Dict]:
    """
    Robust JSON parser that handles various response formats.
    """
    try:
        cleaned = response_text.strip()

        # Try multiple extraction methods

        # Method 1: Extract from markdown code block
        if '```json' in cleaned:
            parts = cleaned.split('```json')
            if len(parts) > 1:
                json_part = parts[1].split('```')[0].strip()
                try:
                    parsed = json.loads(json_part)
                    if 'direction' in parsed:
                        return normalize_parsed(parsed)
                except json.JSONDecodeError:
                    pass

        # Method 2: Extract from generic code block
        if '```' in cleaned:
            for block in cleaned.split('```')[1::2]:  # Get odd indices (inside blocks)
                block = block.strip()
                if block.startswith('json'):
                    block = block[4:].strip()
                if block.startswith('{'):
                    try:
                        parsed = json.loads(block)
                        if 'direction' in parsed:
                            return normalize_parsed(parsed)
                    except json.JSONDecodeError:
                        pass

        # Method 3: Find JSON by brace matching
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

        if 'direction' in parsed:
            return normalize_parsed(parsed)

    except (json.JSONDecodeError, ValueError) as e:
        # Silently fail, let caller handle
        pass

    return None


# ============================================================================
# ENSEMBLE AND AGGREGATION
# ============================================================================

def convert_adjustment_to_pct(direction: str, magnitude: str, confidence: int) -> float:
    """Convert qualitative adjustment to percentage. Tighter bounds than V2."""
    magnitude_map = {
        'none': 0.0,
        'small': 0.06,
        'medium': 0.12,
        'large': 0.22,
    }

    base_adj = magnitude_map.get(magnitude, 0.0)

    # Confidence scaling: 5 is neutral
    # Below 5 = dampen, above 5 = amplify
    if confidence <= 3:
        conf_scale = 0.5
    elif confidence <= 5:
        conf_scale = 0.8
    elif confidence <= 7:
        conf_scale = 1.0
    else:
        conf_scale = 1.2

    base_adj *= conf_scale

    if direction == 'bullish_usd':
        adj = base_adj
    elif direction == 'bearish_usd':
        adj = -base_adj
    else:
        adj = 0.0

    return max(-MAX_LLM_ADJUSTMENT_PCT, min(MAX_LLM_ADJUSTMENT_PCT, adj))


def aggregate_predictions(predictions: List[Dict]) -> Dict:
    """Aggregate with dynamic weighting based on consensus strength."""
    valid = [p for p in predictions if p.get('success')]

    if not valid:
        return {
            'weighted_adjustment_pct': 0.0,
            'consensus_direction': 'neutral',
            'consensus_strength': 0.0,
            'avg_confidence': 5,
            'num_valid': 0,
            'bullish_pct': 0,
            'bearish_pct': 0,
        }

    total_weight = sum(p['weight'] for p in valid)

    # Calculate weighted adjustment
    weighted_adj = sum(
        convert_adjustment_to_pct(p['direction'], p['adjustment_magnitude'], p['confidence'])
        * (p['weight'] / total_weight)
        for p in valid
    )

    # Calculate consensus metrics
    bullish_weight = sum(p['weight'] for p in valid if p['direction'] == 'bullish_usd')
    bearish_weight = sum(p['weight'] for p in valid if p['direction'] == 'bearish_usd')
    neutral_weight = sum(p['weight'] for p in valid if p['direction'] == 'neutral')

    bullish_pct = bullish_weight / total_weight
    bearish_pct = bearish_weight / total_weight

    # Consensus strength: how much agreement is there?
    consensus_strength = abs(bullish_pct - bearish_pct)

    if bullish_pct > bearish_pct + 0.1:
        consensus = 'bullish_usd'
    elif bearish_pct > bullish_pct + 0.1:
        consensus = 'bearish_usd'
    else:
        consensus = 'mixed'

    avg_confidence = sum(p['confidence'] * p['weight'] for p in valid) / total_weight

    return {
        'weighted_adjustment_pct': weighted_adj,
        'consensus_direction': consensus,
        'consensus_strength': consensus_strength,
        'avg_confidence': avg_confidence,
        'num_valid': len(valid),
        'bullish_pct': bullish_pct,
        'bearish_pct': bearish_pct,
    }


def calculate_dynamic_weights(
    consensus_strength: float,
    avg_confidence: float,
    stat_r_squared: float,
    regime_volatility: str
) -> Tuple[float, float]:
    """
    Dynamically calculate stat vs LLM weights based on:
    - Consensus strength (high agreement = trust LLM more)
    - Average confidence (high confidence = trust LLM more)
    - Statistical model R² (high R² = trust stats more)
    - Volatility regime (high vol = trust stats more)
    """
    # Start with default
    stat_weight = DEFAULT_STAT_WEIGHT

    # Adjust based on consensus strength
    # Strong consensus (>0.5) = personas agree = give LLM more weight
    # Weak consensus (<0.3) = personas divided = trust stats more
    if consensus_strength > 0.5:
        stat_weight -= 0.08
    elif consensus_strength < 0.3:
        stat_weight += 0.05

    # Adjust based on confidence
    if avg_confidence >= 7:
        stat_weight -= 0.05
    elif avg_confidence <= 4:
        stat_weight += 0.05

    # Adjust based on statistical model quality
    if stat_r_squared > 0.6:
        stat_weight += 0.05
    elif stat_r_squared < 0.4:
        stat_weight -= 0.03

    # Adjust based on volatility
    if regime_volatility == 'high':
        stat_weight += 0.05  # Trust stats more in volatile markets
    elif regime_volatility == 'low':
        stat_weight -= 0.03  # Can trust LLM more in calm markets

    # Clamp to bounds
    stat_weight = max(MIN_STAT_WEIGHT, min(MAX_STAT_WEIGHT, stat_weight))
    llm_weight = 1 - stat_weight

    return stat_weight, llm_weight


def ensemble_prediction(
    stat_prediction: float,
    llm_adjustment_pct: float,
    stat_weight: float,
    llm_weight: float
) -> float:
    """Combine statistical and LLM predictions."""
    # LLM adjustment scaled by its weight
    effective_adjustment = llm_adjustment_pct * (llm_weight / stat_weight)
    final = stat_prediction * (1 + effective_adjustment / 100)
    return final


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(api_key: str, personas_file: str = PERSONAS_FILE):
    print("=" * 70)
    print("GEMINI USD/INR FOREX SIMULATION V3 - ADVERSARIAL + DYNAMIC WEIGHTS")
    print("=" * 70)
    print(f"Period: {SIMULATION_START.date()} to {SIMULATION_END.date()}")
    print(f"Key fixes: Adversarial personas, stat baseline in prompts, dynamic weights")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load personas
    personas_path = os.path.join(script_dir, personas_file)
    personas = load_personas(personas_path)
    print(f"Loaded {len(personas)} adversarial personas")

    # Show persona balance
    bullish_w = sum(p['weight'] for p in personas if p.get('bias') == 'bullish_usd')
    bearish_w = sum(p['weight'] for p in personas if p.get('bias') == 'bearish_usd')
    neutral_w = sum(p['weight'] for p in personas if p.get('bias') == 'neutral')
    print(f"  Bias balance: Bullish {bullish_w*100:.0f}% | Bearish {bearish_w*100:.0f}% | Neutral {neutral_w*100:.0f}%")

    # Load data
    data_path = os.path.join(script_dir, DATA_FILE)
    market_df = load_market_data(data_path)

    # Initialize
    print("\nInitializing Gemini 2.0 Flash...")
    model = initialize_gemini(api_key)
    stat_model = StatisticalModel(lookback_window=REGRESSION_WINDOW)

    trading_days = get_trading_days(market_df, SIMULATION_START, SIMULATION_END)
    print(f"\nSimulating {len(trading_days)} trading days")
    print()

    all_results = []

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

        stat_implied_change = (stat_pred - last_close) / last_close * 100
        contributions = stat_model.get_feature_contributions(t1_features)

        print(f"  Statistical: {stat_pred:.4f} ({stat_implied_change:+.3f}%) R²={stat_model.r_squared:.3f}")

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
        else:
            # Adjust persona weights for regime
            adjusted_personas = adjust_persona_weights_for_regime(personas, context['regime'])

            # Initial delay before starting API calls for this day
            time.sleep(INITIAL_DELAY)

            # Query personas
            day_predictions = []
            running_consensus = None

            for i, persona in enumerate(adjusted_personas):
                # Rate limiting: longer delay every few requests
                if i > 0 and i % 3 == 0:
                    time.sleep(API_DELAY_SECONDS)  # Extra pause every 3rd persona

                print(f"  {persona['short_name']} ({persona['bias'][:4] if persona['bias'] != 'neutral' else 'neut'})...", end=" ")

                # Format prompt with statistical baseline and current consensus
                prompt = format_persona_prompt(
                    persona,
                    context,
                    stat_pred,
                    stat_implied_change,
                    contributions,
                    running_consensus
                )

                prediction = query_persona(model, persona, prompt)
                day_predictions.append(prediction)

                if prediction['success']:
                    adj = convert_adjustment_to_pct(
                        prediction['direction'],
                        prediction['adjustment_magnitude'],
                        prediction['confidence']
                    )
                    print(f"{prediction['direction'][:4]} {prediction['adjustment_magnitude'][:3]} c={prediction['confidence']} -> {adj:+.2f}%")

                    # Update running consensus for contrarian prompts
                    if i >= 2:  # After first few responses
                        temp_agg = aggregate_predictions(day_predictions)
                        running_consensus = temp_agg['consensus_direction']
                else:
                    print("FAIL")

                time.sleep(API_DELAY_SECONDS)

            # Aggregate
            llm_result = aggregate_predictions(day_predictions)

            print(f"\n  LLM Aggregate: {llm_result['consensus_direction']} "
                  f"(bull:{llm_result['bullish_pct']*100:.0f}% bear:{llm_result['bearish_pct']*100:.0f}%) "
                  f"strength:{llm_result['consensus_strength']:.2f} "
                  f"adj:{llm_result['weighted_adjustment_pct']:+.3f}%")

            # Dynamic weights
            stat_weight, llm_weight = calculate_dynamic_weights(
                llm_result['consensus_strength'],
                llm_result['avg_confidence'],
                stat_model.r_squared,
                context['regime']['volatility']
            )

            print(f"  Dynamic weights: Stat {stat_weight*100:.0f}% | LLM {llm_weight*100:.0f}%")

            # Ensemble
            final_prediction = ensemble_prediction(
                stat_pred,
                llm_result['weighted_adjustment_pct'],
                stat_weight,
                llm_weight
            )

            print(f"  Final: {final_prediction:.4f}")

            day_result = {
                'date': target_date.isoformat(),
                'day_idx': day_idx,
                'mode': 'hybrid',
                'last_close': last_close,
                'stat_prediction': stat_pred,
                'stat_r_squared': stat_model.r_squared,
                'llm_adjustment_pct': llm_result['weighted_adjustment_pct'],
                'llm_consensus': llm_result['consensus_direction'],
                'llm_consensus_strength': llm_result['consensus_strength'],
                'bullish_pct': llm_result['bullish_pct'],
                'bearish_pct': llm_result['bearish_pct'],
                'stat_weight': stat_weight,
                'llm_weight': llm_weight,
                'final_prediction': final_prediction,
                'actual_price': actual_price,
                'prediction_error_pct': ((final_prediction - actual_price) / actual_price * 100) if actual_price else None,
                'stat_error_pct': ((stat_pred - actual_price) / actual_price * 100) if actual_price else None,
                'regime_volatility': context['regime']['volatility'],
                'regime_trend': context['regime']['trend'],
                'persona_predictions': [{k: v for k, v in p.items() if k != 'raw_response'} for p in day_predictions]
            }

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
            'bullish_pct': r.get('bullish_pct'),
            'bearish_pct': r.get('bearish_pct'),
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
            print("V3 PERFORMANCE METRICS")
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

            # Did LLM help?
            llm_helped = sum(
                abs(row['prediction_error_pct']) < abs(row['stat_error_pct'])
                for _, row in valid.iterrows()
                if row['stat_error_pct'] is not None
            )
            total = len([r for _, r in valid.iterrows() if r['stat_error_pct'] is not None])
            print(f"\nLLM helped in {llm_helped}/{total} cases ({100*llm_helped/total:.1f}%)")

            # Bias balance
            print(f"\nBias Balance:")
            print(f"  Avg Bullish: {valid['bullish_pct'].mean()*100:.1f}%")
            print(f"  Avg Bearish: {valid['bearish_pct'].mean()*100:.1f}%")

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

"""
Gemini USD/INR Forex Simulation V5 - PURE LLM PREDICTIONS

This version relies SOLELY on LLM persona predictions without any statistical model.
The goal is to test pure AI-based market prediction capabilities.

Key differences from V4:
1. NO STATISTICAL MODEL - only LLM predictions
2. LLM provides DIRECT PRICE PREDICTIONS (not adjustments)
3. Trimmed weighted mean of persona predictions
4. News digest integration for context
5. Entropy-based confidence weighting
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
PERSONAS_FILE = "forex_personas_v5.json"
RESULTS_FILE = "simulation_results_v5.json"
SUMMARY_CSV = "simulation_summary_v5.csv"

DATA_FILE = "../Super_Master_Dataset.csv"

# News data files
INDIA_NEWS_FILE = "../india_news_combined_sorted.csv"
USA_NEWS_FILE = "../usa_news_combined_sorted.csv"

# News digest settings
USE_NEWS_DIGEST = True
MAX_HEADLINES_PER_DAY = 40

WARMUP_DAYS = 10  # Shorter warmup since we're not training a model

# Trimmed mean parameters
TRIM_FRACTION = 0.1  # Remove top and bottom 10% of predictions

API_DELAY_SECONDS = 6
MAX_RETRIES = 3
INITIAL_DELAY = 2

# Prediction bounds (to filter outliers)
MAX_DAILY_CHANGE_PCT = 2.0  # Max expected daily change


# ============================================================================
# REGIME DETECTION
# ============================================================================

class RegimeDetector:
    """Detect market regime for context."""

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
        return "range_bound"

    @staticmethod
    def get_mean_reversion_signal(prices: List[float], window: int = 20) -> float:
        """Z-score of current price relative to moving average."""
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
            'errors': [],
            'direction_correct': 0,
            'total': 0,
        })

    def record_prediction(self, persona_id: str, predicted_price: float, 
                         confidence: int, direction: str):
        """Record a persona's prediction."""
        self.persona_history[persona_id]['predictions'].append({
            'predicted': predicted_price,
            'confidence': confidence,
            'direction': direction,
            'timestamp': datetime.now().isoformat()
        })

    def record_outcome(self, persona_id: str, predicted: float, actual: float, 
                      prev_close: float):
        """Record outcome of prediction."""
        error_pct = abs(predicted - actual) / actual * 100
        self.persona_history[persona_id]['errors'].append(error_pct)
        self.persona_history[persona_id]['total'] += 1
        
        # Direction accuracy
        pred_dir = 'up' if predicted > prev_close else 'down'
        actual_dir = 'up' if actual > prev_close else 'down'
        if pred_dir == actual_dir:
            self.persona_history[persona_id]['direction_correct'] += 1
        
        # Keep only recent history
        if len(self.persona_history[persona_id]['errors']) > 100:
            self.persona_history[persona_id]['errors'] = \
                self.persona_history[persona_id]['errors'][-100:]

    def get_persona_mae(self, persona_id: str) -> float:
        """Get persona's historical MAE."""
        errors = self.persona_history[persona_id]['errors']
        if len(errors) < 5:
            return 0.3  # Default MAE
        return np.mean(errors)

    def get_persona_accuracy(self, persona_id: str) -> float:
        """Get persona's direction accuracy."""
        stats = self.persona_history[persona_id]
        if stats['total'] < 5:
            return 0.5
        return stats['direction_correct'] / stats['total']

    def get_adaptive_weight(self, persona_id: str, base_weight: float) -> float:
        """Adjust weight based on performance."""
        mae = self.get_persona_mae(persona_id)
        accuracy = self.get_persona_accuracy(persona_id)

        # Better MAE = higher weight
        mae_factor = 0.3 / (mae + 0.1)  # Inverse relationship
        mae_factor = np.clip(mae_factor, 0.5, 2.0)

        # Better accuracy = higher weight
        acc_factor = 1.0 + (accuracy - 0.5) * 0.5

        return base_weight * mae_factor * acc_factor


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
    """Get market context for LLM prompts."""
    available = df[df['Date'] < target_date].tail(60)

    if len(available) < 20:
        return None

    recent_5 = available.tail(5)
    recent_20 = available.tail(20)

    def safe_float(val, default=0.0):
        try:
            return float(val) if pd.notna(val) else default
        except:
            return default

    # Get last 5 days of price action
    price_history = []
    for idx in range(-5, 0):
        if abs(idx) <= len(available):
            row = available.iloc[idx]
            price_history.append({
                'date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
                'usdinr': safe_float(row['INR']),
                'change_pct': safe_float(row['INR_change']),
            })

    t1_row = available.iloc[-1]
    t2_row = available.iloc[-2]

    # Calculate volatility
    returns = available['INR_change'].dropna().values
    realized_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)

    # Mean reversion
    mr_signal = RegimeDetector.get_mean_reversion_signal(list(available['INR'].values))

    context = {
        't2': {
            'date': t2_row['Date'].strftime('%Y-%m-%d') if hasattr(t2_row['Date'], 'strftime') else str(t2_row['Date']),
            'usdinr': safe_float(t2_row['INR']),
            'oil': safe_float(t2_row['OIL']),
            'gold': safe_float(t2_row['GOLD']),
            'us10y': safe_float(t2_row['US10Y']),
            'dxy': safe_float(t2_row['DXY']),
            'india_tone': safe_float(t2_row['IN_Avg_Tone']),
            'us_tone': safe_float(t2_row['US_Avg_Tone']),
        },
        't1': {
            'date': t1_row['Date'].strftime('%Y-%m-%d') if hasattr(t1_row['Date'], 'strftime') else str(t1_row['Date']),
            'usdinr': safe_float(t1_row['INR']),
            'oil': safe_float(t1_row['OIL']),
            'gold': safe_float(t1_row['GOLD']),
            'us10y': safe_float(t1_row['US10Y']),
            'dxy': safe_float(t1_row['DXY']),
            'india_tone': safe_float(t1_row['IN_Avg_Tone']),
            'us_tone': safe_float(t1_row['US_Avg_Tone']),
        },
        'price_history': price_history,
        'stats': {
            'inr_20d_mean': safe_float(recent_20['INR'].mean()),
            'inr_20d_std': safe_float(recent_20['INR'].std()),
            'inr_5d_mean': safe_float(recent_5['INR'].mean()),
            'inr_5d_momentum': safe_float(available['INR_change'].tail(5).sum()),
            'realized_vol': realized_vol,
            'mean_reversion_zscore': mr_signal,
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


def format_v5_prompt(
    persona: Dict,
    context: Dict,
    target_date: datetime,
    persona_accuracy: float,
    persona_mae: float,
    news_digest: Optional[str] = None
) -> str:
    """
    Create a prompt for DIRECT PRICE PREDICTION.
    No statistical model reference - pure LLM prediction.
    """
    t1 = context['t1']
    t2 = context['t2']
    regime = context['regime']
    stats = context['stats']
    price_history = context['price_history']

    # Calculate changes
    dxy_change = t1['dxy'] - t2['dxy']
    dxy_pct = (dxy_change / t2['dxy'] * 100) if t2['dxy'] else 0
    us10y_change = t1['us10y'] - t2['us10y']
    gold_change = t1['gold'] - t2['gold']
    gold_pct = (gold_change / t2['gold'] * 100) if t2['gold'] else 0
    oil_change = t1['oil'] - t2['oil']
    oil_pct = (oil_change / t2['oil'] * 100) if t2['oil'] else 0
    inr_change = t1['usdinr'] - t2['usdinr']
    inr_pct = (inr_change / t2['usdinr'] * 100) if t2['usdinr'] else 0

    # Format price history
    history_text = "\n".join([
        f"  {h['date']}: {h['usdinr']:.4f} ({h['change_pct']:+.3f}%)"
        for h in price_history
    ])

    # Build balanced case analysis
    bull_factors = []
    bear_factors = []

    if dxy_pct > 0.1:
        bull_factors.append(f"DXY strengthening ({dxy_pct:+.2f}%)")
    elif dxy_pct < -0.1:
        bear_factors.append(f"DXY weakening ({dxy_pct:+.2f}%)")

    if us10y_change > 0.02:
        bull_factors.append(f"US yields rising ({us10y_change:+.3f}%)")
    elif us10y_change < -0.02:
        bear_factors.append(f"US yields falling ({us10y_change:+.3f}%)")

    if oil_pct > 1:
        bull_factors.append(f"Oil up ({oil_pct:+.1f}%) - pressures INR")
    elif oil_pct < -1:
        bear_factors.append(f"Oil down ({oil_pct:+.1f}%) - helps INR")

    if t1['india_tone'] > t2['india_tone'] + 0.5:
        bear_factors.append(f"India sentiment improving")
    elif t1['india_tone'] < t2['india_tone'] - 0.5:
        bull_factors.append(f"India sentiment weakening")

    mr_z = stats.get('mean_reversion_zscore', 0)
    if mr_z > 1.5:
        bear_factors.append(f"Overbought (z={mr_z:.2f}), reversion likely")
    elif mr_z < -1.5:
        bull_factors.append(f"Oversold (z={mr_z:.2f}), reversion likely")

    if regime['volatility'] == 'high':
        bull_factors.append("High volatility (risk-off favors USD)")

    # Randomize presentation order
    if random.random() > 0.5:
        case_text = f"""
**Bullish USD factors (rate could go UP):**
{chr(10).join(['  + ' + f for f in bull_factors]) if bull_factors else '  (none strong)'}

**Bearish USD factors (rate could go DOWN):**
{chr(10).join(['  - ' + f for f in bear_factors]) if bear_factors else '  (none strong)'}
"""
    else:
        case_text = f"""
**Bearish USD factors (rate could go DOWN):**
{chr(10).join(['  - ' + f for f in bear_factors]) if bear_factors else '  (none strong)'}

**Bullish USD factors (rate could go UP):**
{chr(10).join(['  + ' + f for f in bull_factors]) if bull_factors else '  (none strong)'}
"""

    # Accuracy feedback
    if persona_accuracy >= 0.55:
        acc_text = f"Your direction accuracy: {persona_accuracy*100:.0f}% (above average)"
    elif persona_accuracy <= 0.45:
        acc_text = f"Your direction accuracy: {persona_accuracy*100:.0f}% (below average - recalibrate)"
    else:
        acc_text = f"Your direction accuracy: {persona_accuracy*100:.0f}%"

    mae_text = f"Your average error: {persona_mae:.3f}%"

    # News section
    news_section = ""
    if news_digest:
        news_section = f"""
---

### NEWS DIGEST

{news_digest}

**Interpret this news from YOUR analytical perspective.**
"""

    prompt = f"""
## USD/INR FOREX PREDICTION

**Predict the USD/INR exchange rate for: {target_date.strftime('%Y-%m-%d')}**

{acc_text} | {mae_text}

---

### RECENT PRICE HISTORY

{history_text}

**Current rate (as of {t1['date']}): {t1['usdinr']:.4f}**
**Yesterday's change: {inr_pct:+.3f}%**

---

### MARKET INDICATORS

| Indicator | Yesterday | Today | Change |
|-----------|-----------|-------|--------|
| DXY | {t2['dxy']:.2f} | {t1['dxy']:.2f} | {dxy_pct:+.2f}% |
| US 10Y | {t2['us10y']:.3f}% | {t1['us10y']:.3f}% | {us10y_change:+.3f}% |
| Gold | ${t2['gold']:.0f} | ${t1['gold']:.0f} | {gold_pct:+.2f}% |
| Oil | ${t2['oil']:.1f} | ${t1['oil']:.1f} | {oil_pct:+.2f}% |
| India Tone | {t2['india_tone']:.2f} | {t1['india_tone']:.2f} | {t1['india_tone']-t2['india_tone']:+.2f} |

**Volatility**: {regime['volatility'].upper()} | **Trend**: {regime['trend'].replace('_', ' ').upper()}
**20-day mean**: {stats['inr_20d_mean']:.4f} | **5-day momentum**: {stats['inr_5d_momentum']:+.3f}%
{news_section}
---

### ANALYSIS FRAMEWORK

{case_text}

---

### YOUR PREDICTION

Based on your expertise, predict tomorrow's USD/INR closing rate.

**Guidelines:**
- Current rate: {t1['usdinr']:.4f}
- Typical daily range: ±0.1% to ±0.3%
- Large moves (>0.5%) require strong justification
- Consider both fundamental and technical factors

Respond with ONLY this JSON:
```json
{{
    "predicted_rate": <your predicted USD/INR rate as a float>,
    "direction": "higher" | "lower" | "unchanged",
    "expected_change_pct": <expected change from current as percentage>,
    "confidence": <1-10>,
    "primary_reason": "<one sentence justification>"
}}
```
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
            'temperature': 0.4,
            'top_p': 0.85,
            'max_output_tokens': 1024,
        }
    )
    return model


def extract_response_text(response) -> str:
    """Extract text from Gemini response."""
    # Collect all text parts
    all_text = []
    
    # Try direct .text access first
    try:
        if response.text:
            return response.text
    except (ValueError, AttributeError):
        pass

    # Fall back to parsing candidates
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    # Skip thinking/reasoning parts
                    if hasattr(part, 'thought') and part.thought:
                        continue
                    if hasattr(part, 'text') and part.text:
                        all_text.append(part.text)
    
    return "\n".join(all_text) if all_text else ""


def parse_v5_response(response_text: str, last_close: float) -> Optional[Dict]:
    """Parse V5 format response with direct price prediction."""
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

        # Extract predicted rate
        predicted_rate = parsed.get('predicted_rate')
        if isinstance(predicted_rate, str):
            predicted_rate = float(re.search(r'[\d.]+', predicted_rate).group())
        predicted_rate = float(predicted_rate)

        # Sanity check: rate should be within reasonable bounds
        max_change = last_close * (MAX_DAILY_CHANGE_PCT / 100)
        if abs(predicted_rate - last_close) > max_change:
            # Clamp to reasonable range
            if predicted_rate > last_close:
                predicted_rate = last_close + max_change
            else:
                predicted_rate = last_close - max_change

        # Normalize direction
        direction = parsed.get('direction', 'unchanged').lower().strip()
        if direction in ['higher', 'bullish', 'up']:
            direction = 'higher'
        elif direction in ['lower', 'bearish', 'down']:
            direction = 'lower'
        else:
            direction = 'unchanged'

        return {
            'predicted_rate': predicted_rate,
            'direction': direction,
            'expected_change_pct': float(parsed.get('expected_change_pct', 0)),
            'confidence': int(parsed.get('confidence', 5)),
            'primary_reason': parsed.get('primary_reason', ''),
        }

    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return None


def query_persona(model: genai.GenerativeModel, persona: Dict, 
                  market_prompt: str, last_close: float, 
                  retries: int = MAX_RETRIES) -> Dict:
    """Query a persona for direct price prediction."""
    system_prompt = persona['system_prompt']
    full_prompt = f"{system_prompt}\n\n{market_prompt}"

    last_error = None
    for attempt in range(retries):
        try:
            response = model.generate_content(full_prompt)
            response_text = extract_response_text(response)

            if not response_text.strip():
                last_error = "Empty response"
                backoff = API_DELAY_SECONDS * (2 ** attempt)
                time.sleep(backoff)
                continue

            parsed = parse_v5_response(response_text, last_close)

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

            last_error = f"Parse failed: {response_text[:100]}..."
            time.sleep(API_DELAY_SECONDS)

        except Exception as e:
            last_error = str(e)
            error_str = str(e).lower()
            if 'rate' in error_str or 'quota' in error_str or '429' in error_str:
                backoff = API_DELAY_SECONDS * (3 ** attempt)
                time.sleep(backoff)
            else:
                time.sleep(API_DELAY_SECONDS * 2)

    # Fallback: return last close as prediction
    return {
        'success': False,
        'persona_id': persona['id'],
        'persona_name': persona['name'],
        'short_name': persona.get('short_name', persona['name'][:15]),
        'weight': persona.get('adaptive_weight', persona['weight']),
        'original_weight': persona['weight'],
        'predicted_rate': last_close,
        'direction': 'unchanged',
        'expected_change_pct': 0,
        'confidence': 1,
        'primary_reason': f'API failure: {last_error}',
        'raw_response': None
    }


# ============================================================================
# AGGREGATION
# ============================================================================

def trimmed_weighted_mean(values: List[float], weights: List[float], 
                          trim_fraction: float = 0.1) -> float:
    """Calculate trimmed weighted mean - removes extreme predictions."""
    if len(values) == 0:
        return 0.0

    if len(values) <= 2:
        return np.average(values, weights=weights)

    sorted_pairs = sorted(zip(values, weights), key=lambda x: x[0])
    n = len(sorted_pairs)
    trim_count = max(1, int(n * trim_fraction))

    trimmed_pairs = sorted_pairs[trim_count:-trim_count] if trim_count > 0 else sorted_pairs

    if len(trimmed_pairs) == 0:
        return np.median(values)

    trimmed_values, trimmed_weights = zip(*trimmed_pairs)
    total_weight = sum(trimmed_weights)

    if total_weight == 0:
        return np.mean(trimmed_values)

    return sum(v * w for v, w in zip(trimmed_values, trimmed_weights)) / total_weight


def calculate_entropy_weight(predictions: List[Dict]) -> float:
    """Calculate ensemble confidence based on agreement."""
    if len(predictions) < 3:
        return 0.5

    directions = [p['direction'] for p in predictions if p.get('success')]
    if not directions:
        return 0.5

    counts = {'higher': 0, 'lower': 0, 'unchanged': 0}
    for d in directions:
        counts[d] = counts.get(d, 0) + 1

    total = len(directions)
    probs = [c / total for c in counts.values() if c > 0]

    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(3)

    # Normalized: 0 = max entropy, 1 = complete agreement
    return 1 - (entropy / max_entropy)


def aggregate_predictions(predictions: List[Dict], last_close: float,
                         trim_fraction: float = TRIM_FRACTION) -> Dict:
    """Aggregate all persona predictions into final prediction."""
    valid = [p for p in predictions if p.get('success')]

    if not valid:
        return {
            'final_prediction': last_close,
            'simple_mean': last_close,
            'consensus_direction': 'unchanged',
            'entropy_confidence': 0.0,
            'avg_confidence': 0,
            'num_valid': 0,
            'higher_pct': 0,
            'lower_pct': 0,
            'prediction_std': 0,
        }

    # Get predictions and weights
    pred_rates = [p['predicted_rate'] for p in valid]
    weights = [p['weight'] for p in valid]

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Trimmed weighted mean of predictions
    final_prediction = trimmed_weighted_mean(pred_rates, weights, trim_fraction)

    # Simple weighted mean for comparison
    simple_mean = np.average(pred_rates, weights=weights)

    # Prediction spread
    pred_std = np.std(pred_rates)

    # Direction percentages
    higher_weight = sum(p['weight'] for p in valid if p['direction'] == 'higher')
    lower_weight = sum(p['weight'] for p in valid if p['direction'] == 'lower')
    unchanged_weight = sum(p['weight'] for p in valid if p['direction'] == 'unchanged')

    total_w = higher_weight + lower_weight + unchanged_weight
    higher_pct = higher_weight / total_w if total_w > 0 else 0
    lower_pct = lower_weight / total_w if total_w > 0 else 0

    # Consensus direction
    if higher_pct > lower_pct + 0.15:
        consensus = 'higher'
    elif lower_pct > higher_pct + 0.15:
        consensus = 'lower'
    else:
        consensus = 'mixed'

    # Entropy confidence
    entropy_confidence = calculate_entropy_weight(valid)

    # Average confidence
    avg_confidence = sum(p['confidence'] * p['weight'] for p in valid) / total_w

    return {
        'final_prediction': final_prediction,
        'simple_mean': simple_mean,
        'consensus_direction': consensus,
        'entropy_confidence': entropy_confidence,
        'avg_confidence': avg_confidence,
        'num_valid': len(valid),
        'higher_pct': higher_pct,
        'lower_pct': lower_pct,
        'prediction_std': pred_std,
    }


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(api_key: str, personas_file: str = PERSONAS_FILE):
    print("=" * 70)
    print("GEMINI USD/INR FOREX SIMULATION V5 - PURE LLM PREDICTIONS")
    print("=" * 70)
    print(f"Period: {SIMULATION_START.date()} to {SIMULATION_END.date()}")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load personas
    personas_path = os.path.join(script_dir, personas_file)
    
    # Fall back to V4 personas if V5 doesn't exist
    if not os.path.exists(personas_path):
        personas_path = os.path.join(script_dir, "forex_personas_v4.json")
        if not os.path.exists(personas_path):
            personas_path = os.path.join(script_dir, "forex_personas_v3.json")
        print(f"Using fallback personas from {personas_path}")
    
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
            print(f"\nWarning: India news file not found")
    else:
        print("\nNews digest disabled")

    trading_days = get_trading_days(market_df, SIMULATION_START, SIMULATION_END)
    print(f"\nSimulating {len(trading_days)} trading days")

    all_results = []

    for day_idx, target_date in enumerate(trading_days):
        is_warmup = day_idx < WARMUP_DAYS

        print(f"\n[Day {day_idx + 1}/{len(trading_days)}] {target_date.date()}")
        if is_warmup:
            print("  MODE: WARMUP (no LLM calls)")
        print("-" * 50)

        context = get_market_context(market_df, target_date)
        if not context:
            print("  Skipping - insufficient data")
            continue

        last_close = context['t1']['usdinr']
        actual_price = get_actual_price(market_df, target_date)

        if is_warmup:
            # During warmup, just use last close as prediction
            final_prediction = last_close
            day_result = {
                'date': target_date.isoformat(),
                'day_idx': day_idx,
                'mode': 'warmup',
                'last_close': last_close,
                'final_prediction': final_prediction,
                'actual_price': actual_price,
                'prediction_error_pct': ((final_prediction - actual_price) / actual_price * 100) if actual_price else None,
                'persona_predictions': []
            }
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

            # Get news digest
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

            time.sleep(INITIAL_DELAY)

            # Query personas
            day_predictions = []

            for i, persona in enumerate(personas):
                if i > 0 and i % 3 == 0:
                    time.sleep(API_DELAY_SECONDS)

                persona_acc = persona_tracker.get_persona_accuracy(persona['id'])
                persona_mae = persona_tracker.get_persona_mae(persona['id'])

                print(f"  {persona['short_name']}...", end=" ")

                prompt = format_v5_prompt(
                    persona,
                    context,
                    target_date,
                    persona_acc,
                    persona_mae,
                    news_digest=news_digest_text
                )

                prediction = query_persona(model, persona, prompt, last_close)
                day_predictions.append(prediction)

                if prediction['success']:
                    pred_change = (prediction['predicted_rate'] - last_close) / last_close * 100
                    print(f"{prediction['predicted_rate']:.4f} ({pred_change:+.3f}%) c={prediction['confidence']}")
                else:
                    reason = prediction.get('primary_reason', 'Unknown')[:50]
                    print(f"FAIL ({reason})")

                time.sleep(API_DELAY_SECONDS)

            # Aggregate predictions
            agg_result = aggregate_predictions(day_predictions, last_close)
            final_prediction = agg_result['final_prediction']

            pred_change = (final_prediction - last_close) / last_close * 100
            print(f"\n  LLM Aggregate: {final_prediction:.4f} ({pred_change:+.3f}%)")
            print(f"  Consensus: {agg_result['consensus_direction']} "
                  f"(higher:{agg_result['higher_pct']*100:.0f}% lower:{agg_result['lower_pct']*100:.0f}%)")
            print(f"  Entropy confidence: {agg_result['entropy_confidence']:.2f}")
            print(f"  Prediction std: {agg_result['prediction_std']:.4f}")

            day_result = {
                'date': target_date.isoformat(),
                'day_idx': day_idx,
                'mode': 'llm_only',
                'last_close': last_close,
                'final_prediction': final_prediction,
                'simple_mean': agg_result['simple_mean'],
                'consensus_direction': agg_result['consensus_direction'],
                'entropy_confidence': agg_result['entropy_confidence'],
                'avg_confidence': agg_result['avg_confidence'],
                'higher_pct': agg_result['higher_pct'],
                'lower_pct': agg_result['lower_pct'],
                'prediction_std': agg_result['prediction_std'],
                'news_available': news_digest_text is not None,
                'actual_price': actual_price,
                'prediction_error_pct': ((final_prediction - actual_price) / actual_price * 100) if actual_price else None,
                'regime_volatility': context['regime']['volatility'],
                'regime_trend': context['regime']['trend'],
                'persona_predictions': [{k: v for k, v in p.items() if k != 'raw_response'} for p in day_predictions]
            }

            # Update persona tracking
            if actual_price:
                for p in day_predictions:
                    if p.get('success'):
                        persona_tracker.record_outcome(
                            p['persona_id'], 
                            p['predicted_rate'], 
                            actual_price,
                            last_close
                        )

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
            'final_prediction': r['final_prediction'],
            'actual_price': r['actual_price'],
            'prediction_error_pct': r.get('prediction_error_pct'),
            'entropy_confidence': r.get('entropy_confidence'),
            'higher_pct': r.get('higher_pct'),
            'lower_pct': r.get('lower_pct'),
            'consensus_direction': r.get('consensus_direction'),
        })

    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, SUMMARY_CSV)
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    # Calculate metrics for LLM-only days
    llm_only = df[df['mode'] == 'llm_only']
    if not llm_only.empty:
        valid = llm_only[llm_only['prediction_error_pct'].notna()]

        if not valid.empty:
            print("\n" + "=" * 50)
            print("V5 PERFORMANCE METRICS (PURE LLM)")
            print("=" * 50)

            print(f"\nLLM Ensemble:")
            print(f"  MAE:  {valid['prediction_error_pct'].abs().mean():.4f}%")
            print(f"  Bias: {valid['prediction_error_pct'].mean():+.4f}%")
            print(f"  Std:  {valid['prediction_error_pct'].std():.4f}%")
            print(f"  Max:  {valid['prediction_error_pct'].abs().max():.4f}%")
            print(f"  Median: {valid['prediction_error_pct'].abs().median():.4f}%")

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

            # Random walk comparison
            rw_errors = []
            for _, row in valid.iterrows():
                rw_error = abs(row['last_close'] - row['actual_price']) / row['actual_price'] * 100
                rw_errors.append(rw_error)
            rw_mae = np.mean(rw_errors)
            print(f"\nRandom Walk Benchmark MAE: {rw_mae:.4f}%")
            if valid['prediction_error_pct'].abs().mean() < rw_mae:
                print("✓ LLM beats random walk!")
            else:
                print("✗ Random walk is better")


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

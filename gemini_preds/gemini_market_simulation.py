"""
Gemini 2.5 Pro USD/INR Forex Simulation using LLM Personas
Simulates USD/INR exchange rate predictions using weighted personas from Dec 1, 2025 to Jan 5, 2026
Uses existing GDELT news sentiment and market data from local CSV files.
"""

import os
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import google.generativeai as genai

# ============================================================================
# CONFIGURATION
# ============================================================================

SIMULATION_START = datetime(2023, 1, 1)
SIMULATION_END = datetime(2023, 12, 31)
PERSONAS_FILE = "forex_personas.json"
RESULTS_FILE = "simulation_results.json"
SUMMARY_CSV = "simulation_summary.csv"

# Data file (relative to project root)
DATA_FILE = "../Super_Master_Dataset.csv"

# Warmup period - number of days at start where we show actual USD/INR price
# After warmup, the LLM must predict without seeing the actual exchange rate
WARMUP_DAYS = 3

# Rate limiting
API_DELAY_SECONDS = 2  # Delay between API calls to avoid rate limits
MAX_RETRIES = 3

# ============================================================================
# DATA LOADING FROM LOCAL FILES
# ============================================================================

def load_market_data(filepath: str) -> pd.DataFrame:
    """Load market data from the Super_Master_Dataset.csv file."""
    print(f"Loading market data from {filepath}...")

    df = pd.read_csv(filepath, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    return df


def get_trading_days(df: pd.DataFrame, start: datetime, end: datetime) -> List[datetime]:
    """Get list of trading days within the simulation period."""
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    trading_days = df.loc[mask, 'Date'].tolist()
    return sorted([d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d for d in trading_days])


def get_market_context(df: pd.DataFrame, target_date: datetime) -> Optional[Dict]:
    """
    Get market data for T-2 and T-1 relative to target_date.
    Returns None if insufficient historical data.
    """
    # Get all dates before target_date
    available = df[df['Date'] < target_date].tail(2)

    if len(available) < 2:
        return None

    t2_row = available.iloc[0]
    t1_row = available.iloc[1]

    def safe_float(val, default=0.0):
        try:
            return float(val) if pd.notna(val) else default
        except:
            return default

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
        }
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


def format_market_prompt(context: Dict, show_price: bool = True, last_known_price: float = None) -> str:
    """
    Format market data into a prompt WITHOUT revealing actual dates.

    Args:
        context: Market data for T-2 and T-1
        show_price: If True, show USD/INR prices. If False, hide them (blind prediction mode)
        last_known_price: The last USD/INR price the model was shown (used when show_price=False)
    """
    t2 = context['t2']
    t1 = context['t1']

    # Calculate changes for other indicators
    oil_change = t1['oil'] - t2['oil']
    oil_pct = (oil_change / t2['oil'] * 100) if t2['oil'] else 0

    gold_change = t1['gold'] - t2['gold']
    gold_pct = (gold_change / t2['gold'] * 100) if t2['gold'] else 0

    dxy_change = t1['dxy'] - t2['dxy']
    dxy_pct = (dxy_change / t2['dxy'] * 100) if t2['dxy'] else 0

    us10y_change = t1['us10y'] - t2['us10y']

    # Build USD/INR section based on whether we show prices
    if show_price:
        usdinr_change = t1['usdinr'] - t2['usdinr']
        usdinr_pct = (usdinr_change / t2['usdinr'] * 100) if t2['usdinr'] else 0
        usdinr_section = f"""### USD/INR Exchange Rate
- Day T-2: {t2['usdinr']:.4f}
- Day T-1: {t1['usdinr']:.4f}
- Change: {usdinr_change:+.4f} ({usdinr_pct:+.2f}%)"""
    else:
        # Blind mode - don't show actual price, only reference point
        usdinr_section = f"""### USD/INR Exchange Rate
- **PRICE DATA HIDDEN** - You must predict based on other market indicators
- Last known reference price: {last_known_price:.4f} (from {WARMUP_DAYS} days ago)
- You need to estimate the current rate based on the indicators below"""

    # Detect volume/mention spikes
    india_mention_change = ((t1['india_mentions'] - t2['india_mentions']) / t2['india_mentions'] * 100) if t2['india_mentions'] else 0
    us_mention_change = ((t1['us_mentions'] - t2['us_mentions']) / t2['us_mentions'] * 100) if t2['us_mentions'] else 0

    volume_alerts = []
    if abs(india_mention_change) > 20:
        direction = "spike" if india_mention_change > 0 else "drop"
        volume_alerts.append(f"- **ALERT**: India news mentions {direction} ({india_mention_change:+.1f}%)")
    if abs(us_mention_change) > 20:
        direction = "spike" if us_mention_change > 0 else "drop"
        volume_alerts.append(f"- **ALERT**: US news mentions {direction} ({us_mention_change:+.1f}%)")
    if abs(t1['india_panic'] - t2['india_panic']) > 0.02:
        direction = "increasing" if t1['india_panic'] > t2['india_panic'] else "decreasing"
        volume_alerts.append(f"- **ALERT**: India panic index {direction}")
    if abs(t1['us_panic'] - t2['us_panic']) > 0.02:
        direction = "increasing" if t1['us_panic'] > t2['us_panic'] else "decreasing"
        volume_alerts.append(f"- **ALERT**: US panic index {direction}")

    volume_alert_section = "\n".join(volume_alerts) if volume_alerts else "- No significant volume or panic alerts"

    prompt = f"""
## MARKET DATA (Past 2 Trading Days)

**IMPORTANT: You are in a backtesting simulation. Base your prediction ONLY on the data provided below.
Do NOT use any external knowledge or attempt to identify the actual date. Treat this as if it's happening in real-time.**

---

{usdinr_section}

### Crude Oil (USD/barrel)
- Day T-2: ${t2['oil']:.2f}
- Day T-1: ${t1['oil']:.2f}
- Change: {oil_change:+.2f} ({oil_pct:+.2f}%)

### Gold (USD/oz)
- Day T-2: ${t2['gold']:.2f}
- Day T-1: ${t1['gold']:.2f}
- Change: {gold_change:+.2f} ({gold_pct:+.2f}%)

### US 10-Year Treasury Yield
- Day T-2: {t2['us10y']:.3f}%
- Day T-1: {t1['us10y']:.3f}%
- Change: {us10y_change:+.3f}%

### Dollar Index (DXY)
- Day T-2: {t2['dxy']:.2f}
- Day T-1: {t1['dxy']:.2f}
- Change: {dxy_change:+.2f} ({dxy_pct:+.2f}%)

---

### GDELT News Sentiment Analysis

**India News Sentiment:**
| Metric | Day T-2 | Day T-1 | Change |
|--------|---------|---------|--------|
| Average Tone | {t2['india_tone']:.3f} | {t1['india_tone']:.3f} | {t1['india_tone']-t2['india_tone']:+.3f} |
| Stability Index | {t2['india_stability']:.3f} | {t1['india_stability']:.3f} | {t1['india_stability']-t2['india_stability']:+.3f} |
| Total Mentions | {int(t2['india_mentions']):,} | {int(t1['india_mentions']):,} | {india_mention_change:+.1f}% |
| Panic Index | {t2['india_panic']:.3f} | {t1['india_panic']:.3f} | {t1['india_panic']-t2['india_panic']:+.3f} |

**US News Sentiment:**
| Metric | Day T-2 | Day T-1 | Change |
|--------|---------|---------|--------|
| Average Tone | {t2['us_tone']:.3f} | {t1['us_tone']:.3f} | {t1['us_tone']-t2['us_tone']:+.3f} |
| Stability Index | {t2['us_stability']:.3f} | {t1['us_stability']:.3f} | {t1['us_stability']-t2['us_stability']:+.3f} |
| Total Mentions | {int(t2['us_mentions']):,} | {int(t1['us_mentions']):,} | {us_mention_change:+.1f}% |
| Panic Index | {t2['us_panic']:.3f} | {t1['us_panic']:.3f} | {t1['us_panic']-t2['us_panic']:+.3f} |

**Volume & Panic Alerts:**
{volume_alert_section}

*Note: Tone ranges from -10 (very negative) to +10 (very positive).
Stability > 0 indicates stabilizing sentiment, < 0 indicates destabilizing.
Panic Index ranges 0-1, higher values indicate more fear/uncertainty.*

---

## YOUR TASK

Based on the above market data and your trading perspective, predict the **USD/INR exchange rate for TODAY (Day T)**.

Note:
- "Bullish on USD" means expecting INR to weaken (USD/INR going UP)
- "Bearish on USD" means expecting INR to strengthen (USD/INR going DOWN)
- Key factors: DXY movement, oil prices (India imports 80%+ oil), US yields, sentiment shifts

Provide your response in the following JSON format ONLY (no other text):
```json
{{
    "prediction": "bullish" | "bearish" | "neutral",
    "confidence": <1-10>,
    "predicted_usdinr": <number with 4 decimal places>,
    "predicted_change_percent": <number>,
    "reasoning": "<2-3 sentence explanation>"
}}
```
"""
    return prompt

# ============================================================================
# GEMINI API INTERACTION
# ============================================================================

def initialize_gemini(api_key: str) -> genai.GenerativeModel:
    """Initialize Gemini model."""
    genai.configure(api_key=api_key)

    # Use gemini-2.0-flash for better compatibility and faster responses
    # gemini-2.5-pro has thinking mode that may cause issues with structured output
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        generation_config={
            'temperature': 0.7,
            'top_p': 0.9,
            'max_output_tokens': 1024,
        }
    )

    return model


def query_persona(model: genai.GenerativeModel, persona: Dict, market_prompt: str, retries: int = MAX_RETRIES) -> Dict:
    """Query Gemini with a specific persona and market context."""

    system_prompt = persona['system_prompt']
    full_prompt = f"{system_prompt}\n\n{market_prompt}"

    for attempt in range(retries):
        try:
            response = model.generate_content(full_prompt)
            
            # Extract text from response - handle various Gemini response formats
            response_text = ""
            
            # Method 1: Try simple .text accessor
            try:
                response_text = response.text
            except (ValueError, AttributeError):
                pass
            
            # Method 2: Try accessing parts directly
            if not response_text and hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
            
            # Method 3: Try accessing through candidates (Gemini 2.5 Pro thinking model)
            if not response_text and hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            # Skip thinking parts, only get text parts
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
            
            # Debug: show what we got
            if not response_text.strip():
                # Try to inspect the response structure
                print(f"    Empty response. Inspecting response structure...")
                if hasattr(response, 'candidates') and response.candidates:
                    print(f"    Candidates: {len(response.candidates)}")
                    for i, cand in enumerate(response.candidates):
                        # Check for finish reason
                        if hasattr(cand, 'finish_reason'):
                            print(f"    Candidate {i} finish_reason: {cand.finish_reason}")
                        if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                            print(f"    Candidate {i} parts: {len(cand.content.parts)}")
                            for j, part in enumerate(cand.content.parts):
                                print(f"      Part {j}: type={type(part)}, has_text={hasattr(part, 'text')}")
                                if hasattr(part, 'thought') and part.thought:
                                    print(f"        (this is a thought/thinking part)")
                # Check for prompt feedback (safety blocks)
                if hasattr(response, 'prompt_feedback'):
                    print(f"    Prompt feedback: {response.prompt_feedback}")
                continue
                
            parsed = parse_prediction_response(response_text)

            if parsed:
                return {
                    'success': True,
                    'persona_id': persona['id'],
                    'persona_name': persona['name'],
                    'weight': persona['weight'],
                    'raw_response': response_text,
                    **parsed
                }
            else:
                print(f"    Failed to parse response for {persona['short_name']}, attempt {attempt + 1}")

        except Exception as e:
            print(f"    API error for {persona['short_name']}: {e}, attempt {attempt + 1}")
            import traceback
            traceback.print_exc()
            time.sleep(API_DELAY_SECONDS * 2)

    # Return default neutral prediction on failure
    return {
        'success': False,
        'persona_id': persona['id'],
        'persona_name': persona['name'],
        'weight': persona['weight'],
        'prediction': 'neutral',
        'confidence': 5,
        'predicted_usdinr': None,
        'predicted_change_percent': 0,
        'reasoning': 'Failed to get prediction from API',
        'raw_response': None
    }


def parse_prediction_response(response_text: str) -> Optional[Dict]:
    """Parse JSON prediction from model response."""
    try:
        # Clean up the response - remove markdown code blocks
        cleaned = response_text.strip()
        
        # Remove ```json and ``` markers if present
        if '```json' in cleaned:
            cleaned = cleaned.split('```json')[1]
        if '```' in cleaned:
            cleaned = cleaned.split('```')[0]
        cleaned = cleaned.strip()
        
        # Try to find JSON by matching opening/closing braces properly
        start_idx = cleaned.find('{')
        if start_idx == -1:
            print(f"      No JSON found in response: {response_text[:200]}...")
            return None
            
        # Find matching closing brace
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
            print(f"      Unbalanced braces in response: {response_text[:200]}...")
            return None
            
        json_str = cleaned[start_idx:end_idx]
        parsed = json.loads(json_str)

        # Validate required fields - check multiple possible field names
        usdinr_value = parsed.get('predicted_usdinr') or parsed.get('predictedUsdInr') or parsed.get('predicted_usd_inr')
        if usdinr_value:
            return {
                'prediction': parsed.get('prediction', 'neutral'),
                'confidence': int(parsed.get('confidence', 5)),
                'predicted_usdinr': float(usdinr_value),
                'predicted_change_percent': float(parsed.get('predicted_change_percent', parsed.get('predictedChangePercent', 0))),
                'reasoning': parsed.get('reasoning', '')
            }
        else:
            print(f"      Missing predicted_usdinr. Parsed keys: {list(parsed.keys())}")
            print(f"      Full parsed: {parsed}")
    except (json.JSONDecodeError, ValueError) as e:
        # Debug: print what we got
        print(f"      Parse error: {e}")
        print(f"      Response snippet: {response_text[:200]}...")

    return None

# ============================================================================
# WEIGHTED AGGREGATION
# ============================================================================

def calculate_weighted_prediction(predictions: List[Dict], last_close: float) -> Dict:
    """
    Calculate weighted average prediction from all personas.
    Uses weights from persona definitions.
    """
    valid_predictions = [p for p in predictions if p.get('success') and p.get('predicted_usdinr')]

    if not valid_predictions:
        return {
            'weighted_price': last_close,
            'weighted_change_pct': 0,
            'total_weight_used': 0,
            'num_valid_predictions': 0,
            'consensus': 'neutral',
            'avg_confidence': 0
        }

    # Normalize weights for valid predictions
    total_weight = sum(p['weight'] for p in valid_predictions)

    weighted_price = sum(
        p['predicted_usdinr'] * (p['weight'] / total_weight)
        for p in valid_predictions
    )

    weighted_change = ((weighted_price - last_close) / last_close) * 100 if last_close else 0

    # Determine consensus
    bullish = sum(p['weight'] for p in valid_predictions if p['prediction'] == 'bullish')
    bearish = sum(p['weight'] for p in valid_predictions if p['prediction'] == 'bearish')

    if bullish > bearish + 0.1:
        consensus = 'bullish'
    elif bearish > bullish + 0.1:
        consensus = 'bearish'
    else:
        consensus = 'neutral'

    avg_confidence = sum(p['confidence'] * p['weight'] for p in valid_predictions) / total_weight

    return {
        'weighted_price': weighted_price,
        'weighted_change_pct': weighted_change,
        'total_weight_used': total_weight,
        'num_valid_predictions': len(valid_predictions),
        'consensus': consensus,
        'avg_confidence': avg_confidence
    }

# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def run_simulation(api_key: str, personas_file: str = PERSONAS_FILE):
    """Run the full market simulation."""

    print("=" * 60)
    print("GEMINI USD/INR FOREX PERSONA SIMULATION")
    print("=" * 60)
    print(f"Period: {SIMULATION_START.date()} to {SIMULATION_END.date()}")
    print()

    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load personas
    personas_path = os.path.join(script_dir, personas_file)
    personas = load_personas(personas_path)
    print(f"Loaded {len(personas)} personas")

    # Load market data from local CSV
    data_path = os.path.join(script_dir, DATA_FILE)
    market_df = load_market_data(data_path)

    # Initialize Gemini
    print("\nInitializing Gemini 2.0 Flash...")
    model = initialize_gemini(api_key)

    # Get trading days
    trading_days = get_trading_days(market_df, SIMULATION_START, SIMULATION_END)
    print(f"\nSimulating {len(trading_days)} trading days")
    print()

    # Results storage
    all_results = []

    # Track the last known price (from warmup period)
    last_known_price = None

    # Run simulation for each trading day
    for day_idx, target_date in enumerate(trading_days):
        # Determine if we're in warmup period
        is_warmup = day_idx < WARMUP_DAYS
        mode_str = "WARMUP (price visible)" if is_warmup else "BLIND (price hidden)"

        print(f"\n[Day {day_idx + 1}/{len(trading_days)}] {target_date.date()} - {mode_str}")
        print("-" * 50)

        # Get market context (T-2, T-1 data)
        context = get_market_context(market_df, target_date)

        if not context:
            print("  Skipping - insufficient historical data")
            continue

        # Get last known close for reference (USD/INR) - we use this internally
        last_close = context['t1']['usdinr']

        # During warmup, update the last known price
        if is_warmup:
            last_known_price = last_close

        # Format prompt - hide price after warmup
        if is_warmup:
            market_prompt = format_market_prompt(context, show_price=True)
        else:
            # Use the last known price from end of warmup as reference
            market_prompt = format_market_prompt(context, show_price=False, last_known_price=last_known_price)

        # Query each persona
        day_predictions = []

        for persona in personas:
            print(f"  Querying {persona['short_name']}...", end=" ")

            prediction = query_persona(model, persona, market_prompt)
            day_predictions.append(prediction)

            if prediction['success']:
                print(f"{prediction['prediction']} @ {prediction['predicted_usdinr']:.4f} (conf: {prediction['confidence']})")
            else:
                print("FAILED")

            time.sleep(API_DELAY_SECONDS)

        # Calculate weighted prediction
        weighted_result = calculate_weighted_prediction(day_predictions, last_close)

        # Get actual USD/INR price
        actual_price = get_actual_price(market_df, target_date)

        # Calculate error if we have actual price
        if actual_price and weighted_result['weighted_price']:
            prediction_error = weighted_result['weighted_price'] - actual_price
            prediction_error_pct = (prediction_error / actual_price) * 100
        else:
            prediction_error = None
            prediction_error_pct = None

        # Store results
        day_result = {
            'date': target_date.isoformat(),
            'last_close': last_close,
            'actual_price': actual_price,
            'weighted_prediction': weighted_result['weighted_price'],
            'weighted_change_pct': weighted_result['weighted_change_pct'],
            'prediction_error': prediction_error,
            'prediction_error_pct': prediction_error_pct,
            'consensus': weighted_result['consensus'],
            'avg_confidence': weighted_result['avg_confidence'],
            'num_valid_predictions': weighted_result['num_valid_predictions'],
            'persona_predictions': day_predictions
        }

        all_results.append(day_result)

        # Print summary
        print(f"\n  Summary:")
        print(f"    Previous USD/INR: {last_close:.4f}")
        print(f"    Weighted Prediction: {weighted_result['weighted_price']:.4f} ({weighted_result['weighted_change_pct']:+.2f}%)")
        print(f"    Consensus: {weighted_result['consensus'].upper()} (avg confidence: {weighted_result['avg_confidence']:.1f})")
        if actual_price:
            print(f"    Actual USD/INR: {actual_price:.4f}")
            if prediction_error_pct:
                print(f"    Prediction Error: {prediction_error_pct:+.3f}%")

        # Save intermediate results
        save_results(all_results, script_dir)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

    # Generate summary statistics
    generate_summary(all_results, script_dir)

    return all_results


def save_results(results: List[Dict], output_dir: str):
    """Save results to JSON file."""
    output_path = os.path.join(output_dir, RESULTS_FILE)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def generate_summary(results: List[Dict], output_dir: str):
    """Generate summary statistics and CSV."""

    # Create summary dataframe
    summary_data = []

    for r in results:
        summary_data.append({
            'date': r['date'],
            'previous_close': r['last_close'],
            'predicted_price': r['weighted_prediction'],
            'predicted_change_pct': r['weighted_change_pct'],
            'actual_price': r['actual_price'],
            'prediction_error': r['prediction_error'],
            'prediction_error_pct': r['prediction_error_pct'],
            'consensus': r['consensus'],
            'avg_confidence': r['avg_confidence']
        })

    df = pd.DataFrame(summary_data)

    # Save CSV
    csv_path = os.path.join(output_dir, SUMMARY_CSV)
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    # Print statistics
    valid_results = df[df['prediction_error_pct'].notna()]

    if not valid_results.empty:
        print("\nPerformance Metrics:")
        print(f"  Mean Absolute Error:  {valid_results['prediction_error_pct'].abs().mean():.3f}%")
        print(f"  Mean Error (Bias):    {valid_results['prediction_error_pct'].mean():+.3f}%")
        print(f"  Max Overestimate:     {valid_results['prediction_error_pct'].max():+.3f}%")
        print(f"  Max Underestimate:    {valid_results['prediction_error_pct'].min():+.3f}%")

        # Direction accuracy
        correct_direction = sum(
            1 for _, row in valid_results.iterrows()
            if (row['predicted_change_pct'] > 0 and row['actual_price'] > row['previous_close']) or
               (row['predicted_change_pct'] < 0 and row['actual_price'] < row['previous_close']) or
               (abs(row['predicted_change_pct']) < 0.05 and abs(row['actual_price'] - row['previous_close']) < 0.05)
        )
        print(f"  Direction Accuracy:   {correct_direction}/{len(valid_results)} ({100*correct_direction/len(valid_results):.1f}%)")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Get API key from environment
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
        raise

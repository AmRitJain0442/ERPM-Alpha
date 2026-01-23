"""
Gemini Daily News-Based Prediction System
==========================================
Analyzes actual GDELT news articles day-by-day to predict exchange rates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import json
import os
from pathlib import Path

# Configuration
GEMINI_API_KEY = "AIzaSyCLRRUW4uDP4km_aHYBCehNaZpD7dsQnMg"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def call_gemini_api(prompt, max_tokens=2000, retry=3):
    """Call Gemini API with retry logic."""
    import time
    
    headers = {"Content-Type": "application/json"}
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens
        }
    }
    
    for attempt in range(retry):
        try:
            # Add delay to avoid rate limiting
            if attempt > 0:
                time.sleep(2 ** attempt)  # Exponential backoff
            
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code == 429:  # Rate limit
                print(f"  Rate limited, retrying...")
                time.sleep(5)
                continue
            else:
                print(f"  API Error {response.status_code}: {response.text[:200]}")
                if attempt < retry - 1:
                    continue
                return None
        except Exception as e:
            print(f"  Exception (attempt {attempt+1}/{retry}): {str(e)[:100]}")
            if attempt < retry - 1:
                time.sleep(2)
                continue
            return None
    
    return None


def get_news_for_date_range(news_df, target_date, lookback_days=2):
    """Get news from the past N days before target date."""
    target = pd.to_datetime(target_date)
    start_date = target - timedelta(days=lookback_days)
    
    # Filter news within date range
    mask = (news_df['Date'] >= start_date) & (news_df['Date'] < target)
    filtered = news_df[mask].copy()
    
    return filtered


def analyze_news_with_gemini(news_df, current_price, target_date):
    """Analyze news articles and predict next-day exchange rate."""
    
    if len(news_df) == 0:
        return {
            'prediction': current_price,
            'confidence': 'low',
            'direction': 'stable',
            'change_percent': 0.0,
            'reasoning': 'No news data available',
            'date': target_date
        }
    
    # Prepare news summary
    news_summary = []
    for idx, row in news_df.head(min(10, len(news_df))).iterrows():
        try:
            date_str = row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])
            
            # Extract key information from aggregated data
            summary_parts = []
            
            # Economic sentiment
            if 'Tone_Economy' in row and pd.notna(row['Tone_Economy']):
                tone_econ = float(row['Tone_Economy'])
                summary_parts.append(f"Economy: {'Positive' if tone_econ > 0 else 'Negative'} tone ({tone_econ:.1f})")
            
            # Conflict
            if 'Tone_Conflict' in row and pd.notna(row['Tone_Conflict']):
                tone_conflict = float(row['Tone_Conflict'])
                summary_parts.append(f"Conflict: {abs(tone_conflict):.1f}")
            
            # Goldstein scale
            goldstein = 0
            if 'Goldstein_Avg' in row and pd.notna(row['Goldstein_Avg']):
                goldstein = float(row['Goldstein_Avg'])
                summary_parts.append(f"Cooperation index: {goldstein:.2f}")
            
            # Event counts
            if 'Count_Total' in row and pd.notna(row['Count_Total']):
                count = int(row['Count_Total'])
                summary_parts.append(f"{count} events")
            
            summary = "; ".join(summary_parts) if summary_parts else "Daily news aggregates"
            
            # Tone for sorting
            tone = float(row.get('Tone_Overall', row.get('Tone_Economy', 0)))
            
            news_summary.append({
                'date': date_str,
                'summary': summary,
                'tone': tone,
                'goldstein': goldstein,
                'url': f"Aggregated daily news for {date_str}"
            })
        except Exception as e:
            # Skip problematic rows
            continue
    
    # Limit to top 15 most impactful news (by tone extremity)
    news_summary = sorted(news_summary, key=lambda x: abs(x['tone']), reverse=True)[:15]
    
    # Build prompt
    news_text = "\n".join([
        f"• [{n['date']}] Tone:{n['tone']:.1f}, Goldstein:{n['goldstein']:.1f}\n"
        f"  {n['summary']}\n"
        f"  URL: {n['url']}"
        for n in news_summary
    ])
    
    prompt = f"""You are an expert forex analyst specializing in USD/INR exchange rates. Analyze the following recent GDELT news events affecting India-US relations and predict tomorrow's exchange rate.

CURRENT RATE: {current_price:.4f} INR per USD
TARGET DATE: {target_date.strftime('%Y-%m-%d')}

RECENT NEWS (Past 2 Days):
{news_text}

ANALYSIS GUIDE:
- Tone: Negative values (<0) indicate negative sentiment, positive values (>0) indicate positive sentiment
- Goldstein Scale: Range -10 to +10, measures cooperation vs conflict (-10=extreme conflict, +10=extreme cooperation)
- Negative economic news typically strengthens USD (increases USD/INR)
- Positive India news or US weakness typically strengthens INR (decreases USD/INR)

Provide your prediction in EXACTLY this format:

PREDICTION: [exact number for tomorrow's USD/INR rate]
DIRECTION: [increase/decrease/stable]
CHANGE_PERCENT: [expected % change, e.g., +0.25 or -0.15]
CONFIDENCE: [high/medium/low]
REASONING: [2-3 sentences explaining your prediction based on the news sentiment, key events, and their likely impact on the exchange rate. Be specific about which news items influenced your decision and why.]

Use exact format above."""

    print(f"  Analyzing {len(news_summary)} news articles for {target_date.strftime('%Y-%m-%d')}...")
    response = call_gemini_api(prompt, max_tokens=1000)
    
    if not response:
        return {
            'prediction': current_price,
            'confidence': 'low',
            'direction': 'stable',
            'change_percent': 0.0,
            'reasoning': 'Gemini API unavailable',
            'date': target_date
        }
    
    # Parse response
    result = {
        'prediction': current_price,
        'confidence': 'medium',
        'direction': 'stable',
        'change_percent': 0.0,
        'reasoning': '',
        'date': target_date,
        'raw_response': response
    }
    
    lines = response.split('\n')
    for line in lines:
        line_upper = line.upper()
        if 'PREDICTION:' in line_upper:
            try:
                pred_str = line.split(':', 1)[1].strip()
                import re
                numbers = re.findall(r'\d+\.?\d*', pred_str)
                if numbers:
                    result['prediction'] = float(numbers[0])
            except:
                pass
        elif 'DIRECTION:' in line_upper:
            value = line.split(':', 1)[1].strip().lower()
            if 'increase' in value or 'rise' in value or 'up' in value:
                result['direction'] = 'increase'
            elif 'decrease' in value or 'fall' in value or 'down' in value:
                result['direction'] = 'decrease'
            else:
                result['direction'] = 'stable'
        elif 'CHANGE_PERCENT:' in line_upper or 'CHANGE PERCENT:' in line_upper:
            try:
                value = line.split(':', 1)[1].strip()
                import re
                numbers = re.findall(r'[+-]?\d+\.?\d*', value)
                if numbers:
                    result['change_percent'] = float(numbers[0])
            except:
                pass
        elif 'CONFIDENCE:' in line_upper:
            value = line.split(':', 1)[1].strip().lower()
            if 'high' in value:
                result['confidence'] = 'high'
            elif 'low' in value:
                result['confidence'] = 'low'
            else:
                result['confidence'] = 'medium'
        elif 'REASONING:' in line_upper:
            reasoning_lines = [line.split(':', 1)[1].strip()]
            # Continue collecting reasoning
            idx = lines.index(line) + 1
            while idx < len(lines) and lines[idx].strip() and not ':' in lines[idx][:20]:
                reasoning_lines.append(lines[idx].strip())
                idx += 1
            result['reasoning'] = ' '.join(reasoning_lines)
    
    # If no reasoning extracted, use shortened response
    if not result['reasoning'] or len(result['reasoning']) < 20:
        result['reasoning'] = response[:200] + '...'
    
    return result


def run_daily_predictions(news_df, exchange_df, lookback_days=2, prediction_days=10):
    """Run day-by-day predictions for the past N days with checkpoint support."""
    
    print(f"\n{'='*70}")
    print(f"  GEMINI DAILY PREDICTION SYSTEM")
    print(f"  Analyzing past {prediction_days} days with {lookback_days}-day news lookback")
    print(f"{'='*70}\n")
    
    # Check for existing checkpoint
    checkpoint_file = output_dir / 'gemini_checkpoint.csv'
    completed_predictions = {}
    
    if checkpoint_file.exists():
        print(f"  Loading checkpoint file...")
        try:
            checkpoint_df = pd.read_csv(checkpoint_file)
            for _, row in checkpoint_df.iterrows():
                completed_predictions[str(row['date'])] = row.to_dict()
            print(f"  Found {len(completed_predictions)} completed predictions\n")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}\n")
    
    # Get the last N days
    latest_date = exchange_df['Date'].max()
    start_date = latest_date - timedelta(days=prediction_days)
    
    # Filter exchange rates
    date_range = pd.date_range(start=start_date, end=latest_date, freq='D')
    
    predictions = []
    
    for i, target_date in enumerate(date_range):
        date_str = target_date.strftime('%Y-%m-%d')
        
        # Skip if already completed
        if date_str in completed_predictions:
            print(f"  [{date_str}] Skipping (already completed)")
            predictions.append(completed_predictions[date_str])
            continue
        
        # Get previous day's rate
        prev_date = target_date - timedelta(days=1)
        prev_rate = exchange_df[exchange_df['Date'] <= prev_date]['USD_to_INR'].iloc[-1] if len(exchange_df[exchange_df['Date'] <= prev_date]) > 0 else 85.0
        
        # Get news for this prediction
        news_for_pred = get_news_for_date_range(news_df, target_date, lookback_days)
        
        # Get Gemini prediction
        pred_result = analyze_news_with_gemini(news_for_pred, prev_rate, target_date)
        
        # Get actual rate if available
        actual_data = exchange_df[exchange_df['Date'] == target_date]
        if len(actual_data) > 0:
            pred_result['actual'] = actual_data['USD_to_INR'].iloc[0]
            pred_result['error'] = pred_result['actual'] - pred_result['prediction']
        else:
            pred_result['actual'] = None
            pred_result['error'] = None
        
        predictions.append(pred_result)
        
        # Save checkpoint after each prediction
        checkpoint_df = pd.DataFrame(predictions)
        checkpoint_df.to_csv(checkpoint_file, index=False)
        
        direction_symbol = "UP" if pred_result['direction'] == 'increase' else "DOWN" if pred_result['direction'] == 'decrease' else "FLAT"
        print(f"  [{target_date.strftime('%Y-%m-%d')}] Predicted: {pred_result['prediction']:.4f} "
              f"({direction_symbol}) "
              f"{pred_result['change_percent']:+.2f}% "
              f"[{pred_result['confidence']}]"
              + (f" | Actual: {pred_result['actual']:.4f} | Error: {pred_result['error']:+.4f}" if pred_result['actual'] is not None else ""))
        
        # Rate limiting - wait between API calls (except for last one)
        if i < len(date_range) - 1:
            wait_time = 8  # 8 seconds between calls
            print(f"  Waiting {wait_time}s before next API call...")
            time.sleep(wait_time)
    
    return predictions


def create_visualizations(predictions, exchange_df, ultimate_forecast=None):
    """Create comprehensive visualizations."""
    
    pred_df = pd.DataFrame(predictions)
    
    # Calculate metrics
    valid_preds = pred_df[pred_df['actual'].notna()].copy()
    if len(valid_preds) > 0:
        mae = abs(valid_preds['error']).mean()
        rmse = np.sqrt((valid_preds['error'] ** 2).mean())
        print(f"\n{'='*70}")
        print(f"  GEMINI PREDICTION METRICS")
        print(f"{'='*70}")
        print(f"  MAE: {mae:.4f} INR")
        print(f"  RMSE: {rmse:.4f} INR")
        print(f"  Predictions: {len(valid_preds)}")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Gemini Predictions vs Actual
    ax1 = fig.add_subplot(gs[0, :])
    dates = pred_df['date']
    
    # Plot actual rates (full history)
    hist_mask = exchange_df['Date'] <= dates.max()
    hist_data = exchange_df[hist_mask].tail(30)
    ax1.plot(hist_data['Date'], hist_data['USD_to_INR'], 'b-', linewidth=2, label='Historical', alpha=0.6)
    
    # Plot Gemini predictions
    ax1.plot(dates, pred_df['prediction'], 'r--', marker='o', linewidth=2, label='Gemini Prediction', markersize=6)
    
    # Plot actual for prediction period
    actual_mask = pred_df['actual'].notna()
    if actual_mask.any():
        ax1.scatter(pred_df.loc[actual_mask, 'date'], pred_df.loc[actual_mask, 'actual'], 
                   c='green', s=100, marker='s', label='Actual', zorder=5, edgecolors='black', linewidth=1.5)
    
    ax1.set_title('Gemini AI Daily Predictions (Based on 2-Day News Analysis)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('USD/INR Exchange Rate')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Prediction Error Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    if len(valid_preds) > 0:
        ax2.hist(valid_preds['error'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Prediction Error Distribution', fontweight='bold')
        ax2.set_xlabel('Error (INR)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Confidence Level Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    confidence_counts = pred_df['confidence'].value_counts()
    colors = {'high': 'green', 'medium': 'orange', 'low': 'red'}
    bars = ax3.bar(confidence_counts.index, confidence_counts.values, 
                   color=[colors.get(c, 'gray') for c in confidence_counts.index], edgecolor='black')
    ax3.set_title('Gemini Confidence Distribution', fontweight='bold')
    ax3.set_xlabel('Confidence Level')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Direction Accuracy
    ax4 = fig.add_subplot(gs[2, 0])
    direction_data = []
    for idx, row in valid_preds.iterrows():
        actual_change = row['actual'] - exchange_df[exchange_df['Date'] < row['date']]['USD_to_INR'].iloc[-1]
        pred_direction = row['direction']
        actual_direction = 'increase' if actual_change > 0.01 else ('decrease' if actual_change < -0.01 else 'stable')
        direction_data.append({
            'predicted': pred_direction,
            'actual': actual_direction,
            'correct': pred_direction == actual_direction
        })
    
    if direction_data:
        dir_df = pd.DataFrame(direction_data)
        accuracy = dir_df['correct'].mean() * 100
        
        correct_counts = dir_df.groupby('predicted')['correct'].sum()
        total_counts = dir_df['predicted'].value_counts()
        
        x = range(len(total_counts))
        width = 0.35
        
        ax4.bar([i - width/2 for i in x], [correct_counts.get(idx, 0) for idx in total_counts.index], 
               width, label='Correct', color='green', alpha=0.8, edgecolor='black')
        ax4.bar([i + width/2 for i in x], [total_counts[idx] - correct_counts.get(idx, 0) for idx in total_counts.index],
               width, label='Incorrect', color='red', alpha=0.8, edgecolor='black')
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(total_counts.index)
        ax4.set_title(f'Direction Prediction Accuracy: {accuracy:.1f}%', fontweight='bold')
        ax4.set_xlabel('Predicted Direction')
        ax4.set_ylabel('Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Combined: Gemini + Ultimate Ensemble
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Plot last 30 days historical
    ax5.plot(hist_data['Date'], hist_data['USD_to_INR'], 'b-', linewidth=2, label='Historical', alpha=0.7)
    
    # Gemini predictions
    ax5.plot(dates, pred_df['prediction'], 'r--', marker='o', linewidth=2, 
            label='Gemini Daily', markersize=5, alpha=0.8)
    
    # Add ultimate ensemble forecast if provided
    if ultimate_forecast is not None:
        ax5.plot(ultimate_forecast['dates'], ultimate_forecast['forecast'], 'g-', 
                linewidth=2.5, label='Ultimate Ensemble', alpha=0.8)
        ax5.fill_between(ultimate_forecast['dates'], 
                         ultimate_forecast['lower'], 
                         ultimate_forecast['upper'], 
                         alpha=0.2, color='green')
    
    ax5.set_title('Combined: Gemini Daily + Ultimate Ensemble', fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('USD/INR')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('GDELT News-Based Daily Predictions with Gemini AI', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(OUTPUT_DIR / 'gemini_daily_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Saved: gemini_daily_analysis.png")
    
    return pred_df


def main():
    """Main execution."""
    base_dir = Path(__file__).parent.parent
    
    # Load news data - use Phase-B processed data for speed
    print("Loading data...")
    news_path = base_dir / 'Phase-B' / 'merged_training_data.csv'
    if not news_path.exists():
        news_path = base_dir / 'india_daily_goldstein_averages.csv'
    
    # Load with low_memory=False
    news_df = pd.read_csv(news_path, parse_dates=['Date'])
    
    print(f"  News: {len(news_df)} daily aggregates")
    
    # Load exchange rates
    exchange_path = base_dir / 'usd_inr_exchange_rates_1year.csv'
    exchange_df = pd.read_csv(exchange_path, parse_dates=['Date'])
    print(f"  Exchange rates: {len(exchange_df)} observations")
    
    # Run predictions
    predictions = run_daily_predictions(news_df, exchange_df, lookback_days=2, prediction_days=7)
    
    # Load ultimate ensemble forecast if available
    ultimate_forecast = None
    ultimate_path = OUTPUT_DIR / 'ultimate_forecast.csv'
    if ultimate_path.exists():
        ult_df = pd.read_csv(ultimate_path)
        latest_date = exchange_df['Date'].max()
        forecast_dates = [latest_date + timedelta(days=i+1) for i in range(len(ult_df))]
        ultimate_forecast = {
            'dates': forecast_dates,
            'forecast': ult_df['Median'].values,
            'lower': ult_df['P5'].values,
            'upper': ult_df['P95'].values
        }
    
    # Create visualizations
    pred_df = create_visualizations(predictions, exchange_df, ultimate_forecast)
    
    # Save results
    pred_df.to_csv(OUTPUT_DIR / 'gemini_daily_predictions.csv', index=False)
    print(f"  ✓ Saved: gemini_daily_predictions.csv")
    
    print(f"\n{'='*70}")
    print(f"  COMPLETE!")
    print(f"{'='*70}\n")
    
    return predictions


if __name__ == "__main__":
    predictions = main()

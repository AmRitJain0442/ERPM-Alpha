"""
V5 Results Analysis - Pure LLM Performance
"""

import json
import os
import numpy as np
import pandas as pd

def analyze_v5_results():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "simulation_results_v5.json")
    
    if not os.path.exists(results_path):
        print("ERROR: simulation_results_v5.json not found")
        print("Run the V5 simulation first: python run_simulation_v5.py")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("=" * 70)
    print("V5 SIMULATION ANALYSIS - PURE LLM PREDICTIONS")
    print("=" * 70)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Filter to LLM-only days (non-warmup)
    llm_only = df[df['mode'] == 'llm_only'].copy()
    
    if llm_only.empty:
        print("No LLM-only days found in results")
        return
    
    valid = llm_only[llm_only['prediction_error_pct'].notna()].copy()
    
    if valid.empty:
        print("No valid predictions found")
        return
    
    print(f"\n### PREDICTION ACCURACY ###\n")
    
    errors = valid['prediction_error_pct']
    print(f"LLM Ensemble:")
    print(f"  MAE:     {errors.abs().mean():.4f}%")
    print(f"  Bias:    {errors.mean():+.4f}%")
    print(f"  Std:     {errors.std():.4f}%")
    print(f"  Max:     {errors.abs().max():.4f}%")
    print(f"  Median:  {errors.abs().median():.4f}%")
    
    # Random walk benchmark
    rw_errors = []
    for _, row in valid.iterrows():
        rw_error = abs(row['last_close'] - row['actual_price']) / row['actual_price'] * 100
        rw_errors.append(rw_error)
    rw_mae = np.mean(rw_errors)
    
    print(f"\nRandom Walk Benchmark:")
    print(f"  MAE:     {rw_mae:.4f}%")
    
    llm_mae = errors.abs().mean()
    if llm_mae < rw_mae:
        improvement = (rw_mae - llm_mae) / rw_mae * 100
        print(f"  ✓ LLM beats random walk by {improvement:.1f}%!")
    else:
        degradation = (llm_mae - rw_mae) / rw_mae * 100
        print(f"  ✗ Random walk is better by {degradation:.1f}%")
    
    # Direction accuracy
    print(f"\n### DIRECTION ACCURACY ###\n")
    
    correct = sum(
        1 for _, row in valid.iterrows()
        if (row['final_prediction'] > row['last_close'] and row['actual_price'] > row['last_close']) or
           (row['final_prediction'] < row['last_close'] and row['actual_price'] < row['last_close'])
    )
    total = len(valid)
    
    print(f"  Direction Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    
    if correct/total > 0.5:
        print(f"  ✓ Above random chance (50%)")
    else:
        print(f"  ✗ Below random chance (50%)")
    
    # Direction balance
    print(f"\n### DIRECTION BALANCE ###\n")
    
    higher_avg = valid['higher_pct'].mean() * 100
    lower_avg = valid['lower_pct'].mean() * 100
    
    print(f"  Avg Higher (bullish USD): {higher_avg:.1f}%")
    print(f"  Avg Lower (bearish USD):  {lower_avg:.1f}%")
    print(f"  Ratio: {higher_avg/(lower_avg+0.01):.2f}:1")
    
    if abs(higher_avg - lower_avg) < 15:
        print(f"  ✓ Reasonably balanced")
    else:
        print(f"  ✗ Imbalanced (>{abs(higher_avg - lower_avg):.0f}% difference)")
    
    # Entropy/confidence analysis
    print(f"\n### CONFIDENCE ANALYSIS ###\n")
    
    if 'entropy_confidence' in valid.columns:
        print(f"  Avg Entropy Confidence: {valid['entropy_confidence'].mean():.3f}")
        print(f"  Std Entropy Confidence: {valid['entropy_confidence'].std():.3f}")
        
        # Performance by confidence level
        high_conf = valid[valid['entropy_confidence'] > 0.6]
        low_conf = valid[valid['entropy_confidence'] <= 0.4]
        
        if len(high_conf) > 5:
            print(f"\n  High confidence days ({len(high_conf)} days):")
            print(f"    MAE: {high_conf['prediction_error_pct'].abs().mean():.4f}%")
        
        if len(low_conf) > 5:
            print(f"\n  Low confidence days ({len(low_conf)} days):")
            print(f"    MAE: {low_conf['prediction_error_pct'].abs().mean():.4f}%")
    
    # Prediction spread analysis
    if 'prediction_std' in valid.columns:
        print(f"\n### PREDICTION SPREAD ###\n")
        print(f"  Avg persona disagreement (std): {valid['prediction_std'].mean():.4f}")
        print(f"  Max disagreement: {valid['prediction_std'].max():.4f}")
    
    # Regime analysis
    if 'regime_volatility' in valid.columns:
        print(f"\n### REGIME-CONDITIONAL PERFORMANCE ###\n")
        
        for regime in ['low', 'normal', 'high']:
            regime_data = valid[valid['regime_volatility'] == regime]
            if len(regime_data) > 0:
                print(f"  {regime.upper()} volatility: MAE={regime_data['prediction_error_pct'].abs().mean():.4f}% ({len(regime_data)} days)")
    
    # Persona breakdown
    print(f"\n### PERSONA BREAKDOWN ###\n")
    
    persona_stats = {}
    for _, row in valid.iterrows():
        if 'persona_predictions' not in row or not row['persona_predictions']:
            continue
        for p in row['persona_predictions']:
            pid = p.get('persona_id', p.get('short_name', 'unknown'))
            if pid not in persona_stats:
                persona_stats[pid] = {
                    'predictions': [],
                    'directions': [],
                    'confidences': [],
                }
            if p.get('success'):
                pred = p.get('predicted_rate')
                actual = row['actual_price']
                if pred and actual:
                    error = abs(pred - actual) / actual * 100
                    persona_stats[pid]['predictions'].append(error)
                    persona_stats[pid]['directions'].append(p.get('direction'))
                    persona_stats[pid]['confidences'].append(p.get('confidence', 5))
    
    if persona_stats:
        print(f"{'Persona':<18} {'MAE':>8} {'AvgConf':>8} {'Higher%':>8} {'Lower%':>8}")
        print("-" * 60)
        
        for pid, stats in sorted(persona_stats.items()):
            if len(stats['predictions']) < 5:
                continue
            mae = np.mean(stats['predictions'])
            avg_conf = np.mean(stats['confidences'])
            higher_pct = sum(1 for d in stats['directions'] if d == 'higher') / len(stats['directions']) * 100
            lower_pct = sum(1 for d in stats['directions'] if d == 'lower') / len(stats['directions']) * 100
            print(f"{pid:<18} {mae:>7.3f}% {avg_conf:>8.1f} {higher_pct:>7.1f}% {lower_pct:>7.1f}%")
    
    # Compare with V4 if available
    v4_path = os.path.join(script_dir, "simulation_results_v4.json")
    if os.path.exists(v4_path):
        print(f"\n" + "=" * 70)
        print("V5 vs V4 COMPARISON")
        print("=" * 70)
        
        with open(v4_path, 'r') as f:
            v4_results = json.load(f)
        
        v4_df = pd.DataFrame(v4_results)
        v4_hybrid = v4_df[v4_df['mode'] == 'hybrid']
        v4_valid = v4_hybrid[v4_hybrid['prediction_error_pct'].notna()]
        
        if not v4_valid.empty:
            v4_mae = v4_valid['prediction_error_pct'].abs().mean()
            v5_mae = errors.abs().mean()
            
            v4_dir = sum(
                1 for _, row in v4_valid.iterrows()
                if (row['final_prediction'] > row['last_close'] and row['actual_price'] > row['last_close']) or
                   (row['final_prediction'] < row['last_close'] and row['actual_price'] < row['last_close'])
            ) / len(v4_valid) * 100
            
            v5_dir = correct / total * 100
            
            print(f"\n{'Metric':<30} {'V4 (Hybrid)':>15} {'V5 (Pure LLM)':>15} {'Winner':>10}")
            print("-" * 75)
            print(f"{'MAE':<30} {v4_mae:>14.4f}% {v5_mae:>14.4f}% {'V5' if v5_mae < v4_mae else 'V4':>10}")
            print(f"{'Direction Accuracy':<30} {v4_dir:>14.1f}% {v5_dir:>14.1f}% {'V5' if v5_dir > v4_dir else 'V4':>10}")
            print(f"{'Higher/Bullish %':<30} {v4_valid['higher_pct'].mean()*100:>14.1f}% {higher_avg:>14.1f}% {'-':>10}")
            
            print(f"\n")
            if v5_mae < v4_mae:
                print("✓ V5 (Pure LLM) has LOWER MAE than V4 (Hybrid)")
            else:
                print("✗ V4 (Hybrid) has lower MAE than V5 (Pure LLM)")
            
            if v5_dir > v4_dir:
                print("✓ V5 (Pure LLM) has BETTER direction accuracy")
            else:
                print("✗ V4 (Hybrid) has better direction accuracy")


if __name__ == "__main__":
    analyze_v5_results()

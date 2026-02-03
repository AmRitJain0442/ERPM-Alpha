"""
Analyze V4 results and compare with V3.

V4 improvements to validate:
1. Is higher/lower ratio closer to 50/50? (vs 66/34 in V3)
2. Does LLM help more often? (>50% target)
3. Is MAE lower than stat-only baseline?
4. Is entropy confidence metric useful?
5. Is bias correction working?
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List


def load_results(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_v4(results: List[Dict]) -> pd.DataFrame:
    """Comprehensive V4 analysis."""
    
    hybrid = [r for r in results if r.get('mode') == 'hybrid' and r.get('actual_price')]
    
    if not hybrid:
        print("No hybrid results found.")
        return None
    
    print("=" * 70)
    print("V4 SIMULATION ANALYSIS - DEBIASED ROBUST ENSEMBLE")
    print("=" * 70)
    
    # =========================================================================
    # PREDICTION ACCURACY
    # =========================================================================
    print("\n### PREDICTION ACCURACY ###")
    
    errors = [r['prediction_error_pct'] for r in hybrid if r.get('prediction_error_pct') is not None]
    stat_errors = [r['stat_error_pct'] for r in hybrid if r.get('stat_error_pct') is not None]
    
    ensemble_mae = np.mean(np.abs(errors))
    stat_mae = np.mean(np.abs(stat_errors))
    
    print(f"\nEnsemble Model (Stat + LLM):")
    print(f"  MAE:     {ensemble_mae:.4f}%")
    print(f"  Bias:    {np.mean(errors):+.4f}%")
    print(f"  Std:     {np.std(errors):.4f}%")
    print(f"  Max:     {np.max(np.abs(errors)):.4f}%")
    print(f"  Median:  {np.median(np.abs(errors)):.4f}%")
    
    print(f"\nStatistical Model Only:")
    print(f"  MAE:     {stat_mae:.4f}%")
    print(f"  Bias:    {np.mean(stat_errors):+.4f}%")
    
    # Random walk benchmark (yesterday's price predicts today)
    rw_errors = []
    for r in hybrid:
        if r.get('actual_price') and r.get('last_close'):
            rw_error = (r['last_close'] - r['actual_price']) / r['actual_price'] * 100
            rw_errors.append(rw_error)
    
    if rw_errors:
        rw_mae = np.mean(np.abs(rw_errors))
        print(f"\nRandom Walk Benchmark:")
        print(f"  MAE:     {rw_mae:.4f}%")
        
        if ensemble_mae < rw_mae:
            improvement = (rw_mae - ensemble_mae) / rw_mae * 100
            print(f"  Ensemble beats random walk by {improvement:.1f}%")
        else:
            print(f"  WARNING: Random walk is better!")
    
    # =========================================================================
    # LLM CONTRIBUTION
    # =========================================================================
    print(f"\n### LLM CONTRIBUTION ###")
    
    llm_helped = sum(
        1 for r in hybrid
        if r.get('prediction_error_pct') is not None and r.get('stat_error_pct') is not None
        and abs(r['prediction_error_pct']) < abs(r['stat_error_pct'])
    )
    total = sum(
        1 for r in hybrid
        if r.get('prediction_error_pct') is not None and r.get('stat_error_pct') is not None
    )
    
    llm_help_pct = 100 * llm_helped / total if total > 0 else 0
    print(f"  LLM helped in {llm_helped}/{total} cases ({llm_help_pct:.1f}%)")
    
    if llm_help_pct > 50:
        print(f"  ✓ TARGET MET: LLM helps >50% of cases")
    else:
        print(f"  ✗ TARGET MISSED: LLM helps <50% of cases")
    
    if ensemble_mae < stat_mae:
        improvement = (stat_mae - ensemble_mae) / stat_mae * 100
        print(f"\n  LLM IMPROVED overall MAE by {improvement:.2f}%")
        print(f"  ✓ Ensemble is better than stat-only")
    else:
        degradation = (ensemble_mae - stat_mae) / stat_mae * 100
        print(f"\n  LLM DEGRADED overall MAE by {degradation:.2f}%")
        print(f"  ✗ Stat-only would be better")
    
    # =========================================================================
    # DIRECTION BALANCE
    # =========================================================================
    print(f"\n### DIRECTION BALANCE ###")
    
    higher_pcts = [r['higher_pct'] for r in hybrid if r.get('higher_pct') is not None]
    lower_pcts = [r['lower_pct'] for r in hybrid if r.get('lower_pct') is not None]
    
    avg_higher = np.mean(higher_pcts) * 100
    avg_lower = np.mean(lower_pcts) * 100
    
    print(f"  Avg Higher (bullish USD): {avg_higher:.1f}%")
    print(f"  Avg Lower (bearish USD):  {avg_lower:.1f}%")
    print(f"  Higher/Lower Ratio: {avg_higher/avg_lower:.2f}:1")
    
    imbalance = abs(avg_higher - avg_lower)
    if imbalance < 10:
        print(f"  ✓ WELL BALANCED (within 10%)")
    elif imbalance < 20:
        print(f"  ~ MODERATELY BALANCED (within 20%)")
    else:
        print(f"  ✗ STILL IMBALANCED (>{imbalance:.0f}% difference)")
    
    # =========================================================================
    # DIRECTION ACCURACY
    # =========================================================================
    print(f"\n### DIRECTION ACCURACY ###")
    
    def check_direction(row):
        pred_up = row['final_prediction'] > row['last_close']
        actual_up = row['actual_price'] > row['last_close']
        return pred_up == actual_up
    
    ensemble_correct = sum(1 for r in hybrid if check_direction(r))
    stat_correct = sum(
        1 for r in hybrid 
        if (r['stat_prediction'] > r['last_close']) == (r['actual_price'] > r['last_close'])
    )
    
    print(f"  Ensemble: {ensemble_correct}/{len(hybrid)} ({100*ensemble_correct/len(hybrid):.1f}%)")
    print(f"  Statistical: {stat_correct}/{len(hybrid)} ({100*stat_correct/len(hybrid):.1f}%)")
    
    if ensemble_correct > stat_correct:
        print(f"  ✓ Ensemble has better direction accuracy")
    elif ensemble_correct < stat_correct:
        print(f"  ✗ Stat model has better direction accuracy")
    else:
        print(f"  = Same direction accuracy")
    
    # =========================================================================
    # ENTROPY CONFIDENCE ANALYSIS
    # =========================================================================
    print(f"\n### ENTROPY CONFIDENCE ANALYSIS ###")
    
    entropy_confs = [r['entropy_confidence'] for r in hybrid if r.get('entropy_confidence') is not None]
    
    print(f"  Avg Entropy Confidence: {np.mean(entropy_confs):.3f}")
    print(f"  Std Entropy Confidence: {np.std(entropy_confs):.3f}")
    
    # When entropy confidence is high, did LLM help more?
    high_entropy = [r for r in hybrid if r.get('entropy_confidence', 0) > 0.6]
    low_entropy = [r for r in hybrid if r.get('entropy_confidence', 0) < 0.4]
    
    if high_entropy:
        high_helped = sum(
            1 for r in high_entropy
            if abs(r.get('prediction_error_pct', 0)) < abs(r.get('stat_error_pct', 0))
        )
        print(f"\n  High entropy (agreement): LLM helped {high_helped}/{len(high_entropy)} ({100*high_helped/len(high_entropy):.1f}%)")
    
    if low_entropy:
        low_helped = sum(
            1 for r in low_entropy
            if abs(r.get('prediction_error_pct', 0)) < abs(r.get('stat_error_pct', 0))
        )
        print(f"  Low entropy (disagreement): LLM helped {low_helped}/{len(low_entropy)} ({100*low_helped/len(low_entropy):.1f}%)")
    
    # =========================================================================
    # DYNAMIC WEIGHTS ANALYSIS
    # =========================================================================
    print(f"\n### DYNAMIC WEIGHTS ###")
    
    stat_weights = [r['stat_weight'] for r in hybrid if r.get('stat_weight') is not None]
    
    print(f"  Avg Stat Weight: {np.mean(stat_weights)*100:.1f}%")
    print(f"  Min Stat Weight: {np.min(stat_weights)*100:.1f}%")
    print(f"  Max Stat Weight: {np.max(stat_weights)*100:.1f}%")
    print(f"  Std Stat Weight: {np.std(stat_weights)*100:.2f}%")
    
    # =========================================================================
    # BIAS ANALYSIS
    # =========================================================================
    print(f"\n### BIAS TRACKING ###")
    
    historical_biases = [r['historical_bias'] for r in hybrid if r.get('historical_bias') is not None]
    
    print(f"  Avg Historical Bias: {np.mean(historical_biases):+.4f}%")
    print(f"  Final Historical Bias: {historical_biases[-1]:+.4f}%" if historical_biases else "  N/A")
    
    # Did bias correction help?
    if abs(np.mean(errors)) < abs(np.mean(stat_errors)):
        print(f"  ✓ Ensemble has lower bias than stat model")
    else:
        print(f"  Stat model has lower bias")
    
    # =========================================================================
    # PER-PERSONA ANALYSIS
    # =========================================================================
    print(f"\n### PERSONA BREAKDOWN ###")
    
    persona_stats = {}
    
    for r in hybrid:
        for p in r.get('persona_predictions', []):
            name = p.get('short_name', 'Unknown')
            if name not in persona_stats:
                persona_stats[name] = {
                    'directions': [],
                    'confidences': [],
                    'pips': [],
                }
            if p.get('success'):
                persona_stats[name]['directions'].append(p.get('direction', 'unchanged'))
                persona_stats[name]['confidences'].append(p.get('confidence', 5))
                persona_stats[name]['pips'].append(p.get('adjustment_pips', 0))
    
    print(f"\n{'Persona':<15} {'Higher%':>8} {'Lower%':>8} {'Unch%':>8} {'AvgConf':>8} {'AvgPips':>8}")
    print("-" * 65)
    
    for name, stats in sorted(persona_stats.items()):
        if not stats['directions']:
            continue
        total = len(stats['directions'])
        higher = sum(1 for d in stats['directions'] if d == 'higher') / total * 100
        lower = sum(1 for d in stats['directions'] if d == 'lower') / total * 100
        unchanged = sum(1 for d in stats['directions'] if d == 'unchanged') / total * 100
        conf = np.mean(stats['confidences'])
        pips = np.mean(stats['pips'])
        
        print(f"{name:<15} {higher:>7.1f}% {lower:>7.1f}% {unchanged:>7.1f}% {conf:>8.1f} {pips:>8.1f}")
    
    # =========================================================================
    # REGIME ANALYSIS
    # =========================================================================
    print(f"\n### REGIME-CONDITIONAL PERFORMANCE ###")
    
    for vol_regime in ['low', 'normal', 'high']:
        regime_days = [r for r in hybrid if r.get('regime_volatility') == vol_regime]
        if regime_days:
            regime_errors = [abs(r['prediction_error_pct']) for r in regime_days if r.get('prediction_error_pct')]
            regime_mae = np.mean(regime_errors)
            print(f"  {vol_regime.upper()} volatility: MAE={regime_mae:.4f}% ({len(regime_days)} days)")
    
    return pd.DataFrame(hybrid)


def compare_versions():
    """Compare V3 and V4 if both exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    v3_path = os.path.join(script_dir, "simulation_results_v3.json")
    v4_path = os.path.join(script_dir, "simulation_results_v4.json")
    
    if not os.path.exists(v3_path):
        print("\nV3 results not found for comparison")
        return
    
    if not os.path.exists(v4_path):
        print("\nV4 results not found")
        return
    
    v3 = load_results(v3_path)
    v4 = load_results(v4_path)
    
    v3_hybrid = [r for r in v3 if r.get('mode') == 'hybrid' and r.get('prediction_error_pct') is not None]
    v4_hybrid = [r for r in v4 if r.get('mode') == 'hybrid' and r.get('prediction_error_pct') is not None]
    
    if not v3_hybrid or not v4_hybrid:
        return
    
    print("\n" + "=" * 70)
    print("V3 vs V4 COMPARISON")
    print("=" * 70)
    
    # Calculate metrics
    v3_mae = np.mean([abs(r['prediction_error_pct']) for r in v3_hybrid])
    v4_mae = np.mean([abs(r['prediction_error_pct']) for r in v4_hybrid])
    
    v3_stat_mae = np.mean([abs(r['stat_error_pct']) for r in v3_hybrid if r.get('stat_error_pct')])
    v4_stat_mae = np.mean([abs(r['stat_error_pct']) for r in v4_hybrid if r.get('stat_error_pct')])
    
    v3_helped = sum(1 for r in v3_hybrid if abs(r['prediction_error_pct']) < abs(r.get('stat_error_pct', float('inf'))))
    v4_helped = sum(1 for r in v4_hybrid if abs(r['prediction_error_pct']) < abs(r.get('stat_error_pct', float('inf'))))
    
    # V3 uses bullish/bearish, V4 uses higher/lower
    v3_bullish = np.mean([r.get('bullish_pct', 0.5) for r in v3_hybrid]) * 100
    v4_higher = np.mean([r.get('higher_pct', 0.5) for r in v4_hybrid]) * 100
    
    # Direction accuracy
    v3_dir = sum(1 for r in v3_hybrid 
                 if (r['final_prediction'] > r['last_close']) == (r['actual_price'] > r['last_close']))
    v4_dir = sum(1 for r in v4_hybrid 
                 if (r['final_prediction'] > r['last_close']) == (r['actual_price'] > r['last_close']))
    
    print(f"\n{'Metric':<35} {'V3':>12} {'V4':>12} {'Change':>12}")
    print("-" * 75)
    print(f"{'Ensemble MAE':<35} {v3_mae:>11.4f}% {v4_mae:>11.4f}% {(v4_mae-v3_mae)/v3_mae*100:>+11.1f}%")
    print(f"{'Statistical MAE':<35} {v3_stat_mae:>11.4f}% {v4_stat_mae:>11.4f}% {(v4_stat_mae-v3_stat_mae)/v3_stat_mae*100:>+11.1f}%")
    print(f"{'LLM Helped %':<35} {100*v3_helped/len(v3_hybrid):>11.1f}% {100*v4_helped/len(v4_hybrid):>11.1f}% {100*v4_helped/len(v4_hybrid)-100*v3_helped/len(v3_hybrid):>+11.1f}%")
    print(f"{'Bullish/Higher %':<35} {v3_bullish:>11.1f}% {v4_higher:>11.1f}% {v4_higher-v3_bullish:>+11.1f}%")
    print(f"{'Direction Accuracy':<35} {100*v3_dir/len(v3_hybrid):>11.1f}% {100*v4_dir/len(v4_hybrid):>11.1f}% {100*v4_dir/len(v4_hybrid)-100*v3_dir/len(v3_hybrid):>+11.1f}%")
    
    # Summary verdict
    print("\n" + "=" * 70)
    improvements = 0
    
    if v4_mae < v3_mae:
        improvements += 1
        print("✓ V4 has lower MAE")
    else:
        print("✗ V3 has lower MAE")
    
    if 100*v4_helped/len(v4_hybrid) > 100*v3_helped/len(v3_hybrid):
        improvements += 1
        print("✓ V4 LLM helps more often")
    else:
        print("✗ V3 LLM helps more often")
    
    if abs(v4_higher - 50) < abs(v3_bullish - 50):
        improvements += 1
        print("✓ V4 has better direction balance")
    else:
        print("✗ V3 has better direction balance")
    
    if 100*v4_dir/len(v4_hybrid) > 100*v3_dir/len(v3_hybrid):
        improvements += 1
        print("✓ V4 has better direction accuracy")
    else:
        print("~ Direction accuracy similar or V3 better")
    
    print(f"\nVERDICT: {improvements}/4 metrics improved in V4")
    print("=" * 70)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    v4_path = os.path.join(script_dir, "simulation_results_v4.json")
    
    if os.path.exists(v4_path):
        results = load_results(v4_path)
        df = analyze_v4(results)
        compare_versions()
    else:
        print("V4 results not found. Run run_simulation_v4.py first.")
        
        # If V4 doesn't exist, analyze V3
        v3_path = os.path.join(script_dir, "simulation_results_v3.json")
        if os.path.exists(v3_path):
            print("\nShowing V3 analysis instead:")
            from analyze_results_v3 import analyze_v3
            results = load_results(v3_path)
            analyze_v3(results)

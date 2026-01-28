"""
Analyze V3 results and compare with V2.
Focus on whether the fixes worked:
1. Is bullish/bearish ratio closer to 50/50?
2. Does LLM help more often?
3. Is MAE lower?
4. Is direction accuracy better?
"""

import json
import pandas as pd
import numpy as np
import os

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_v3(results):
    """Full analysis of V3 results."""

    hybrid = [r for r in results if r.get('mode') == 'hybrid' and r.get('actual_price')]

    if not hybrid:
        print("No hybrid results found.")
        return

    print("=" * 70)
    print("V3 SIMULATION ANALYSIS")
    print("=" * 70)

    # Basic metrics
    print("\n### PREDICTION ACCURACY ###")

    errors = [r['prediction_error_pct'] for r in hybrid if r.get('prediction_error_pct') is not None]
    stat_errors = [r['stat_error_pct'] for r in hybrid if r.get('stat_error_pct') is not None]

    print(f"\nEnsemble Model (Stat + LLM):")
    print(f"  MAE:     {np.mean(np.abs(errors)):.4f}%")
    print(f"  Bias:    {np.mean(errors):+.4f}%")
    print(f"  Std:     {np.std(errors):.4f}%")
    print(f"  Max:     {np.max(np.abs(errors)):.4f}%")

    print(f"\nStatistical Model Only:")
    print(f"  MAE:     {np.mean(np.abs(stat_errors)):.4f}%")
    print(f"  Bias:    {np.mean(stat_errors):+.4f}%")

    # Did LLM help?
    llm_helped = sum(
        1 for r in hybrid
        if r.get('prediction_error_pct') is not None and r.get('stat_error_pct') is not None
        and abs(r['prediction_error_pct']) < abs(r['stat_error_pct'])
    )
    total = sum(
        1 for r in hybrid
        if r.get('prediction_error_pct') is not None and r.get('stat_error_pct') is not None
    )

    print(f"\n### LLM CONTRIBUTION ###")
    print(f"  LLM helped in {llm_helped}/{total} cases ({100*llm_helped/total:.1f}%)")

    if np.mean(np.abs(errors)) < np.mean(np.abs(stat_errors)):
        improvement = (np.mean(np.abs(stat_errors)) - np.mean(np.abs(errors))) / np.mean(np.abs(stat_errors)) * 100
        print(f"  LLM IMPROVED overall MAE by {improvement:.1f}%")
    else:
        degradation = (np.mean(np.abs(errors)) - np.mean(np.abs(stat_errors))) / np.mean(np.abs(stat_errors)) * 100
        print(f"  LLM DEGRADED overall MAE by {degradation:.1f}%")

    # Bias balance
    print(f"\n### BIAS BALANCE ###")
    bullish_pcts = [r['bullish_pct'] for r in hybrid if r.get('bullish_pct') is not None]
    bearish_pcts = [r['bearish_pct'] for r in hybrid if r.get('bearish_pct') is not None]

    print(f"  Avg Bullish: {np.mean(bullish_pcts)*100:.1f}%")
    print(f"  Avg Bearish: {np.mean(bearish_pcts)*100:.1f}%")
    print(f"  Bull/Bear Ratio: {np.mean(bullish_pcts)/np.mean(bearish_pcts):.2f}:1")

    if abs(np.mean(bullish_pcts) - np.mean(bearish_pcts)) < 0.15:
        print("  BALANCED (within 15%)")
    else:
        print("  STILL IMBALANCED")

    # Direction accuracy
    print(f"\n### DIRECTION ACCURACY ###")
    correct = sum(
        1 for r in hybrid
        if (r['final_prediction'] > r['last_close'] and r['actual_price'] > r['last_close']) or
           (r['final_prediction'] < r['last_close'] and r['actual_price'] < r['last_close'])
    )
    print(f"  Ensemble: {correct}/{len(hybrid)} ({100*correct/len(hybrid):.1f}%)")

    stat_correct = sum(
        1 for r in hybrid
        if (r['stat_prediction'] > r['last_close'] and r['actual_price'] > r['last_close']) or
           (r['stat_prediction'] < r['last_close'] and r['actual_price'] < r['last_close'])
    )
    print(f"  Statistical: {stat_correct}/{len(hybrid)} ({100*stat_correct/len(hybrid):.1f}%)")

    # Dynamic weights analysis
    print(f"\n### DYNAMIC WEIGHTS ###")
    stat_weights = [r['stat_weight'] for r in hybrid if r.get('stat_weight') is not None]
    print(f"  Avg Stat Weight: {np.mean(stat_weights)*100:.1f}%")
    print(f"  Min Stat Weight: {np.min(stat_weights)*100:.1f}%")
    print(f"  Max Stat Weight: {np.max(stat_weights)*100:.1f}%")

    # Consensus strength analysis
    print(f"\n### CONSENSUS ANALYSIS ###")
    consensus_strengths = [r['llm_consensus_strength'] for r in hybrid if r.get('llm_consensus_strength') is not None]
    print(f"  Avg Consensus Strength: {np.mean(consensus_strengths):.3f}")

    # When strong consensus, did it help?
    strong_consensus = [r for r in hybrid if r.get('llm_consensus_strength', 0) > 0.4]
    if strong_consensus:
        strong_helped = sum(
            1 for r in strong_consensus
            if abs(r.get('prediction_error_pct', 0)) < abs(r.get('stat_error_pct', 0))
        )
        print(f"  Strong consensus helped: {strong_helped}/{len(strong_consensus)} ({100*strong_helped/len(strong_consensus):.1f}%)")

    # Per-persona analysis
    print(f"\n### PERSONA BREAKDOWN ###")
    persona_stats = {}

    for r in hybrid:
        for p in r.get('persona_predictions', []):
            name = p.get('short_name', 'Unknown')
            if name not in persona_stats:
                persona_stats[name] = {
                    'directions': [],
                    'confidences': [],
                    'bias': p.get('bias', 'neutral')
                }
            if p.get('success'):
                persona_stats[name]['directions'].append(p['direction'])
                persona_stats[name]['confidences'].append(p['confidence'])

    print(f"\n{'Persona':<15} {'Bias':<8} {'Bullish%':>8} {'Bearish%':>8} {'Neutral%':>8} {'AvgConf':>8}")
    print("-" * 65)

    for name, stats in sorted(persona_stats.items()):
        if not stats['directions']:
            continue
        total = len(stats['directions'])
        bull = sum(1 for d in stats['directions'] if d == 'bullish_usd') / total * 100
        bear = sum(1 for d in stats['directions'] if d == 'bearish_usd') / total * 100
        neut = sum(1 for d in stats['directions'] if d == 'neutral') / total * 100
        conf = np.mean(stats['confidences'])

        print(f"{name:<15} {stats['bias'][:7]:<8} {bull:>7.1f}% {bear:>7.1f}% {neut:>7.1f}% {conf:>8.1f}")

    # Check if personas are following their bias properly
    print(f"\n### BIAS ADHERENCE CHECK ###")
    for name, stats in persona_stats.items():
        if not stats['directions']:
            continue

        bias = stats['bias']
        total = len(stats['directions'])
        bull_pct = sum(1 for d in stats['directions'] if d == 'bullish_usd') / total
        bear_pct = sum(1 for d in stats['directions'] if d == 'bearish_usd') / total

        if bias == 'bullish_usd' and bull_pct < 0.4:
            print(f"  {name}: Bullish bias but only {bull_pct*100:.0f}% bullish calls - CONTRARIAN")
        elif bias == 'bearish_usd' and bear_pct < 0.4:
            print(f"  {name}: Bearish bias but only {bear_pct*100:.0f}% bearish calls - CONTRARIAN")
        elif bias == 'neutral' and (bull_pct > 0.6 or bear_pct > 0.6):
            print(f"  {name}: Neutral bias but showing strong direction - ISSUE")

    return hybrid


def compare_v2_v3():
    """Compare V2 and V3 if both exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    v2_path = os.path.join(script_dir, "simulation_results_v2.json")
    v3_path = os.path.join(script_dir, "simulation_results_v3.json")

    if not os.path.exists(v2_path) or not os.path.exists(v3_path):
        print("\nCannot compare: need both V2 and V3 results")
        return

    v2 = load_results(v2_path)
    v3 = load_results(v3_path)

    v2_hybrid = [r for r in v2 if r.get('mode') == 'hybrid' and r.get('prediction_error_pct') is not None]
    v3_hybrid = [r for r in v3 if r.get('mode') == 'hybrid' and r.get('prediction_error_pct') is not None]

    if not v2_hybrid or not v3_hybrid:
        return

    print("\n" + "=" * 70)
    print("V2 vs V3 COMPARISON")
    print("=" * 70)

    v2_mae = np.mean([abs(r['prediction_error_pct']) for r in v2_hybrid])
    v3_mae = np.mean([abs(r['prediction_error_pct']) for r in v3_hybrid])

    v2_stat_mae = np.mean([abs(r['stat_error_pct']) for r in v2_hybrid if r.get('stat_error_pct') is not None])
    v3_stat_mae = np.mean([abs(r['stat_error_pct']) for r in v3_hybrid if r.get('stat_error_pct') is not None])

    print(f"\n{'Metric':<30} {'V2':>12} {'V3':>12} {'Change':>12}")
    print("-" * 66)
    print(f"{'Ensemble MAE':<30} {v2_mae:>11.4f}% {v3_mae:>11.4f}% {(v3_mae-v2_mae)/v2_mae*100:>+11.1f}%")
    print(f"{'Statistical MAE':<30} {v2_stat_mae:>11.4f}% {v3_stat_mae:>11.4f}% {(v3_stat_mae-v2_stat_mae)/v2_stat_mae*100:>+11.1f}%")

    v2_helped = sum(1 for r in v2_hybrid if abs(r.get('prediction_error_pct', 0)) < abs(r.get('stat_error_pct', 0)))
    v3_helped = sum(1 for r in v3_hybrid if abs(r.get('prediction_error_pct', 0)) < abs(r.get('stat_error_pct', 0)))

    print(f"{'LLM Helped %':<30} {100*v2_helped/len(v2_hybrid):>11.1f}% {100*v3_helped/len(v3_hybrid):>11.1f}% {100*v3_helped/len(v3_hybrid) - 100*v2_helped/len(v2_hybrid):>+11.1f}%")

    # Bias balance
    v2_bull = np.mean([r.get('bullish_pct', 0.5) for r in v2_hybrid])
    v3_bull = np.mean([r.get('bullish_pct', 0.5) for r in v3_hybrid])

    print(f"{'Avg Bullish %':<30} {v2_bull*100:>11.1f}% {v3_bull*100:>11.1f}% {(v3_bull-v2_bull)*100:>+11.1f}%")

    # Direction accuracy
    v2_dir = sum(1 for r in v2_hybrid if (r['final_prediction'] > r['last_close']) == (r['actual_price'] > r['last_close'])) / len(v2_hybrid)
    v3_dir = sum(1 for r in v3_hybrid if (r['final_prediction'] > r['last_close']) == (r['actual_price'] > r['last_close'])) / len(v3_hybrid)

    print(f"{'Direction Accuracy':<30} {v2_dir*100:>11.1f}% {v3_dir*100:>11.1f}% {(v3_dir-v2_dir)*100:>+11.1f}%")

    print("\n" + "=" * 70)
    if v3_mae < v2_mae and 100*v3_helped/len(v3_hybrid) > 100*v2_helped/len(v2_hybrid):
        print("VERDICT: V3 IMPROVEMENTS WORKING")
    elif v3_mae < v2_mae:
        print("VERDICT: V3 MAE improved but LLM help rate unchanged")
    else:
        print("VERDICT: V3 needs more tuning")
    print("=" * 70)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    v3_path = os.path.join(script_dir, "simulation_results_v3.json")

    if not os.path.exists(v3_path):
        print("No V3 results found. Run run_simulation_v3.py first.")
        return

    results = load_results(v3_path)
    analyze_v3(results)
    compare_v2_v3()


if __name__ == "__main__":
    main()

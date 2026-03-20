"""
Walk-forward expanding-window backtest with strict temporal ordering.
Outputs: RMSE, MAE, directional accuracy, R², per-regime breakdown.
Compares against MA_Momentum baseline (R²=0.571).
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import config
from data_loader import detect_regime, prepare_dataset
from llm_tasks import LLMAnalyst
from meta_learner import MetaLearner
from market_agents import MarketSimulation
from pipeline import DailyPipeline
from stat_engine import RegimeConditionalEngine


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict:
    """Compute regression metrics."""
    if len(actuals) == 0:
        return {}

    errors = predictions - actuals
    abs_errors = np.abs(errors)

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Directional accuracy (predicting next-day direction)
    if len(actuals) >= 2:
        actual_dir = np.diff(actuals) > 0
        pred_dir = np.diff(predictions) > 0
        dir_acc = np.mean(actual_dir == pred_dir)
    else:
        dir_acc = 0.0

    return {
        "n": len(actuals),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(abs_errors)),
        "r2": float(r2),
        "directional_accuracy": float(dir_acc),
        "max_error": float(np.max(abs_errors)),
        "mean_error": float(np.mean(errors)),  # bias
    }


def ma_momentum_baseline(df: pd.DataFrame, test_indices: List[int]) -> np.ndarray:
    """Simple MA_Momentum baseline: predict next INR as current + MA momentum.
    This is the benchmark to beat (R²~0.571).
    """
    predictions = []
    for idx in test_indices:
        row = df.iloc[idx - 1]  # features from day before
        current = row["INR"]
        momentum = row.get("MA_momentum", 0)
        if pd.isna(momentum):
            momentum = 0
        # Predict: current price adjusted by momentum
        pred = current * (1 + momentum)
        predictions.append(pred)
    return np.array(predictions)


def naive_baseline(df: pd.DataFrame, test_indices: List[int]) -> np.ndarray:
    """Naive baseline: predict next INR = current INR (random walk)."""
    return np.array([df.iloc[idx - 1]["INR"] for idx in test_indices])


def run_backtest(
    use_llm: bool = True,
    use_agents: bool = False,
    start_year: int = 2023,
    end_year: int = 2025,
    refit_freq: int = config.REFIT_FREQUENCY,
    verbose: bool = True,
) -> Dict:
    """Run full walk-forward backtest.

    Args:
        use_llm: Whether to use LLM features (set False for stat-only comparison)
        start_year: Start of test period
        end_year: End of test period
        refit_freq: How often to refit models (in trading days)
        verbose: Print progress

    Returns:
        Dict with overall metrics, per-regime breakdown, and comparison
    """
    t0 = time.time()

    # Load and prepare data
    if verbose:
        print("Loading data...")
    df = prepare_dataset()

    # Define test period
    test_mask = (df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)
    test_indices = df[test_mask].index.tolist()

    if not test_indices:
        return {"error": f"No data found for years {start_year}-{end_year}"}

    # Ensure enough training data before test period
    first_test = test_indices[0]
    if first_test < config.EXPANDING_WINDOW_START:
        test_indices = [i for i in test_indices if i >= config.EXPANDING_WINDOW_START]
        if not test_indices:
            return {"error": "Not enough training data before test period"}

    if verbose:
        print(f"Test period: {df.iloc[test_indices[0]]['Date'].date()} to "
              f"{df.iloc[test_indices[-1]]['Date'].date()} ({len(test_indices)} days)",
              flush=True)

    # Initialize components
    llm = LLMAnalyst() if use_llm else None
    sim = MarketSimulation() if use_agents else None
    engine = RegimeConditionalEngine()
    meta = MetaLearner()
    # Pass date range so GDELT loader only reads the relevant slice (avoids loading 14M rows)
    gdelt_start = df.iloc[test_indices[0]]["Date"] - pd.Timedelta(days=5)
    gdelt_end = df.iloc[test_indices[-1]]["Date"]
    pipeline = DailyPipeline(
        df, llm=llm, sim=sim, engine=engine, meta=meta,
        use_llm=use_llm, use_agents=use_agents,
        gdelt_start=gdelt_start, gdelt_end=gdelt_end,
    )

    # Run walk-forward
    if verbose:
        print("Running walk-forward backtest...")

    results = pipeline.predict_range(
        test_indices[0], test_indices[-1] + 1, refit_freq=refit_freq
    )

    # Collect actuals and predictions
    actuals = []
    predictions = []
    regimes = []
    valid_indices = []

    for i, r in enumerate(results):
        if "error" not in r and r.get("actual") is not None:
            actuals.append(r["actual"])
            predictions.append(r["prediction"])
            regimes.append(r["regime"])
            valid_indices.append(test_indices[i] if i < len(test_indices) else i)

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    if len(actuals) == 0:
        return {"error": "No valid predictions generated"}

    # Overall metrics
    overall = compute_metrics(actuals, predictions)

    # Per-regime breakdown
    regime_metrics = {}
    for regime in config.REGIMES:
        mask = np.array(regimes) == regime
        if mask.sum() > 0:
            regime_metrics[regime] = compute_metrics(actuals[mask], predictions[mask])

    # Baselines
    ma_preds = ma_momentum_baseline(df, valid_indices)
    naive_preds = naive_baseline(df, valid_indices)
    ma_metrics = compute_metrics(actuals, ma_preds)
    naive_metrics = compute_metrics(actuals, naive_preds)

    elapsed = time.time() - t0

    report = {
        "mode": (
            "llm+agents+stat" if (use_llm and use_agents)
            else ("agents+stat" if use_agents else ("llm+stat" if use_llm else "stat_only"))
        ),
        "test_period": {
            "start": str(df.iloc[test_indices[0]]["Date"].date()),
            "end": str(df.iloc[test_indices[-1]]["Date"].date()),
            "n_days": len(test_indices),
            "n_valid": len(actuals),
        },
        "overall": overall,
        "per_regime": regime_metrics,
        "baselines": {
            "ma_momentum": ma_metrics,
            "naive_random_walk": naive_metrics,
        },
        "comparison": {
            "vs_ma_momentum_r2": round(overall["r2"] - ma_metrics.get("r2", 0), 4),
            "vs_ma_momentum_mae": round(ma_metrics.get("mae", 0) - overall["mae"], 6),
            "vs_naive_r2": round(overall["r2"] - naive_metrics.get("r2", 0), 4),
            "beats_ma_momentum": overall["r2"] > ma_metrics.get("r2", 0),
        },
        "meta_learner": meta.summary(),
        "elapsed_seconds": round(elapsed, 1),
    }

    if verbose:
        _print_report(report)

    return report


def _print_report(report: Dict):
    """Pretty-print backtest results."""
    print("\n" + "=" * 70)
    print(f"BACKTEST RESULTS ({report['mode']})")
    print("=" * 70)

    tp = report["test_period"]
    print(f"Period: {tp['start']} to {tp['end']} ({tp['n_valid']}/{tp['n_days']} valid days)")

    o = report["overall"]
    print(f"\n--- Overall Metrics ---")
    print(f"  RMSE:                {o['rmse']:.4f}")
    print(f"  MAE:                 {o['mae']:.4f}")
    print(f"  R²:                  {o['r2']:.4f}")
    print(f"  Directional Acc:     {o['directional_accuracy']:.1%}")
    print(f"  Max Error:           {o['max_error']:.4f}")
    print(f"  Bias (Mean Error):   {o['mean_error']:.6f}")

    print(f"\n--- Baselines ---")
    b = report["baselines"]
    for name, m in b.items():
        if m:
            print(f"  {name}: R²={m.get('r2', 0):.4f}  MAE={m.get('mae', 0):.4f}  "
                  f"DirAcc={m.get('directional_accuracy', 0):.1%}")

    c = report["comparison"]
    print(f"\n--- vs MA_Momentum ---")
    print(f"  R² delta:    {c['vs_ma_momentum_r2']:+.4f}")
    print(f"  MAE delta:   {c['vs_ma_momentum_mae']:+.6f} (positive=better)")
    print(f"  Beats MA:    {'YES' if c['beats_ma_momentum'] else 'NO'}")

    print(f"\n--- Per-Regime ---")
    for regime, m in report.get("per_regime", {}).items():
        print(f"  {regime}: n={m['n']}  R²={m['r2']:.4f}  MAE={m['mae']:.4f}  "
              f"DirAcc={m['directional_accuracy']:.1%}")

    ml = report.get("meta_learner", {})
    if ml.get("regimes"):
        print(f"\n--- Meta-Learner ---")
        for regime, info in ml["regimes"].items():
            w = info.get("weight", 0)
            s = info.get("samples", 0)
            mae = info.get("mae")
            mae_str = f"{mae:.6f}" if mae else "N/A"
            print(f"  {regime}: weight={w:.4f}  samples={s}  mae={mae_str}")

    print(f"\nElapsed: {report['elapsed_seconds']:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    # Quick test: stat-only backtest
    report = run_backtest(use_llm=False, start_year=2024, end_year=2025, verbose=True)

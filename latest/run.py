"""
CLI entry point for the LLM-as-Analyst USD/INR Prediction System.
Supports backtest mode, single-date mode, and date-range mode.
"""

import argparse
import json
import sys
import os

# Ensure latest/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_loader import prepare_dataset, build_context_packet, load_gdelt_headlines
from llm_tasks import LLMAnalyst
from stat_engine import RegimeConditionalEngine
from meta_learner import MetaLearner
from pipeline import DailyPipeline
from backtest import run_backtest

import pandas as pd
import numpy as np


def cmd_backtest(args):
    """Run walk-forward backtest."""
    report = run_backtest(
        use_llm=not args.no_llm,
        use_agents=args.agents,
        start_year=args.start_year,
        end_year=args.end_year,
        refit_freq=args.refit_freq,
        verbose=True,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return report


def cmd_predict(args):
    """Predict for a single date."""
    df = prepare_dataset()

    target_date = pd.to_datetime(args.date)
    matches = df[df["Date"].dt.date == target_date.date()]
    if matches.empty:
        print(f"Date {args.date} not found in dataset")
        sys.exit(1)

    target_idx = matches.index[0]

    # Fit on all data before target
    llm = LLMAnalyst() if not args.no_llm else None
    engine = RegimeConditionalEngine()
    meta = MetaLearner()
    use_agents = args.agents or getattr(args, "quick_agents", False)
    agent_personas = None
    if use_agents and getattr(args, "quick_agents", False):
        from market_agents import AGENT_PERSONAS
        # Quick mode: one representative from each archetype (6 agents, one full Ollama batch)
        seen_arch = set()
        agent_personas = []
        for p in AGENT_PERSONAS:
            if p["archetype"] not in seen_arch:
                agent_personas.append(p)
                seen_arch.add(p["archetype"])

    pipeline = DailyPipeline(
        df, llm=llm, engine=engine, meta=meta,
        use_llm=not args.no_llm, use_agents=use_agents,
        agent_personas=agent_personas,
    )
    pipeline.fit(target_idx)

    result = pipeline.predict_single(target_idx)

    print(json.dumps(result, indent=2, default=str))
    return result


def cmd_range(args):
    """Predict for a date range."""
    df = prepare_dataset()

    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    mask = (df["Date"].dt.date >= start.date()) & (df["Date"].dt.date <= end.date())
    range_indices = df[mask].index.tolist()

    if not range_indices:
        print(f"No data in range {args.start} to {args.end}")
        sys.exit(1)

    llm = LLMAnalyst() if not args.no_llm else None
    engine = RegimeConditionalEngine()
    meta = MetaLearner()
    pipeline = DailyPipeline(
        df, llm=llm, engine=engine, meta=meta,
        use_llm=not args.no_llm, use_agents=args.agents,
    )

    results = pipeline.predict_range(
        range_indices[0], range_indices[-1] + 1, refit_freq=args.refit_freq
    )

    # Summary
    valid = [r for r in results if "error" not in r and r.get("actual") is not None]
    if valid:
        actuals = np.array([r["actual"] for r in valid])
        preds = np.array([r["prediction"] for r in valid])
        errors = preds - actuals
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\nRange summary: {len(valid)} predictions")
        print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Analyst USD/INR Prediction System"
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Backtest command
    bt = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt.add_argument("--start-year", type=int, default=2024, help="Test start year")
    bt.add_argument("--end-year", type=int, default=2025, help="Test end year")
    bt.add_argument("--no-llm", action="store_true", help="Disable Gemini analyst tasks")
    bt.add_argument("--agents", action="store_true", help="Enable 30-agent Ollama simulation")
    bt.add_argument("--refit-freq", type=int, default=config.REFIT_FREQUENCY,
                     help="Refit frequency in trading days")
    bt.add_argument("--output", "-o", type=str, help="Save results to JSON file")

    # Single predict
    pr = sub.add_parser("predict", help="Predict for a single date")
    pr.add_argument("date", type=str, help="Target date (YYYY-MM-DD)")
    pr.add_argument("--no-llm", action="store_true", help="Disable Gemini analyst tasks")
    pr.add_argument("--agents", action="store_true", help="Enable 30-agent Ollama simulation")
    pr.add_argument("--quick-agents", action="store_true", help="Use 10 agents only (faster)")

    # Range predict
    rg = sub.add_parser("range", help="Predict for a date range")
    rg.add_argument("start", type=str, help="Start date (YYYY-MM-DD)")
    rg.add_argument("end", type=str, help="End date (YYYY-MM-DD)")
    rg.add_argument("--no-llm", action="store_true", help="Disable Gemini analyst tasks")
    rg.add_argument("--agents", action="store_true", help="Enable 30-agent Ollama simulation")
    rg.add_argument("--refit-freq", type=int, default=config.REFIT_FREQUENCY)
    rg.add_argument("--output", "-o", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "range":
        cmd_range(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

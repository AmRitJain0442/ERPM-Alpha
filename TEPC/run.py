"""
CLI entry point for the INR/USD TEPC pipeline.
"""

from pathlib import Path
import argparse
import json
import sys


MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from tepc.config import RunConfig, build_experiment_specs
from tepc.evaluation import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the INR/USD Topology-Enabled Predictions from Chaos pipeline."
    )
    parser.add_argument("--test-days", type=int, default=90, help="Walk-forward test window in decision dates.")
    parser.add_argument("--train-min-days", type=int, default=180, help="Minimum historical rows required for training.")
    parser.add_argument("--validation-days", type=int, default=45, help="Validation rows inside the training window.")
    parser.add_argument("--forecast-horizon-days", type=int, default=1, help="Forecast horizon for future INR/USD return.")
    parser.add_argument("--volatility-window", type=int, default=5, help="Forward realized volatility window.")
    parser.add_argument("--breakout-threshold", type=float, default=0.005, help="Breakout label threshold on future return.")
    parser.add_argument("--corr-window", type=int, default=30, help="Rolling window for dynamic topology.")
    parser.add_argument("--chaos-lookback-days", type=int, default=20, help="Lookback window for deterministic oscillator initialization.")
    parser.add_argument("--integration-steps", type=int, default=24, help="RK4 steps per coupling stage.")
    parser.add_argument("--refit-frequency", type=int, default=5, help="Refit cadence in test rows.")
    parser.add_argument("--experiment", action="append", default=[], help="Run only a specific experiment; may be repeated.")
    parser.add_argument("--output-dir", type=str, default="", help="Optional explicit output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RunConfig(
        test_days=args.test_days,
        train_min_days=args.train_min_days,
        validation_days=args.validation_days,
        forecast_horizon_days=args.forecast_horizon_days,
        volatility_window=args.volatility_window,
        breakout_threshold=args.breakout_threshold,
        corr_window=args.corr_window,
        chaos_lookback_days=args.chaos_lookback_days,
        integration_steps=args.integration_steps,
        refit_frequency=args.refit_frequency,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )

    experiments = build_experiment_specs(args.experiment)
    result = run_experiments(config, experiments)
    print(json.dumps(result["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

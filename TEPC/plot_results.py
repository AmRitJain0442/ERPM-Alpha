"""
CLI entry point for TEPC result plotting.
"""

from pathlib import Path
import argparse
import json
import sys


MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from tepc.plotting import plot_multi_run, plot_single_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison plots for TEPC run outputs.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory containing ablation_summary.csv and daily_predictions.csv. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional explicit plot output directory. For a single run, defaults to <run-dir>/plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [Path(value).resolve() for value in args.run_dir]

    if len(run_dirs) == 1:
        result = plot_single_run(run_dirs[0], Path(args.output_dir).resolve() if args.output_dir else None)
    else:
        if not args.output_dir:
            raise ValueError("An explicit --output-dir is required when plotting multiple run directories.")
        result = plot_multi_run(run_dirs, Path(args.output_dir).resolve())

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

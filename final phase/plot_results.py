"""
CLI entry point for final-phase result comparison plots.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys


MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from final_phase.plotting import generate_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for final-phase backtest outputs."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Path to a final-phase output directory containing ablation_summary.csv; may be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where comparison plots should be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of best experiments per horizon to include in date-level plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = [Path(path).resolve() for path in args.run_dir]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (MODULE_ROOT / "plots" / "comparison_latest")
    )
    generated = generate_plots(run_dirs, output_dir, top_k=args.top_k)
    print(f"Generated {len(generated)} files in {output_dir}")
    for path in generated:
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

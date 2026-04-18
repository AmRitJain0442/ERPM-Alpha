"""
CLI entry point for pulling and storing TEPC market and GDELT data.
"""

from pathlib import Path
import argparse
import json
import sys


MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from tepc.fetchers import PullConfig, pull_and_store_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull live market and GDELT data for the INR/USD TEPC pipeline."
    )
    parser.add_argument("--start-date", type=str, default="2024-12-01", help="Start date for market pulls.")
    parser.add_argument("--end-date", type=str, default="", help="Optional end date; defaults to today.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where pulled TEPC data should be written. Defaults to TEPC/data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PullConfig(
        start_date=args.start_date,
        end_date=args.end_date or None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    result = pull_and_store_data(config)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

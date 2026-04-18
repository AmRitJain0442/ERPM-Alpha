"""
CLI entry point for the final-phase market memory prediction system.
"""

from pathlib import Path
import argparse
import json
import sys

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from final_phase.config import RunConfig, build_experiment_specs
from final_phase.evaluation import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final-phase USD/INR hybrid forecasting and ablation system"
    )
    parser.add_argument("--test-days", type=int, default=100, help="Walk-forward test window")
    parser.add_argument("--train-min-days", type=int, default=120, help="Minimum training rows")
    parser.add_argument("--validation-days", type=int, default=30, help="Validation rows inside train window")
    parser.add_argument("--refit-frequency", type=int, default=10, help="Refit cadence in rows")
    parser.add_argument("--forecast-horizon-days", type=int, default=1, help="Forecast horizon in trading days")
    parser.add_argument("--embargo-days", type=int, default=1, help="Information embargo to avoid immediate-price anchoring")
    parser.add_argument("--rich-only", action="store_true", help="Use only dates where all rich GDELT layers are available")
    parser.add_argument("--experiment", action="append", default=[], help="Run a specific experiment by name; may be repeated")
    parser.add_argument("--llm-model", action="append", default=[], help="Persona backend spec, e.g. openrouter:<model-id> or ollama:<model-id>")
    parser.add_argument("--persona-limit", type=int, default=8, help="Maximum personas to use per experiment")
    parser.add_argument("--output-dir", type=str, default="", help="Optional explicit output directory")
    parser.add_argument(
        "--allow-premium-llm-models",
        action="store_true",
        help="Allow premium LLM specs such as Claude Opus. Disabled by default to avoid accidental spend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    blocked = [
        spec for spec in args.llm_model
        if ("claude-opus" in spec.lower() and not args.allow_premium_llm_models)
    ]
    if blocked:
        raise SystemExit(
            "Blocked premium model spec(s): "
            + ", ".join(blocked)
            + ". Use a cheaper OpenRouter/Ollama model or pass --allow-premium-llm-models."
        )

    config = RunConfig(
        test_days=args.test_days,
        train_min_days=args.train_min_days,
        validation_days=args.validation_days,
        refit_frequency=args.refit_frequency,
        forecast_horizon_days=args.forecast_horizon_days,
        embargo_days=args.embargo_days,
        rich_only=args.rich_only,
        persona_limit=args.persona_limit,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )

    experiments = build_experiment_specs(
        selected=args.experiment,
        llm_models=args.llm_model,
    )

    result = run_experiments(config, experiments)
    print(json.dumps(result["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

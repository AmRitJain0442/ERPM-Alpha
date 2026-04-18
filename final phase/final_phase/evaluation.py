"""
Walk-forward evaluation and ablation orchestration.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ExperimentSpec, RunConfig
from .data import load_integrated_dataset
from .features import build_feature_frame
from .memory import MarketMemoryBuilder
from .modeling import HybridModelSuite, blend_prediction
from .personas import LLMPersonaEngine, PersonaMemoryStore, RulePersonaEngine
from .reporting import write_outputs


def _sign_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual_sign = np.sign(actual)
    pred_sign = np.sign(predicted)
    tp = np.sum((actual_sign != 0) & (actual_sign == pred_sign))
    fp = np.sum((pred_sign != 0) & (actual_sign != pred_sign))
    fn = np.sum((actual_sign != 0) & (pred_sign == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def compute_metrics(daily_records: List[Dict]) -> Dict:
    df = pd.DataFrame(daily_records)
    actual_price = df["actual_price"].to_numpy(dtype=float)
    pred_price = df["predicted_price"].to_numpy(dtype=float)
    actual_return = df["actual_return"].to_numpy(dtype=float)
    pred_return = df["final_return"].to_numpy(dtype=float)

    mae_price = float(np.mean(np.abs(pred_price - actual_price)))
    rmse_price = float(np.sqrt(np.mean((pred_price - actual_price) ** 2)))
    mae_return = float(np.mean(np.abs(pred_return - actual_return)))
    directional_accuracy = float(np.mean(np.sign(pred_return) == np.sign(actual_return)))
    bias = float(np.mean(pred_return - actual_return))
    return {
        "n_days": int(len(df)),
        "mae_price": mae_price,
        "rmse_price": rmse_price,
        "mae_return": mae_return,
        "directional_accuracy": directional_accuracy,
        "sign_f1": float(_sign_f1(actual_return, pred_return)),
        "bias_return": bias,
    }


def _make_persona_engine(spec: ExperimentSpec, config: RunConfig):
    if spec.persona_backend is None:
        return None
    if spec.persona_backend == "rule":
        return RulePersonaEngine(config)
    return LLMPersonaEngine(spec.persona_backend, config)


def run_single_experiment(
    feature_frame: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    config: RunConfig,
    spec: ExperimentSpec,
) -> Dict:
    selected_cols: List[str] = []
    for group in spec.include_groups:
        selected_cols.extend(feature_groups.get(group, []))
    selected_cols = list(dict.fromkeys(selected_cols))

    needed_cols = selected_cols + ["Date", "INR", "target_date", "target_price", "target_return", "target_direction", "raw_return_1d", "expected_abs_move_proxy"]
    work = feature_frame[needed_cols].copy()
    work = work.dropna(subset=["target_price", "target_return"])
    if selected_cols:
        work = work.dropna(subset=selected_cols)
    work = work.reset_index(drop=True)

    if len(work) < config.train_min_days + config.test_days:
        return {
            "experiment": spec.name,
            "status": "skipped",
            "reason": f"Not enough rows after filtering: {len(work)}",
            "daily_records": [],
            "metrics": {},
            "persona_memory": {},
        }

    test_start = len(work) - config.test_days
    model_suite = HybridModelSuite(config)
    memory_builder = MarketMemoryBuilder()
    persona_store = PersonaMemoryStore()
    persona_engine = _make_persona_engine(spec, config)

    daily_records: List[Dict] = []
    last_fit_idx = -1
    pending_votes: List[Dict] = []

    for pos in range(test_start, len(work)):
        current_decision_date = pd.to_datetime(work.iloc[pos]["Date"])

        # Update persona calibration only once the corresponding target date is now observable.
        matured = []
        still_pending = []
        for item in pending_votes:
            if pd.to_datetime(item["target_date"]) <= current_decision_date:
                matured.append(item)
            else:
                still_pending.append(item)
        pending_votes = still_pending
        for item in matured:
            for vote in item["votes"]:
                from .personas import PersonaVote

                persona_store.record(
                    PersonaVote(
                        backend=vote["backend"],
                        persona=vote["persona"],
                        direction=vote["direction"],
                        expected_return=float(vote["expected_return"]),
                        magnitude=float(vote["magnitude"]),
                        confidence=float(vote["confidence"]),
                        thesis=vote["thesis"],
                        risk_flags=list(vote["risk_flags"]),
                        base_weight=float(vote["base_weight"]),
                        calibration_weight=float(vote["calibration_weight"]),
                    ),
                    actual_return=float(item["actual_return"]),
                )

        if (pos - last_fit_idx >= config.refit_frequency) or not model_suite.fitted:
            train_df = work.iloc[:pos].copy()
            train_df = train_df[pd.to_datetime(train_df["target_date"]) <= current_decision_date]
            if len(train_df) < config.train_min_days:
                continue
            model_suite.fit(train_df, selected_cols)
            last_fit_idx = pos

        row = work.iloc[pos]
        snapshot = memory_builder.build(
            frame=work,
            row_idx=pos,
            persona_state=persona_store.summary(),
        )
        stat_output = model_suite.predict(row)
        persona_output = persona_engine.run(snapshot, persona_store) if persona_engine is not None else None
        blended = blend_prediction(row, stat_output, persona_output, config)

        record = {
            "experiment": spec.name,
            "decision_date": str(pd.to_datetime(row["Date"]).date()),
            "target_date": str(pd.to_datetime(row["target_date"]).date()),
            "predicted_price": round(float(blended["predicted_price"]), 6),
            "actual_price": round(float(row["target_price"]), 6),
            "final_return": round(float(blended["final_return"]), 8),
            "actual_return": round(float(row["target_return"]), 8),
            "ci_lower": round(float(blended["ci_lower"]), 6),
            "ci_upper": round(float(blended["ci_upper"]), 6),
            "volatility_estimate": round(float(blended["volatility_estimate"]), 8),
            "direction_score": round(float(blended["direction_score"]), 6),
            "persona_weight": round(float(blended["persona_weight"]), 6),
            "persona_backend": blended["persona_backend"],
            "base_predictions": blended["base_predictions"],
            "model_weights": blended["model_weights"],
            "direction_probabilities": blended["direction_probabilities"],
            "market_memory": snapshot.to_dict(),
            "persona_output": persona_output,
        }
        daily_records.append(record)

        if persona_output and persona_output.get("votes"):
            pending_votes.append(
                {
                    "target_date": record["target_date"],
                    "actual_return": float(row["target_return"]),
                    "votes": persona_output["votes"],
                }
            )

    metrics = compute_metrics(daily_records)
    return {
        "experiment": spec.name,
        "status": "ok",
        "description": spec.description,
        "daily_records": daily_records,
        "metrics": metrics,
        "persona_memory": persona_store.summary(),
    }


def run_experiments(config: RunConfig, experiments: List[ExperimentSpec]) -> Dict:
    dataset, dataset_summary = load_integrated_dataset(config)
    feature_frame, feature_groups = build_feature_frame(dataset, config)

    results = []
    for spec in experiments:
        results.append(run_single_experiment(feature_frame, feature_groups, config, spec))

    output_dir = config.resolve_output_dir()
    write_outputs(
        output_dir=output_dir,
        config_payload={
            "run_config": asdict(config),
            "experiments": [asdict(spec) for spec in experiments],
        },
        dataset_summary=dataset_summary,
        experiment_results=results,
    )

    ok_results = [r for r in results if r.get("status") == "ok" and r.get("metrics")]
    summary = {
        "output_dir": str(output_dir),
        "experiments_requested": len(experiments),
        "experiments_completed": len(ok_results),
        "experiments_skipped": len(results) - len(ok_results),
        "best_experiment": min(ok_results, key=lambda r: r["metrics"]["mae_price"])["experiment"] if ok_results else None,
    }
    return {
        "summary": summary,
        "dataset_summary": dataset_summary,
        "results": results,
    }

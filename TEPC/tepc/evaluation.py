"""
Walk-forward evaluation for the TEPC pipeline.
"""

from __future__ import annotations

from typing import Dict, List
import json
import math
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from .config import ExperimentSpec, RunConfig
from .data import load_market_dataset
from .features import build_feature_bundle
from .modeling import fit_ensemble, predict_ensemble
from .reporting import write_outputs


def _select_feature_columns(groups: Dict[str, List[str]], include_groups: List[str]) -> List[str]:
    columns: List[str] = []
    for group in include_groups:
        columns.extend(groups.get(group, []))
    return columns


def _row_payload(row: pd.Series, feature_cols: List[str], prediction: Dict, experiment: str) -> Dict:
    return {
        "experiment": experiment,
        "decision_date": str(pd.Timestamp(row.name).date()),
        "target_date": str(pd.Timestamp(row["target_date"]).date()),
        "current_rate": float(row["current_rate"]),
        "predicted_rate": float(prediction["predicted_rate"]),
        "actual_rate": float(row["actual_rate"]),
        "predicted_return": float(prediction["predicted_return"]),
        "actual_return": float(row["future_return"]),
        "predicted_volatility": float(prediction["predicted_volatility"]),
        "actual_volatility": float(row["future_volatility"]),
        "predicted_label": prediction["predicted_label"],
        "actual_label": str(row["future_label"]),
        "direction_score": float(prediction["direction_score"]),
        "breakout_probabilities": json.dumps(prediction["breakout_probabilities"]),
        "return_model_predictions": json.dumps(prediction["return_model_predictions"]),
        "vol_model_predictions": json.dumps(prediction["vol_model_predictions"]),
        "class_model_probabilities": json.dumps(prediction["class_model_probabilities"]),
        "return_model_weights": json.dumps(prediction["return_model_weights"]),
        "vol_model_weights": json.dumps(prediction["vol_model_weights"]),
        "class_model_weights": json.dumps(prediction["class_model_weights"]),
        "feature_count": len(feature_cols),
    }


def _compute_metrics(records: List[Dict]) -> Dict:
    df = pd.DataFrame(records)
    return {
        "n_days": int(len(df)),
        "breakout_accuracy": float(accuracy_score(df["actual_label"], df["predicted_label"])),
        "macro_f1": float(f1_score(df["actual_label"], df["predicted_label"], average="macro")),
        "mae_return": float(mean_absolute_error(df["actual_return"], df["predicted_return"])),
        "rmse_return": float(math.sqrt(mean_squared_error(df["actual_return"], df["predicted_return"]))),
        "mae_volatility": float(mean_absolute_error(df["actual_volatility"], df["predicted_volatility"])),
        "bias_return": float((df["predicted_return"] - df["actual_return"]).mean()),
    }


def run_single_experiment(bundle, spec: ExperimentSpec, config: RunConfig) -> Dict:
    frame = bundle.frame.sort_index().copy()
    feature_cols = _select_feature_columns(bundle.groups, spec.include_groups)
    if not feature_cols:
        raise ValueError(f"Experiment {spec.name} selected no features.")

    test_index = list(frame.index[-config.test_days :])
    artifacts = None
    records: List[Dict] = []

    for test_pos, date in enumerate(test_index):
        row = frame.loc[date]
        train = frame[(frame.index < date) & (frame["target_date"] <= date)].copy()
        if len(train) < config.train_min_days:
            continue

        if artifacts is None or test_pos % max(config.refit_frequency, 1) == 0:
            artifacts = fit_ensemble(
                train_df=train,
                feature_cols=feature_cols,
                validation_days=config.validation_days,
                seed=config.random_seed,
            )

        prediction = predict_ensemble(artifacts, row, feature_cols)
        records.append(_row_payload(row, feature_cols, prediction, spec.name))

    metrics = _compute_metrics(records) if records else {}
    return {
        "experiment": spec.name,
        "description": spec.description,
        "include_groups": spec.include_groups,
        "feature_count": len(feature_cols),
        "daily_records": records,
        "metrics": metrics,
    }


def run_experiments(config: RunConfig, experiments: List[ExperimentSpec]) -> Dict:
    dataset = load_market_dataset(config)
    bundle = build_feature_bundle(dataset, config)
    output_dir = config.resolve_output_dir()

    results = [run_single_experiment(bundle, spec, config) for spec in experiments]
    write_outputs(output_dir, config, bundle, results)

    completed = [result for result in results if result.get("metrics")]
    ranked = sorted(
        completed,
        key=lambda item: (-item["metrics"]["macro_f1"], item["metrics"]["mae_return"]),
    )
    best = ranked[0]["experiment"] if ranked else None

    return {
        "summary": {
            "output_dir": str(output_dir),
            "experiments_requested": len(experiments),
            "experiments_completed": len(completed),
            "best_experiment": best,
        },
        "results": results,
        "dataset_summary": bundle.dataset_summary,
    }

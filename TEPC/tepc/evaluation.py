"""
Walk-forward evaluation for the TEPC pipeline.
"""

from __future__ import annotations

from typing import Dict, List
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from .config import ExperimentSpec, RunConfig
from .data import load_market_dataset
from .features import build_feature_bundle
from .modeling import fit_ensemble, predict_ensemble
from .personas import MarketMemoryBuilder, PersonaMemoryStore, RulePersonaEngine
from .reporting import write_outputs


def _select_feature_columns(groups: Dict[str, List[str]], include_groups: List[str]) -> List[str]:
    columns: List[str] = []
    for group in include_groups:
        columns.extend(groups.get(group, []))
    return columns


def _row_payload(row: pd.Series, feature_cols: List[str], prediction: Dict, experiment: str) -> Dict:
    payload = {
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
    if "persona_consensus" in prediction:
        payload["persona_consensus"] = prediction["persona_consensus"]
        payload["persona_expected_return"] = float(prediction.get("persona_expected_return", 0.0))
        payload["persona_direction_score"] = float(prediction.get("persona_direction_score", 0.0))
        payload["persona_confidence"] = float(prediction.get("persona_confidence", 0.0))
        payload["persona_entropy"] = float(prediction.get("persona_entropy", 1.0))
        payload["persona_weight"] = float(prediction.get("persona_weight", 0.0))
        payload["persona_breakout_probabilities"] = json.dumps(prediction.get("persona_breakout_probabilities", {}))
        payload["persona_votes"] = json.dumps(prediction.get("persona_votes", []))
    return payload


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


def _persona_prediction(row: pd.Series, persona_result: Dict) -> Dict:
    current_rate = float(row["current_rate"])
    predicted_return = float(persona_result["expected_return"])
    predicted_rate = current_rate * (1.0 + predicted_return)
    predicted_volatility = max(abs(predicted_return), 1e-6)
    return {
        "predicted_return": predicted_return,
        "predicted_rate": float(predicted_rate),
        "predicted_volatility": float(predicted_volatility),
        "predicted_label": str(persona_result["consensus"]),
        "direction_score": float(persona_result["direction_score"]),
        "breakout_probabilities": dict(persona_result["breakout_probabilities"]),
        "return_model_predictions": {"persona": predicted_return},
        "vol_model_predictions": {"persona": predicted_volatility},
        "class_model_probabilities": {"persona": persona_result["breakout_probabilities"]},
        "return_model_weights": {"persona": 1.0},
        "vol_model_weights": {"persona": 1.0},
        "class_model_weights": {"persona": 1.0},
        "persona_consensus": str(persona_result["consensus"]),
        "persona_expected_return": predicted_return,
        "persona_direction_score": float(persona_result["direction_score"]),
        "persona_confidence": float(persona_result["confidence"]),
        "persona_entropy": float(persona_result["entropy"]),
        "persona_weight": 1.0,
        "persona_breakout_probabilities": dict(persona_result["breakout_probabilities"]),
        "persona_votes": list(persona_result["votes"]),
    }


def _blend_predictions(ml_prediction: Dict, persona_result: Dict, persona_summary: Dict) -> Dict:
    hit_rate = float(persona_summary.get("mean_directional_hit_rate", 0.5))
    persona_weight = float(np.clip(0.12 + 0.28 * persona_result["confidence"] + 0.25 * (hit_rate - 0.5), 0.10, 0.45))

    blended_return = (1.0 - persona_weight) * float(ml_prediction["predicted_return"]) + persona_weight * float(persona_result["expected_return"])
    blended_vol = (1.0 - persona_weight) * float(ml_prediction["predicted_volatility"]) + persona_weight * max(abs(persona_result["expected_return"]), 1e-6)
    blended_direction_score = (
        (1.0 - persona_weight) * float(ml_prediction["direction_score"])
        + persona_weight * float(persona_result["direction_score"])
    )

    ml_probs = dict(ml_prediction["breakout_probabilities"])
    persona_probs = dict(persona_result["breakout_probabilities"])
    blended_probs = {
        label: (1.0 - persona_weight) * float(ml_probs.get(label, 0.0)) + persona_weight * float(persona_probs.get(label, 0.0))
        for label in ["down", "range", "up"]
    }
    predicted_label = max(blended_probs.items(), key=lambda item: item[1])[0]

    payload = dict(ml_prediction)
    payload.update(
        {
            "predicted_return": float(blended_return),
            "predicted_rate": float(ml_prediction["predicted_rate"] / (1.0 + float(ml_prediction["predicted_return"])) * (1.0 + blended_return)),
            "predicted_volatility": float(max(blended_vol, 1e-6)),
            "predicted_label": predicted_label,
            "direction_score": float(blended_direction_score),
            "breakout_probabilities": blended_probs,
            "persona_consensus": str(persona_result["consensus"]),
            "persona_expected_return": float(persona_result["expected_return"]),
            "persona_direction_score": float(persona_result["direction_score"]),
            "persona_confidence": float(persona_result["confidence"]),
            "persona_entropy": float(persona_result["entropy"]),
            "persona_weight": persona_weight,
            "persona_breakout_probabilities": persona_probs,
            "persona_votes": list(persona_result["votes"]),
        }
    )
    return payload


def run_single_experiment(bundle, spec: ExperimentSpec, config: RunConfig) -> Dict:
    frame = bundle.frame.sort_index().copy()
    feature_cols = _select_feature_columns(bundle.groups, spec.include_groups)
    if spec.prediction_mode in {"ml", "blend"} and not feature_cols:
        raise ValueError(f"Experiment {spec.name} selected no features.")

    test_index = list(frame.index[-config.test_days :])
    artifacts = None
    records: List[Dict] = []
    persona_store = PersonaMemoryStore()
    persona_builder = MarketMemoryBuilder()
    persona_engine = RulePersonaEngine(breakout_threshold=config.breakout_threshold)
    pending_persona_records: List[Dict] = []

    for test_pos, date in enumerate(test_index):
        row = frame.loc[date]
        train = frame[(frame.index < date) & (frame["target_date"] <= date)].copy()
        if len(train) < config.train_min_days:
            continue

        current_date = pd.Timestamp(date)
        unsettled = []
        for record in pending_persona_records:
            if pd.Timestamp(record["maturity_date"]) < current_date:
                persona_store.record_votes(
                    record["vote_objects"],
                    record["actual_return"],
                    actual_label=record["actual_label"],
                )
            else:
                unsettled.append(record)
        pending_persona_records = unsettled

        row_pos = int(frame.index.get_loc(date))
        persona_result = None
        if spec.prediction_mode in {"persona", "blend"}:
            snapshot = persona_builder.build(frame, row_pos)
            persona_result = persona_engine.run(snapshot, persona_store)
            pending_persona_records.append(
                {
                    "maturity_date": pd.Timestamp(row["target_date"]),
                    "vote_objects": list(persona_result.get("vote_objects", [])),
                    "actual_return": float(row["future_return"]),
                    "actual_label": str(row["future_label"]),
                }
            )

        ml_prediction = None
        if spec.prediction_mode in {"ml", "blend"}:
            if artifacts is None or test_pos % max(config.refit_frequency, 1) == 0:
                artifacts = fit_ensemble(
                    train_df=train,
                    feature_cols=feature_cols,
                    validation_days=config.validation_days,
                    seed=config.random_seed,
                )
            ml_prediction = predict_ensemble(artifacts, row, feature_cols)

        if spec.prediction_mode == "ml":
            prediction = ml_prediction
        elif spec.prediction_mode == "persona":
            prediction = _persona_prediction(row, persona_result)
        elif spec.prediction_mode == "blend":
            prediction = _blend_predictions(ml_prediction, persona_result, persona_store.summary())
        else:
            raise ValueError(f"Unknown prediction mode: {spec.prediction_mode}")

        records.append(_row_payload(row, feature_cols, prediction, spec.name))

    metrics = _compute_metrics(records) if records else {}
    return {
        "experiment": spec.name,
        "description": spec.description,
        "include_groups": spec.include_groups,
        "prediction_mode": spec.prediction_mode,
        "feature_count": len(feature_cols),
        "daily_records": records,
        "metrics": metrics,
        "persona_summary": persona_store.summary() if spec.prediction_mode in {"persona", "blend"} else None,
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

"""
Model fitting and prediction for the TEPC pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None


CLASS_ORDER = np.array([-1, 0, 1], dtype=int)


class ConstantRegressor:
    def __init__(self):
        self.value = 0.0

    def fit(self, X, y):
        self.value = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class ConstantClassifier:
    def __init__(self, classes: Iterable[int] = CLASS_ORDER):
        self.classes_ = np.array(list(classes), dtype=int)
        self.majority_class = 0

    def fit(self, X, y):
        if len(y):
            values, counts = np.unique(y, return_counts=True)
            self.majority_class = int(values[np.argmax(counts)])
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class, dtype=int)

    def predict_proba(self, X):
        arr = np.zeros((len(X), len(self.classes_)), dtype=float)
        class_idx = int(np.where(self.classes_ == self.majority_class)[0][0])
        arr[:, class_idx] = 1.0
        return arr


@dataclass
class EnsembleArtifacts:
    return_models: Dict[str, object]
    return_weights: Dict[str, float]
    vol_models: Dict[str, object]
    vol_weights: Dict[str, float]
    class_models: Dict[str, object]
    class_weights: Dict[str, float]


def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(value, 0.0) for value in raw.values()))
    if total <= 0:
        uniform = 1.0 / max(len(raw), 1)
        return {key: uniform for key in raw}
    return {key: max(value, 0.0) / total for key, value in raw.items()}


def _make_regressors(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            random_state=seed,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=3,
            random_state=seed,
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            objective="reg:squarederror",
            verbosity=0,
        )
    return models


def _make_classifiers(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=3,
            random_state=seed,
        ),
    }
    return models


def _train_regression_models(X_train, y_train, X_val, y_val, seed: int) -> tuple[Dict[str, object], Dict[str, float]]:
    fitted: Dict[str, object] = {}
    scores: Dict[str, float] = {}

    for name, model in _make_regressors(seed).items():
        model.fit(X_train, y_train)
        if X_val is not None and len(X_val):
            pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            scores[name] = float(1.0 / (mae + 1e-6))
        else:
            scores[name] = 1.0
        fitted[name] = model.fit(np.vstack([X_train, X_val]) if X_val is not None and len(X_val) else X_train, np.concatenate([y_train, y_val]) if X_val is not None and len(X_val) else y_train)

    return fitted, _normalize_weights(scores)


def _train_classification_models(X_train, y_train, X_val, y_val, seed: int) -> tuple[Dict[str, object], Dict[str, float]]:
    fitted: Dict[str, object] = {}
    scores: Dict[str, float] = {}

    if len(np.unique(y_train)) < 2:
        model = ConstantClassifier().fit(X_train, y_train)
        return {"constant": model}, {"constant": 1.0}

    for name, model in _make_classifiers(seed).items():
        model.fit(X_train, y_train)
        if X_val is not None and len(X_val):
            pred = model.predict(X_val)
            scores[name] = float(accuracy_score(y_val, pred))
        else:
            scores[name] = 1.0
        fitted[name] = model.fit(np.vstack([X_train, X_val]) if X_val is not None and len(X_val) else X_train, np.concatenate([y_train, y_val]) if X_val is not None and len(X_val) else y_train)

    return fitted, _normalize_weights(scores)


def fit_ensemble(train_df, feature_cols: List[str], validation_days: int, seed: int) -> EnsembleArtifacts:
    X = train_df[feature_cols].to_numpy(dtype=float)
    y_return = train_df["future_return"].to_numpy(dtype=float)
    y_vol = train_df["future_volatility"].to_numpy(dtype=float)
    y_label = train_df["future_label_int"].to_numpy(dtype=int)

    if len(train_df) <= max(validation_days + 20, 40):
        X_train, X_val = X, None
        y_return_train, y_return_val = y_return, None
        y_vol_train, y_vol_val = y_vol, None
        y_label_train, y_label_val = y_label, None
    else:
        split = len(train_df) - validation_days
        X_train, X_val = X[:split], X[split:]
        y_return_train, y_return_val = y_return[:split], y_return[split:]
        y_vol_train, y_vol_val = y_vol[:split], y_vol[split:]
        y_label_train, y_label_val = y_label[:split], y_label[split:]

    return_models, return_weights = _train_regression_models(X_train, y_return_train, X_val, y_return_val, seed)
    vol_models, vol_weights = _train_regression_models(X_train, y_vol_train, X_val, y_vol_val, seed + 17)
    class_models, class_weights = _train_classification_models(X_train, y_label_train, X_val, y_label_val, seed + 31)

    return EnsembleArtifacts(
        return_models=return_models,
        return_weights=return_weights,
        vol_models=vol_models,
        vol_weights=vol_weights,
        class_models=class_models,
        class_weights=class_weights,
    )


def _weighted_regression_prediction(models: Dict[str, object], weights: Dict[str, float], X_row) -> tuple[float, Dict[str, float]]:
    preds: Dict[str, float] = {}
    value = 0.0
    for name, model in models.items():
        pred = float(model.predict(X_row)[0])
        preds[name] = pred
        value += weights[name] * pred
    return float(value), preds


def _weighted_class_prediction(models: Dict[str, object], weights: Dict[str, float], X_row) -> tuple[Dict[str, float], Dict[str, Dict[int, float]]]:
    total = np.zeros(len(CLASS_ORDER), dtype=float)
    raw: Dict[str, Dict[int, float]] = {}
    for name, model in models.items():
        proba = model.predict_proba(X_row)[0]
        aligned = np.zeros(len(CLASS_ORDER), dtype=float)
        model_classes = getattr(model, "classes_", CLASS_ORDER)
        for idx, cls in enumerate(model_classes):
            target_idx = int(np.where(CLASS_ORDER == int(cls))[0][0])
            aligned[target_idx] = proba[idx]
        total += weights[name] * aligned
        raw[name] = {int(cls): float(aligned[pos]) for pos, cls in enumerate(CLASS_ORDER)}
    return (
        {
            "down": float(total[0]),
            "range": float(total[1]),
            "up": float(total[2]),
        },
        raw,
    )


def predict_ensemble(artifacts: EnsembleArtifacts, row, feature_cols: List[str]) -> Dict:
    X_row = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
    current_rate = float(row["current_rate"])

    return_pred, return_base = _weighted_regression_prediction(
        artifacts.return_models, artifacts.return_weights, X_row
    )
    vol_pred, vol_base = _weighted_regression_prediction(
        artifacts.vol_models, artifacts.vol_weights, X_row
    )
    class_probs, class_base = _weighted_class_prediction(
        artifacts.class_models, artifacts.class_weights, X_row
    )

    direction_score = class_probs["up"] - class_probs["down"]
    blended_return = 0.85 * return_pred + 0.15 * direction_score * max(abs(vol_pred), 1e-6)
    predicted_rate = current_rate * (1.0 + blended_return)
    predicted_label = max(class_probs.items(), key=lambda item: item[1])[0]

    return {
        "predicted_return": float(blended_return),
        "predicted_rate": float(predicted_rate),
        "predicted_volatility": float(max(vol_pred, 1e-6)),
        "predicted_label": predicted_label,
        "direction_score": float(direction_score),
        "breakout_probabilities": class_probs,
        "return_model_predictions": return_base,
        "vol_model_predictions": vol_base,
        "class_model_probabilities": class_base,
        "return_model_weights": artifacts.return_weights,
        "vol_model_weights": artifacts.vol_weights,
        "class_model_weights": artifacts.class_weights,
    }

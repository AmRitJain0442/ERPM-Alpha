"""
Plotting utilities for TEPC backtest outputs.
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_METRICS = [
    ("breakout_accuracy", "Breakout Accuracy", False),
    ("macro_f1", "Macro F1", False),
    ("mae_return", "MAE Return", True),
    ("mae_volatility", "MAE Volatility", True),
]


def _rank_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    return ranked.sort_values(["macro_f1", "mae_return"], ascending=[False, True]).reset_index(drop=True)


def _top_experiments(metrics: pd.DataFrame, top_n: int = 3) -> list[str]:
    ranked = _rank_metrics(metrics)
    return ranked["experiment"].head(top_n).tolist()


def _style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)


def plot_single_run(run_dir: Path, output_dir: Path | None = None) -> dict:
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "ablation_summary.csv"
    predictions_path = run_dir / "daily_predictions.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing ablation summary: {metrics_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing daily predictions: {predictions_path}")

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["decision_date", "target_date"])
    ranked = _rank_metrics(metrics)
    top_experiments = _top_experiments(metrics)

    _plot_metric_bars(ranked, output_dir / "metric_bars.png")
    _plot_accuracy_vs_error(ranked, output_dir / "accuracy_vs_error.png")
    _plot_prediction_overlay(predictions, ranked, output_dir / "predictions_best3.png")
    _plot_rolling_error(predictions, ranked, output_dir / "rolling_abs_error_top3.png")
    _write_plot_summary(ranked, top_experiments, output_dir / "plot_summary.md")

    return {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "plots": [
            "metric_bars.png",
            "accuracy_vs_error.png",
            "predictions_best3.png",
            "rolling_abs_error_top3.png",
            "plot_summary.md",
        ],
    }


def plot_multi_run(run_dirs: list[Path], output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir).resolve()
        metrics_path = run_dir / "ablation_summary.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        metrics["run_name"] = run_dir.name
        rows.append(metrics)

    if not rows:
        raise FileNotFoundError("No ablation_summary.csv files found in the provided run directories.")

    combined = pd.concat(rows, ignore_index=True)
    combined["run_experiment"] = combined["run_name"] + " :: " + combined["experiment"]
    combined = combined.sort_values(["macro_f1", "mae_return"], ascending=[False, True]).reset_index(drop=True)

    _plot_multi_leaderboard(combined, output_dir / "leaderboard_all_runs.png")
    _plot_multi_pareto(combined, output_dir / "pareto_all_runs.png")

    combined.to_csv(output_dir / "combined_ablation_summary.csv", index=False)
    (output_dir / "plot_summary.md").write_text(
        "\n".join(
            [
                "# TEPC Cross-Run Plot Summary",
                "",
                f"Runs compared: {len({Path(run_dir).name for run_dir in run_dirs})}",
                "",
                combined.head(15).to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "plots": [
            "leaderboard_all_runs.png",
            "pareto_all_runs.png",
            "combined_ablation_summary.csv",
            "plot_summary.md",
        ],
    }


def _plot_metric_bars(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = ["#0f4c5c", "#2c7da0", "#84a98c", "#bc4749", "#dda15e"]

    for ax, (metric, title, lower_is_better) in zip(axes.flat, PLOT_METRICS):
        ordered = metrics.sort_values(metric, ascending=lower_is_better)
        ax.bar(ordered["experiment"], ordered[metric], color=colors[: len(ordered)])
        _style_axes(ax, title, ylabel=metric)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("TEPC Experiment Metrics", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_accuracy_vs_error(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.scatter(
        metrics["mae_return"],
        metrics["breakout_accuracy"],
        s=120,
        c=np.linspace(0.2, 0.9, len(metrics)),
        cmap="viridis",
    )
    for _, row in metrics.iterrows():
        ax.annotate(
            row["experiment"],
            (row["mae_return"], row["breakout_accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    _style_axes(ax, "Accuracy vs Return Error", xlabel="MAE Return", ylabel="Breakout Accuracy")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_prediction_overlay(predictions: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    top = _top_experiments(metrics)
    subset = predictions[predictions["experiment"].isin(top)].copy()
    subset = subset.sort_values(["decision_date", "experiment"])

    actual = (
        subset[["decision_date", "actual_rate"]]
        .drop_duplicates(subset=["decision_date"])
        .sort_values("decision_date")
        .set_index("decision_date")
    )
    predicted = subset.pivot(index="decision_date", columns="experiment", values="predicted_rate").sort_index()

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.plot(actual.index, actual["actual_rate"], label="actual_rate", color="#111111", linewidth=2.3)
    palette = ["#2a9d8f", "#e76f51", "#264653"]
    for color, experiment in zip(palette, top):
        if experiment in predicted.columns:
            ax.plot(predicted.index, predicted[experiment], label=experiment, linewidth=1.8, color=color)

    _style_axes(ax, "Actual vs Predicted INR/USD", xlabel="Decision Date", ylabel="USD/INR")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_rolling_error(predictions: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    top = _top_experiments(metrics)
    subset = predictions[predictions["experiment"].isin(top)].copy()
    subset["abs_error"] = (subset["predicted_rate"] - subset["actual_rate"]).abs()
    subset = subset.sort_values(["experiment", "decision_date"])

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    palette = ["#2a9d8f", "#e76f51", "#264653"]
    for color, experiment in zip(palette, top):
        exp_df = subset[subset["experiment"] == experiment].copy()
        exp_df["rolling_abs_error_10"] = exp_df["abs_error"].rolling(10, min_periods=5).mean()
        ax.plot(
            exp_df["decision_date"],
            exp_df["rolling_abs_error_10"],
            label=experiment,
            linewidth=1.8,
            color=color,
        )

    _style_axes(ax, "Rolling Absolute Error (10-day)", xlabel="Decision Date", ylabel="Abs Error")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_multi_leaderboard(metrics: pd.DataFrame, path: Path) -> None:
    top = metrics.head(12).copy()
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    ax.barh(top["run_experiment"], top["macro_f1"], color="#2c7da0")
    ax.invert_yaxis()
    _style_axes(ax, "Cross-Run Leaderboard by Macro F1", xlabel="Macro F1")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_multi_pareto(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.scatter(metrics["mae_return"], metrics["breakout_accuracy"], s=90, c="#bc4749", alpha=0.8)
    for _, row in metrics.head(15).iterrows():
        ax.annotate(
            row["run_experiment"],
            (row["mae_return"], row["breakout_accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    _style_axes(ax, "Cross-Run Pareto View", xlabel="MAE Return", ylabel="Breakout Accuracy")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_plot_summary(metrics: pd.DataFrame, top_experiments: list[str], path: Path) -> None:
    payload = {
        "best_experiment": top_experiments[0] if top_experiments else None,
        "top_experiments": top_experiments,
        "metric_table": metrics.to_dict(orient="records"),
    }
    lines = [
        "# TEPC Plot Summary",
        "",
        f"Best experiment: `{payload['best_experiment']}`",
        "",
        metrics.to_markdown(index=False),
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

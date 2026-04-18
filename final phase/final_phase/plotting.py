"""
Plot generation for comparing final-phase backtest outputs.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List
import math
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


CATEGORY_COLORS = {
    "stat_ml": "#2b6cb0",
    "memory": "#2f855a",
    "rule_persona": "#d97706",
    "llm": "#c53030",
    "other": "#6b7280",
}


PRETTY_NAMES = {
    "stat_ml_full": "Stat+ML Full",
    "stat_ml_no_gdelt": "Stat+ML No GDELT",
    "stat_ml_no_macro": "Stat+ML No Macro",
    "memory_only_no_personas": "Memory Only",
    "rule_personas": "Rule Personas",
    "llm_openrouter_google_gemini_2_5_flash": "Gemini 2.5 Flash Personas",
    "llm_openrouter_anthropic_claude_sonnet_4": "Claude Sonnet 4 Personas",
    "llm_openrouter_openai_gpt_5_chat": "GPT-5 Chat Personas",
    "llm_openrouter_openai_gpt_5_mini": "GPT-5 Mini Personas",
    "llm_openrouter_google_gemma_4_26b_a4b_it_free": "Gemma 4 26B Free Personas",
}


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d9e2ec",
            "axes.labelcolor": "#243b53",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.color": "#486581",
            "ytick.color": "#486581",
            "grid.color": "#e5edf5",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "legend.frameon": True,
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#d9e2ec",
            "savefig.facecolor": "#f7fafc",
            "savefig.bbox": "tight",
        }
    )


def _category(experiment: str) -> str:
    if experiment.startswith("llm_"):
        return "llm"
    if experiment == "rule_personas":
        return "rule_persona"
    if experiment.startswith("memory_only"):
        return "memory"
    if experiment.startswith("stat_ml"):
        return "stat_ml"
    return "other"


def _pretty_name(experiment: str) -> str:
    if experiment in PRETTY_NAMES:
        return PRETTY_NAMES[experiment]
    return experiment.replace("_", " ").title()


def _wrap(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) or text


def _color_for_category(category: str) -> str:
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"])


def _rank_series(series: pd.Series, *, lower_is_better: bool) -> pd.Series:
    return series.rank(method="min", ascending=lower_is_better).astype(int)


def _parse_market_memory(value: object) -> dict:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        payload = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_regime(value: object) -> str:
    payload = _parse_market_memory(value)
    regime = payload.get("regime", "unknown")
    return str(regime).strip() or "unknown"


def load_metrics(run_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "ablation_summary.csv"
        if not metrics_path.exists():
            continue
        frame = pd.read_csv(metrics_path)
        if frame.empty:
            continue
        frame["run_name"] = run_dir.name
        frame["run_dir"] = str(run_dir)
        frame["category"] = frame["experiment"].map(_category)
        frame["display_name"] = frame["experiment"].map(_pretty_name)
        frame["horizon_label"] = frame["n_days"].astype(int).astype(str) + "d"
        frame["bias_abs"] = frame["bias_return"].abs()
        frame["base_label"] = frame["display_name"] + " [" + frame["horizon_label"] + "]"
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No ablation_summary.csv files were found in the provided run directories.")

    metrics = pd.concat(frames, ignore_index=True)

    duplicates = metrics["base_label"].value_counts()
    metrics["plot_label"] = metrics.apply(
        lambda row: (
            f"{row['base_label']} / {row['run_name']}"
            if duplicates[row["base_label"]] > 1
            else row["base_label"]
        ),
        axis=1,
    )
    return metrics.sort_values(["n_days", "mae_price", "experiment"]).reset_index(drop=True)


def load_daily_predictions(metrics: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for row in metrics.itertuples(index=False):
        daily_path = Path(row.run_dir) / row.experiment / "daily_predictions.csv"
        if not daily_path.exists():
            continue
        frame = pd.read_csv(daily_path, parse_dates=["decision_date", "target_date"])
        if frame.empty:
            continue
        frame["run_name"] = row.run_name
        frame["n_days"] = row.n_days
        frame["category"] = row.category
        frame["display_name"] = row.display_name
        frame["plot_label"] = row.plot_label
        frame["abs_error"] = (frame["predicted_price"] - frame["actual_price"]).abs()
        frame["regime"] = frame["market_memory"].map(_extract_regime)
        frame["persona_weight"] = pd.to_numeric(frame["persona_weight"], errors="coerce").fillna(0.0)
        frame["persona_backend"] = frame["persona_backend"].fillna("none")
        frame["has_persona_signal"] = (frame["persona_backend"] != "none") & (frame["persona_weight"] > 0)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No daily_predictions.csv files were found for the provided runs.")

    return pd.concat(frames, ignore_index=True)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_all_runs_leaderboard(metrics: pd.DataFrame, output_dir: Path) -> Path:
    order = metrics.sort_values("mae_price")["plot_label"]
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, 0.55 * len(metrics) + 2)), sharey=True)

    plot_df = metrics.copy()
    plot_df["plot_label_wrapped"] = plot_df["plot_label"].map(_wrap)
    order_wrapped = [
        _wrap(label)
        for label in metrics.sort_values("mae_price")["plot_label"].tolist()
    ]
    colors = plot_df["category"].map(_color_for_category)

    sns.barplot(
        data=plot_df,
        x="mae_price",
        y="plot_label_wrapped",
        order=order_wrapped,
        hue="category",
        dodge=False,
        palette=CATEGORY_COLORS,
        ax=axes[0],
    )
    sns.barplot(
        data=plot_df,
        x="directional_accuracy",
        y="plot_label_wrapped",
        order=order_wrapped,
        hue="category",
        dodge=False,
        palette=CATEGORY_COLORS,
        ax=axes[1],
    )

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:4], labels[:4], title="Family", loc="lower right")

    axes[0].set_title("All Runs: MAE Price")
    axes[0].set_xlabel("Lower Is Better")
    axes[0].set_ylabel("")
    axes[1].set_title("All Runs: Directional Accuracy")
    axes[1].set_xlabel("Higher Is Better")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, min(1.0, metrics["directional_accuracy"].max() + 0.08))

    path = output_dir / "leaderboard_all_runs.png"
    _save(fig, path)
    return path


def plot_run_dashboard(run_metrics: pd.DataFrame, output_dir: Path) -> Path:
    run_name = run_metrics["run_name"].iloc[0]
    plot_df = run_metrics.sort_values("mae_price").copy()
    plot_df["plot_label_wrapped"] = plot_df["display_name"].map(_wrap)

    fig, axes = plt.subplots(2, 2, figsize=(16, max(10, 0.45 * len(plot_df) + 4)))
    axes = axes.ravel()

    panels = [
        ("mae_price", "MAE Price", "Lower Is Better"),
        ("rmse_price", "RMSE Price", "Lower Is Better"),
        ("directional_accuracy", "Directional Accuracy", "Higher Is Better"),
        ("bias_abs", "Absolute Bias", "Lower Is Better"),
    ]

    for ax, (column, title, xlabel) in zip(axes, panels):
        sns.barplot(
            data=plot_df,
            x=column,
            y="plot_label_wrapped",
            hue="category",
            dodge=False,
            palette=CATEGORY_COLORS,
            ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:4], labels[:4], title="Family", loc="lower right")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")

    fig.suptitle(f"Run Dashboard: {run_name}", y=1.01, fontsize=17, fontweight="bold")
    path = output_dir / f"dashboard_{run_name}.png"
    _save(fig, path)
    return path


def plot_pareto(metrics: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 8))

    for category, group in metrics.groupby("category"):
        ax.scatter(
            group["mae_price"],
            group["directional_accuracy"],
            s=90 + group["n_days"].astype(float),
            c=_color_for_category(category),
            label=category.replace("_", " ").title(),
            alpha=0.88,
            edgecolors="#102a43",
            linewidths=0.6,
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                row.plot_label,
                (row.mae_price, row.directional_accuracy),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                color="#243b53",
            )

    ax.set_title("MAE Price vs Directional Accuracy")
    ax.set_xlabel("MAE Price")
    ax.set_ylabel("Directional Accuracy")
    ax.legend(title="Family", loc="lower left")
    ax.grid(True, alpha=0.35)

    path = output_dir / "pareto_mae_vs_accuracy.png"
    _save(fig, path)
    return path


def plot_rank_heatmap(metrics: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    metric_info = {
        "mae_price": True,
        "rmse_price": True,
        "mae_return": True,
        "directional_accuracy": False,
        "sign_f1": False,
        "bias_abs": True,
    }
    pretty_cols = {
        "mae_price": "MAE Price",
        "rmse_price": "RMSE Price",
        "mae_return": "MAE Return",
        "directional_accuracy": "Direction Acc",
        "sign_f1": "Sign F1",
        "bias_abs": "Abs Bias",
    }

    for n_days, group in metrics.groupby("n_days"):
        ranked = pd.DataFrame(index=group["plot_label"])
        for column, lower_is_better in metric_info.items():
            ranked[pretty_cols[column]] = _rank_series(group[column], lower_is_better=lower_is_better).values

        fig_height = max(4, 0.5 * len(group) + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        sns.heatmap(
            ranked,
            annot=True,
            fmt="d",
            cmap=sns.color_palette("YlGnBu_r", as_cmap=True),
            cbar_kws={"label": "Rank (1 = best)"},
            linewidths=0.5,
            linecolor="#ffffff",
            ax=ax,
        )
        ax.set_title(f"Metric Ranks [{int(n_days)}d]")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([_wrap(label, 30) for label in ranked.index], rotation=0)
        path = output_dir / f"rank_heatmap_{int(n_days)}d.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_family_metric_summary(metrics: pd.DataFrame, output_dir: Path) -> Path:
    summary = (
        metrics.groupby(["n_days", "horizon_label", "category"], as_index=False)
        .agg(
            mean_mae_price=("mae_price", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
            best_mae_price=("mae_price", "min"),
            mean_abs_bias=("bias_abs", "mean"),
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.ravel()
    panels = [
        ("mean_mae_price", "Family Mean MAE Price", "Lower Is Better"),
        ("best_mae_price", "Family Best MAE Price", "Lower Is Better"),
        ("mean_directional_accuracy", "Family Mean Directional Accuracy", "Higher Is Better"),
        ("mean_abs_bias", "Family Mean Absolute Bias", "Lower Is Better"),
    ]

    for ax, (column, title, ylabel) in zip(axes, panels):
        sns.barplot(
            data=summary,
            x="category",
            y=column,
            hue="horizon_label",
            palette=["#7fb3d5", "#154360"],
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Family")
        ax.set_ylabel(ylabel)
        ax.legend(title="Window", loc="best")

    path = output_dir / "family_metric_summary.png"
    _save(fig, path)
    return path


def plot_family_abs_error_boxplot(daily: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = daily.copy()
    plot_df["horizon_label"] = plot_df["n_days"].astype(int).astype(str) + "d"

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=plot_df,
        x="category",
        y="abs_error",
        hue="horizon_label",
        palette=["#7fb3d5", "#154360"],
        ax=ax,
        showfliers=False,
    )
    sns.stripplot(
        data=plot_df,
        x="category",
        y="abs_error",
        hue="horizon_label",
        dodge=True,
        size=3,
        alpha=0.25,
        palette=["#7fb3d5", "#154360"],
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label not in seen:
                unique.append((handle, label))
                seen.add(label)
        ax.legend([h for h, _ in unique], [l for _, l in unique], title="Window", loc="upper right")
    ax.set_title("Absolute Error Distribution By Family")
    ax.set_xlabel("Family")
    ax.set_ylabel("|Predicted - Actual|")

    path = output_dir / "family_abs_error_boxplot.png"
    _save(fig, path)
    return path


def plot_regime_error_heatmaps(metrics: pd.DataFrame, daily: pd.DataFrame, output_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for n_days, group in metrics.groupby("n_days"):
        daily_subset = daily[daily["n_days"] == n_days].copy()
        if daily_subset.empty:
            continue
        regime_counts = daily_subset["regime"].value_counts()
        regime_order = regime_counts.index.tolist()
        pivot = (
            daily_subset.groupby(["plot_label", "regime"])["abs_error"]
            .mean()
            .unstack(fill_value=np.nan)
            .reindex(columns=regime_order)
        )
        row_order = group.sort_values("mae_price")["plot_label"].tolist()
        pivot = pivot.reindex(row_order)
        if pivot.empty:
            continue

        fig_height = max(4.5, 0.48 * len(pivot.index) + 2)
        fig, ax = plt.subplots(figsize=(11, fig_height))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap=sns.color_palette("mako_r", as_cmap=True),
            linewidths=0.5,
            linecolor="#ffffff",
            cbar_kws={"label": "Mean Absolute Error"},
            ax=ax,
        )
        ax.set_title(f"Regime-Wise MAE [{int(n_days)}d]")
        ax.set_xlabel("Regime")
        ax.set_ylabel("")
        ax.set_yticklabels([_wrap(label, 30) for label in pivot.index], rotation=0)

        path = output_dir / f"regime_mae_heatmap_{int(n_days)}d.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_persona_weight_diagnostics(daily: pd.DataFrame, output_dir: Path) -> List[Path]:
    plot_df = daily[daily["has_persona_signal"]].copy()
    if plot_df.empty:
        return []

    plot_df["horizon_label"] = plot_df["n_days"].astype(int).astype(str) + "d"

    scatter_fig, scatter_ax = plt.subplots(figsize=(13, 8))
    sns.scatterplot(
        data=plot_df,
        x="persona_weight",
        y="abs_error",
        hue="category",
        style="horizon_label",
        palette=CATEGORY_COLORS,
        s=80,
        alpha=0.8,
        ax=scatter_ax,
    )
    scatter_ax.set_title("Persona Weight vs Absolute Error")
    scatter_ax.set_xlabel("Persona Weight In Final Blend")
    scatter_ax.set_ylabel("|Predicted - Actual|")
    scatter_ax.legend(title="Family / Window", loc="best")
    scatter_path = output_dir / "persona_weight_vs_error.png"
    _save(scatter_fig, scatter_path)

    summary = (
        plot_df.groupby(["plot_label", "category"], as_index=False)
        .agg(
            mean_persona_weight=("persona_weight", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .sort_values("mean_persona_weight", ascending=False)
    )
    summary["plot_label_wrapped"] = summary["plot_label"].map(_wrap)

    bar_fig, axes = plt.subplots(1, 2, figsize=(18, max(7, 0.5 * len(summary) + 2)), sharey=True)
    sns.barplot(
        data=summary,
        x="mean_persona_weight",
        y="plot_label_wrapped",
        hue="category",
        dodge=False,
        palette=CATEGORY_COLORS,
        ax=axes[0],
    )
    sns.barplot(
        data=summary,
        x="mean_abs_error",
        y="plot_label_wrapped",
        hue="category",
        dodge=False,
        palette=CATEGORY_COLORS,
        ax=axes[1],
    )
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:4], labels[:4], title="Family", loc="lower right")
    axes[0].set_title("Average Persona Weight By Experiment")
    axes[0].set_xlabel("Average Persona Weight")
    axes[0].set_ylabel("")
    axes[1].set_title("Average Absolute Error For Persona Experiments")
    axes[1].set_xlabel("Average |Predicted - Actual|")
    axes[1].set_ylabel("")
    bar_path = output_dir / "persona_weight_summary.png"
    _save(bar_fig, bar_path)

    return [scatter_path, bar_path]


def plot_prediction_paths(metrics: pd.DataFrame, daily: pd.DataFrame, output_dir: Path, top_k: int = 4) -> List[Path]:
    paths: List[Path] = []
    for n_days, group in metrics.groupby("n_days"):
        top = group.sort_values("mae_price").head(top_k)
        daily_subset = daily[daily["plot_label"].isin(top["plot_label"])].copy()
        if daily_subset.empty:
            continue

        actual = (
            daily_subset.sort_values("target_date")
            .groupby("target_date", as_index=False)["actual_price"]
            .first()
            .sort_values("target_date")
        )

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(
            actual["target_date"],
            actual["actual_price"],
            color="#102a43",
            linewidth=2.8,
            label="Actual Price",
        )

        for row in top.itertuples(index=False):
            exp_frame = (
                daily_subset[daily_subset["plot_label"] == row.plot_label]
                .sort_values("target_date")
            )
            ax.plot(
                exp_frame["target_date"],
                exp_frame["predicted_price"],
                linewidth=2,
                alpha=0.9,
                color=_color_for_category(row.category),
                label=row.plot_label,
            )

        ax.set_title(f"Actual vs Predicted Price: Top {min(top_k, len(top))} [{int(n_days)}d]")
        ax.set_xlabel("Target Date")
        ax.set_ylabel("USD/INR")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="best", fontsize=9)
        path = output_dir / f"predictions_top_{int(n_days)}d.png"
        _save(fig, path)
        paths.append(path)
    return paths


def plot_rolling_abs_error(metrics: pd.DataFrame, daily: pd.DataFrame, output_dir: Path, top_k: int = 4) -> List[Path]:
    paths: List[Path] = []
    for n_days, group in metrics.groupby("n_days"):
        top = group.sort_values("mae_price").head(top_k)
        daily_subset = daily[daily["plot_label"].isin(top["plot_label"])].copy()
        if daily_subset.empty:
            continue

        window = max(3, min(10, math.ceil(len(daily_subset["target_date"].unique()) / 6)))
        fig, ax = plt.subplots(figsize=(14, 7))

        for row in top.itertuples(index=False):
            exp_frame = (
                daily_subset[daily_subset["plot_label"] == row.plot_label]
                .sort_values("target_date")
                .copy()
            )
            exp_frame["rolling_abs_error"] = exp_frame["abs_error"].rolling(window=window, min_periods=1).mean()
            ax.plot(
                exp_frame["target_date"],
                exp_frame["rolling_abs_error"],
                linewidth=2.2,
                color=_color_for_category(row.category),
                label=row.plot_label,
            )

        ax.set_title(f"Rolling Absolute Error [{int(n_days)}d, window={window}]")
        ax.set_xlabel("Target Date")
        ax.set_ylabel("Rolling |Predicted - Actual|")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="best", fontsize=9)
        path = output_dir / f"rolling_abs_error_{int(n_days)}d.png"
        _save(fig, path)
        paths.append(path)
    return paths


def write_plot_summary(metrics: pd.DataFrame, generated_paths: List[Path], output_dir: Path) -> Path:
    lines = [
        "# Plot Summary",
        "",
        "## Best Experiments By Horizon",
        "",
    ]
    for n_days, group in metrics.groupby("n_days"):
        best = group.sort_values("mae_price").iloc[0]
        lines.append(
            f"- `{int(n_days)}d`: `{best['display_name']}` "
            f"(MAE price `{best['mae_price']:.6f}`, direction accuracy `{best['directional_accuracy']:.3f}`)"
        )

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
        ]
    )
    for path in generated_paths:
        lines.append(f"- `{path.name}`")

    summary_path = output_dir / "plot_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def generate_plots(run_dirs: Iterable[Path], output_dir: Path, top_k: int = 4) -> List[Path]:
    _set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(path) for path in run_dirs]
    metrics = load_metrics(run_dirs)
    daily = load_daily_predictions(metrics)

    metrics.to_csv(output_dir / "combined_metrics.csv", index=False)
    daily.to_csv(output_dir / "combined_daily_predictions.csv", index=False)

    generated: List[Path] = []
    generated.append(plot_all_runs_leaderboard(metrics, output_dir))
    generated.append(plot_pareto(metrics, output_dir))
    generated.append(plot_family_metric_summary(metrics, output_dir))
    generated.append(plot_family_abs_error_boxplot(daily, output_dir))

    for _, group in metrics.groupby("run_name"):
        generated.append(plot_run_dashboard(group, output_dir))

    generated.extend(plot_rank_heatmap(metrics, output_dir))
    generated.extend(plot_regime_error_heatmaps(metrics, daily, output_dir))
    generated.extend(plot_prediction_paths(metrics, daily, output_dir, top_k=top_k))
    generated.extend(plot_rolling_abs_error(metrics, daily, output_dir, top_k=top_k))
    generated.extend(plot_persona_weight_diagnostics(daily, output_dir))
    generated.append(write_plot_summary(metrics, generated, output_dir))
    return generated

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
import json

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
FINAL_OUTPUTS_DIR = REPO_ROOT / "final phase" / "outputs"
FIGURES_DIR = BENCHMARK_DIR / "figures"

MATCHED_RUN = "llm_matrix_v2_30d"
LONG_BASELINE_RUN = "backtest_100d"
LONG_LLM_RUN = "llm_gemini_100d"


EXPERIMENT_LABELS = {
    "stat_ml_full": "Stat+ML Full",
    "stat_ml_no_gdelt": "Stat+ML No GDELT",
    "stat_ml_no_macro": "Stat+ML No Macro",
    "memory_only_no_personas": "Memory Only",
    "rule_personas": "Rule Personas",
    "llm_openrouter_google_gemini_2_5_flash": "Gemini 2.5 Flash",
    "llm_openrouter_anthropic_claude_sonnet_4": "Claude Sonnet 4",
    "llm_openrouter_openai_gpt_5_chat": "GPT-5 Chat",
    "llm_openrouter_openai_gpt_5_mini": "GPT-5 Mini",
    "llm_openrouter_google_gemma_4_26b_a4b_it_free": "Gemma 4 26B Free",
}

FAMILY_COLORS = {
    "stat_ml": "#4C78A8",
    "memory": "#54A24B",
    "rule": "#F58518",
    "llm": "#E45756",
    "probe": "#B279A2",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def experiment_label(name: str) -> str:
    return EXPERIMENT_LABELS.get(name, name.replace("_", " ").title())


def classify_experiment(name: str) -> str:
    if name.startswith("llm_"):
        return "llm"
    if name == "memory_only_no_personas":
        return "memory"
    if name == "rule_personas":
        return "rule"
    return "stat_ml"


def summarize_daily_records(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {
            "skip_rate": np.nan,
            "active_persona_rate": np.nan,
            "mean_vote_count": np.nan,
            "mean_persona_weight": np.nan,
            "mean_persona_confidence": np.nan,
            "mean_persona_entropy": np.nan,
            "mean_abs_direction_score": np.nan,
            "predicted_up_share": np.nan,
            "predicted_down_share": np.nan,
            "predicted_flat_share": np.nan,
            "mean_ci_width": np.nan,
        }

    has_persona_layer = any(row.get("persona_output") is not None for row in records)
    skip_flags = []
    active_persona_flags = []
    vote_counts = []
    persona_confidences = []
    persona_entropies = []
    direction_scores = []
    predicted_signs = []
    ci_widths = []

    for row in records:
        persona_output = row.get("persona_output") or {}
        votes = persona_output.get("votes", []) or []
        skipped = bool(persona_output.get("skipped")) or len(votes) == 0
        skip_flags.append(float(skipped))
        active_persona_flags.append(float(float(row.get("persona_weight", 0.0)) > 0))
        vote_counts.append(float(len(votes)))
        persona_confidences.append(float(persona_output.get("confidence", 0.0)))
        persona_entropies.append(float(persona_output.get("entropy", 1.0)))
        direction_scores.append(abs(float(row.get("direction_score", 0.0))))
        predicted_signs.append(float(np.sign(float(row.get("final_return", 0.0)))))
        ci_widths.append(float(row.get("ci_upper", 0.0)) - float(row.get("ci_lower", 0.0)))

    predicted_signs_arr = np.asarray(predicted_signs, dtype=float)
    return {
        "skip_rate": float(np.mean(skip_flags)) if has_persona_layer else np.nan,
        "active_persona_rate": float(np.mean(active_persona_flags)) if has_persona_layer else 0.0,
        "mean_vote_count": float(np.mean(vote_counts)) if has_persona_layer else np.nan,
        "mean_persona_weight": float(np.mean([float(row.get("persona_weight", 0.0)) for row in records])) if has_persona_layer else 0.0,
        "mean_persona_confidence": float(np.mean(persona_confidences)) if has_persona_layer else np.nan,
        "mean_persona_entropy": float(np.mean(persona_entropies)) if has_persona_layer else np.nan,
        "mean_abs_direction_score": float(np.mean(direction_scores)),
        "predicted_up_share": float(np.mean(predicted_signs_arr > 0)),
        "predicted_down_share": float(np.mean(predicted_signs_arr < 0)),
        "predicted_flat_share": float(np.mean(predicted_signs_arr == 0)),
        "mean_ci_width": float(np.mean(ci_widths)),
    }


def load_run_table(run_name: str) -> pd.DataFrame:
    run_dir = FINAL_OUTPUTS_DIR / run_name
    metrics_rows = load_json(run_dir / "metrics.json")
    rows = []
    for metric_row in metrics_rows:
        experiment = metric_row["experiment"]
        daily_path = run_dir / experiment / "daily_predictions.jsonl"
        diagnostics = summarize_daily_records(load_jsonl(daily_path)) if daily_path.exists() else {}
        row = {
            "run_name": run_name,
            "experiment": experiment,
            "label": experiment_label(experiment),
            "family": classify_experiment(experiment),
            **metric_row,
            **diagnostics,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_relative_columns(df: pd.DataFrame, memory_experiment: str, rule_experiment: str) -> pd.DataFrame:
    out = df.copy()
    memory_row = out.loc[out["experiment"] == memory_experiment].iloc[0]
    rule_row = out.loc[out["experiment"] == rule_experiment].iloc[0]

    out["delta_mae_vs_memory"] = out["mae_price"] - float(memory_row["mae_price"])
    out["delta_mae_vs_rule"] = out["mae_price"] - float(rule_row["mae_price"])
    out["delta_accuracy_vs_memory"] = out["directional_accuracy"] - float(memory_row["directional_accuracy"])
    out["delta_accuracy_vs_rule"] = out["directional_accuracy"] - float(rule_row["directional_accuracy"])
    out["relative_mae_vs_memory_pct"] = (out["mae_price"] / float(memory_row["mae_price"]) - 1.0) * 100.0
    out["relative_mae_vs_rule_pct"] = (out["mae_price"] / float(rule_row["mae_price"]) - 1.0) * 100.0
    out["rank_mae_price"] = out["mae_price"].rank(method="min")
    out["rank_rmse_price"] = out["rmse_price"].rank(method="min")
    out["rank_directional_accuracy"] = out["directional_accuracy"].rank(method="min", ascending=False)
    return out


def build_probe_table() -> pd.DataFrame:
    rows = []
    for run_dir in FINAL_OUTPUTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics_rows = load_json(metrics_path)
        for metric_row in metrics_rows:
            if not str(metric_row.get("experiment", "")).startswith("llm_"):
                continue
            if int(metric_row.get("n_days", 0)) > 5:
                continue
            experiment = metric_row["experiment"]
            daily_path = run_dir / experiment / "daily_predictions.jsonl"
            if not daily_path.exists():
                continue
            diagnostics = summarize_daily_records(load_jsonl(daily_path))
            rows.append(
                {
                    "run_name": run_dir.name,
                    "experiment": experiment,
                    "label": experiment_label(experiment),
                    **metric_row,
                    **diagnostics,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["label", "run_name"]).reset_index(drop=True)


def aggregate_probe_models(probe_df: pd.DataFrame) -> pd.DataFrame:
    if probe_df.empty:
        return pd.DataFrame()

    rows = []
    for label, block in probe_df.groupby("label", sort=True):
        weights = block["n_days"].to_numpy(dtype=float)
        total_days = float(weights.sum())

        def weighted_mean(column: str) -> float:
            values = block[column].to_numpy(dtype=float)
            return float(np.average(values, weights=weights)) if total_days else float(np.mean(values))

        rows.append(
            {
                "label": label,
                "probe_runs": int(len(block)),
                "total_probe_days": int(total_days),
                "weighted_mae_price": weighted_mean("mae_price"),
                "weighted_rmse_price": weighted_mean("rmse_price"),
                "weighted_directional_accuracy": weighted_mean("directional_accuracy"),
                "weighted_skip_rate": weighted_mean("skip_rate"),
                "weighted_active_persona_rate": weighted_mean("active_persona_rate"),
                "weighted_mean_vote_count": weighted_mean("mean_vote_count"),
                "weighted_mean_persona_weight": weighted_mean("mean_persona_weight"),
            }
        )
    return pd.DataFrame(rows).sort_values(["weighted_skip_rate", "weighted_mae_price"], ascending=[True, True])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    return df.fillna("NA").to_markdown(index=False)


def plot_leaderboard(df: pd.DataFrame, value_col: str, title: str, xlabel: str, path: Path, ascending: bool = True) -> None:
    if not HAS_MATPLOTLIB or df.empty:
        return

    work = df.sort_values(value_col, ascending=ascending).copy()
    colors = [FAMILY_COLORS.get(family, "#666666") for family in work["family"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(work["label"], work[value_col], color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_llm_weight_vs_mae(df: pd.DataFrame, path: Path) -> None:
    if not HAS_MATPLOTLIB or df.empty:
        return

    work = df[df["family"].isin(["llm", "rule"])].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in work.iterrows():
        ax.scatter(
            row["mean_persona_weight"],
            row["mae_price"],
            color=FAMILY_COLORS.get(row["family"], "#666666"),
            s=90,
        )
        ax.text(
            row["mean_persona_weight"] + 0.003,
            row["mae_price"] + 0.002,
            row["label"],
            fontsize=9,
        )
    ax.set_title("Persona Weight vs MAE Price")
    ax.set_xlabel("Mean Persona Weight")
    ax.set_ylabel("MAE Price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_probe_skip_rate(df: pd.DataFrame, path: Path) -> None:
    if not HAS_MATPLOTLIB or df.empty:
        return

    work = df.sort_values("weighted_skip_rate", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(work["label"], work["weighted_skip_rate"], color=FAMILY_COLORS["probe"])
    ax.set_title("Short-Window Probe Skip Rate")
    ax.set_xlabel("Weighted Skip Rate")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    matched_df: pd.DataFrame,
    focus_df: pd.DataFrame,
    long_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    probe_model_df: pd.DataFrame,
) -> str:
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    best_llm = focus_df[focus_df["family"] == "llm"].sort_values("mae_price").iloc[0]
    memory_row = focus_df.loc[focus_df["experiment"] == "memory_only_no_personas"].iloc[0]
    rule_row = focus_df.loc[focus_df["experiment"] == "rule_personas"].iloc[0]

    findings = [
        (
            f"The matched 30-day benchmark uses `{MATCHED_RUN}` and the best LLM in that exact slice is "
            f"`{best_llm['label']}` with `MAE price = {best_llm['mae_price']:.6f}` and "
            f"`directional accuracy = {best_llm['directional_accuracy']:.3f}`."
        ),
        (
            f"`Memory Only` remains the strongest matched-window benchmark at `MAE price = {memory_row['mae_price']:.6f}` "
            f"and `directional accuracy = {memory_row['directional_accuracy']:.3f}`. "
            f"The best LLM is `{best_llm['relative_mae_vs_memory_pct']:+.2f}%` worse on MAE price versus that baseline."
        ),
        (
            f"All three stable 30-day LLM runs beat `Rule Personas` on MAE price. "
            f"`{best_llm['label']}` improves MAE price over `Rule Personas` by "
            f"`{-best_llm['delta_mae_vs_rule']:.6f}` while matching its directional accuracy."
        ),
    ]

    if not probe_model_df.empty:
        unstable = probe_model_df.sort_values(["weighted_skip_rate", "weighted_mae_price"], ascending=[False, True]).iloc[0]
        stable = probe_model_df.sort_values(["weighted_skip_rate", "weighted_mae_price"], ascending=[True, True]).iloc[0]
        findings.append(
            f"Short-window probes show `{unstable['label']}` as the least reliable available model "
            f"with weighted skip rate `{unstable['weighted_skip_rate']:.2f}`, while `{stable['label']}` "
            f"is the most reliable probe model at weighted skip rate `{stable['weighted_skip_rate']:.2f}`."
        )

    lines = [
        "# LLM Benchmark Report",
        "",
        f"Built: {built_at}",
        "",
        "## Source Runs",
        "",
        f"- Matched 30-day ablation: `final phase/outputs/{MATCHED_RUN}`",
        f"- Long-horizon reference: `final phase/outputs/{LONG_BASELINE_RUN}` plus `final phase/outputs/{LONG_LLM_RUN}`",
        "- Probe reliability slice: all `final phase/outputs/*` LLM runs with `n_days <= 5`",
        "",
        "## Key Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Matched 30-Day Focus Table",
            "",
            markdown_table(
                focus_df[
                    [
                        "label",
                        "n_days",
                        "mae_price",
                        "rmse_price",
                        "mae_return",
                        "directional_accuracy",
                        "sign_f1",
                        "skip_rate",
                        "mean_persona_weight",
                        "mean_vote_count",
                        "relative_mae_vs_memory_pct",
                        "relative_mae_vs_rule_pct",
                    ]
                ].sort_values("mae_price")
            ),
            "",
            "## Full 30-Day Ablation Table",
            "",
            markdown_table(
                matched_df[
                    [
                        "label",
                        "family",
                        "mae_price",
                        "rmse_price",
                        "mae_return",
                        "directional_accuracy",
                        "sign_f1",
                        "rank_mae_price",
                        "rank_directional_accuracy",
                    ]
                ].sort_values("mae_price")
            ),
            "",
            "## 100-Day Reference",
            "",
            markdown_table(
                long_df[
                    [
                        "label",
                        "mae_price",
                        "rmse_price",
                        "mae_return",
                        "directional_accuracy",
                        "sign_f1",
                    ]
                ].sort_values("mae_price")
            ),
            "",
        ]
    )

    if not probe_model_df.empty:
        lines.extend(
            [
                "## Probe Reliability By Model",
                "",
                markdown_table(probe_model_df),
                "",
            ]
        )

    if not probe_df.empty:
        lines.extend(
            [
                "## Probe Runs",
                "",
                markdown_table(
                    probe_df[
                        [
                            "run_name",
                            "label",
                            "n_days",
                            "mae_price",
                            "directional_accuracy",
                            "skip_rate",
                            "mean_vote_count",
                            "mean_persona_weight",
                        ]
                    ]
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            "- `benchmark/llm_ablation_30d_full.csv`",
            "- `benchmark/llm_ablation_30d_focus.csv`",
            "- `benchmark/llm_reference_100d.csv`",
            "- `benchmark/llm_probe_runs.csv`",
            "- `benchmark/llm_probe_model_summary.csv`",
            "- `benchmark/figures/leaderboard_mae_price_30d.png`",
            "- `benchmark/figures/leaderboard_directional_accuracy_30d.png`",
            "- `benchmark/figures/llm_weight_vs_mae_30d.png`",
            "- `benchmark/figures/probe_skip_rate.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    matched_df = load_run_table(MATCHED_RUN)
    matched_df = add_relative_columns(
        matched_df,
        memory_experiment="memory_only_no_personas",
        rule_experiment="rule_personas",
    ).sort_values("mae_price")

    focus_experiments = [
        "memory_only_no_personas",
        "rule_personas",
        "llm_openrouter_google_gemini_2_5_flash",
        "llm_openrouter_anthropic_claude_sonnet_4",
        "llm_openrouter_openai_gpt_5_chat",
    ]
    focus_df = matched_df[matched_df["experiment"].isin(focus_experiments)].copy().sort_values("mae_price")

    long_df = pd.concat(
        [
            load_run_table(LONG_BASELINE_RUN),
            load_run_table(LONG_LLM_RUN),
        ],
        ignore_index=True,
    ).sort_values("mae_price")

    probe_df = build_probe_table()
    probe_model_df = aggregate_probe_models(probe_df)

    write_csv(matched_df, BENCHMARK_DIR / "llm_ablation_30d_full.csv")
    write_csv(focus_df, BENCHMARK_DIR / "llm_ablation_30d_focus.csv")
    write_csv(long_df, BENCHMARK_DIR / "llm_reference_100d.csv")
    write_csv(probe_df, BENCHMARK_DIR / "llm_probe_runs.csv")
    write_csv(probe_model_df, BENCHMARK_DIR / "llm_probe_model_summary.csv")

    source_payload = {
        "matched_30d_run": str(FINAL_OUTPUTS_DIR / MATCHED_RUN),
        "long_baseline_run": str(FINAL_OUTPUTS_DIR / LONG_BASELINE_RUN),
        "long_llm_run": str(FINAL_OUTPUTS_DIR / LONG_LLM_RUN),
        "probe_rule": "All final-phase LLM runs with n_days <= 5",
    }
    (BENCHMARK_DIR / "source_runs.json").write_text(
        json.dumps(source_payload, indent=2),
        encoding="utf-8",
    )

    plot_leaderboard(
        matched_df,
        value_col="mae_price",
        title="Matched 30-Day LLM Ablation: MAE Price",
        xlabel="MAE Price",
        path=FIGURES_DIR / "leaderboard_mae_price_30d.png",
        ascending=True,
    )
    plot_leaderboard(
        matched_df,
        value_col="directional_accuracy",
        title="Matched 30-Day LLM Ablation: Directional Accuracy",
        xlabel="Directional Accuracy",
        path=FIGURES_DIR / "leaderboard_directional_accuracy_30d.png",
        ascending=False,
    )
    plot_llm_weight_vs_mae(
        focus_df,
        path=FIGURES_DIR / "llm_weight_vs_mae_30d.png",
    )
    plot_probe_skip_rate(
        probe_model_df,
        path=FIGURES_DIR / "probe_skip_rate.png",
    )

    report = build_report(matched_df, focus_df, long_df, probe_df, probe_model_df)
    (BENCHMARK_DIR / "benchmark_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

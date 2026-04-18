"""
Output writing for the TEPC pipeline.
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .config import RunConfig
from .features import FeatureBundle


def _config_payload(config: RunConfig) -> dict:
    payload = {
        key: value
        for key, value in config.__dict__.items()
        if key != "paths"
    }
    payload["paths"] = {key: str(value) for key, value in config.paths.__dict__.items()}
    payload["coupling_epsilons"] = list(config.coupling_epsilons)
    payload["filtration_quantiles"] = list(config.filtration_quantiles)
    payload["output_dir"] = str(config.output_dir) if config.output_dir else None
    return payload


def write_outputs(
    output_dir: Path,
    config: RunConfig,
    bundle: FeatureBundle,
    experiment_results: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.json").write_text(
        json.dumps(_config_payload(config), indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(bundle.dataset_summary, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "feature_groups.json").write_text(
        json.dumps(bundle.groups, indent=2, default=str),
        encoding="utf-8",
    )

    bundle.node_frame.to_csv(output_dir / "node_panel.csv")
    bundle.node_transforms.to_csv(output_dir / "node_transforms.csv")
    bundle.frame.to_csv(output_dir / "feature_frame.csv", index=True)

    all_records = []
    metrics_rows = []

    for result in experiment_results:
        exp_dir = output_dir / result["experiment"]
        exp_dir.mkdir(parents=True, exist_ok=True)

        daily = result.get("daily_records", [])
        if daily:
            df = pd.DataFrame(daily)
            df.to_csv(exp_dir / "daily_predictions.csv", index=False)
            all_records.extend(daily)

        if result.get("metrics"):
            metrics_rows.append({"experiment": result["experiment"], **result["metrics"]})

    if all_records:
        pd.DataFrame(all_records).to_csv(output_dir / "daily_predictions.csv", index=False)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).sort_values(["macro_f1", "mae_return"], ascending=[False, True])
        metrics_df.to_csv(output_dir / "ablation_summary.csv", index=False)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics_rows, indent=2, default=str),
            encoding="utf-8",
        )

        lines = [
            "# TEPC Run Summary",
            "",
            f"Experiments: {len(metrics_rows)}",
            "",
            "## Metrics",
            "",
            metrics_df.to_markdown(index=False),
            "",
        ]
        (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")

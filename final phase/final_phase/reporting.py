"""
Output writing for the final-phase system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import pandas as pd


def write_outputs(
    output_dir: Path,
    config_payload: Dict,
    dataset_summary: Dict,
    experiment_results: List[Dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, default=str)

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, default=str)

    all_daily = []
    ablation_rows = []
    persona_memory = {}

    for result in experiment_results:
        name = result["experiment"]
        daily = result.get("daily_records", [])
        if daily:
            exp_dir = output_dir / name
            exp_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(daily).to_csv(exp_dir / "daily_predictions.csv", index=False)
            with (exp_dir / "daily_predictions.jsonl").open("w", encoding="utf-8") as f:
                for row in daily:
                    f.write(json.dumps(row, default=str) + "\n")
            all_daily.extend(daily)

        metrics = result.get("metrics", {})
        if metrics:
            ablation_rows.append({"experiment": name, **metrics})
        persona_memory[name] = result.get("persona_memory", {})

    if all_daily:
        pd.DataFrame(all_daily).to_csv(output_dir / "daily_predictions.csv", index=False)

    if ablation_rows:
        ablation_df = pd.DataFrame(ablation_rows).sort_values("mae_price")
        ablation_df.to_csv(output_dir / "ablation_summary.csv", index=False)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(ablation_rows, f, indent=2, default=str)

        lines = [
            "# Run Summary",
            "",
            f"Experiments: {len(ablation_rows)}",
            "",
            "## Metrics",
            "",
            ablation_df.to_markdown(index=False),
            "",
        ]
        (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")

    with (output_dir / "persona_memory.json").open("w", encoding="utf-8") as f:
        json.dump(persona_memory, f, indent=2, default=str)

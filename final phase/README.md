# Final Phase

This folder contains a new market-memory-first prediction system for USD/INR.

The design goal is to combine:

- financial and macro signals
- statistical models
- machine learning models
- multi-persona market simulation
- optional LLM-based persona engines
- explicit ablation and detailed result logging

It is intentionally separate from the earlier research code so it can evolve without
breaking `latest/`, `gemini_preds/`, `GARCH/`, or the phase-based experiments.

## What This System Does

The pipeline:

1. loads and integrates the repo's core datasets
2. builds a feature frame from macro, GDELT, thematic, Goldstein, and political-news inputs
3. creates a shared market memory snapshot for each decision date
4. fits a hybrid statistical + ML model suite
5. runs a rule-based or LLM-based multi-persona market simulation
6. blends the numeric and persona views into a final prediction
7. performs walk-forward backtests and ablations
8. records every daily result, ensemble component, persona vote, and summary metric

## Bias Controls

This module explicitly avoids using the immediate previous close as a direct feature.

It does this in two ways:

- the model predicts future returns, not a copied lagged price
- the feature set excludes raw `INR` as a direct input
- an information embargo is applied by default, so the target is shifted beyond the most recent observation

Default setup:

- forecast horizon: 1 trading day
- information embargo: 1 trading day
- effective target gap: 2 trading days
- default evaluation window: 100 trading days

This makes the task harder, but it is more honest and reduces anchoring/persistence bias.

## Output Structure

Each run creates a timestamped folder under `outputs/` containing:

- `config.json`
- `dataset_summary.json`
- `daily_predictions.csv`
- `daily_predictions.jsonl`
- `metrics.json`
- `ablation_summary.csv`
- `persona_memory.json`
- `run_summary.md`

## Running

From the repository root:

```powershell
python "final phase\\run.py" --help
python "final phase\\run.py"
python "final phase\\run.py" --test-days 100
python "final phase\\run.py" --test-days 100 --rich-only
python "final phase\\run.py" --test-days 120 --llm-model openrouter:your-model-id
```

Oil is already part of the base macro layer through `Super_Master_Dataset.csv`, so the default stack is already using cross-asset oil information.

## LLM Providers

Supported provider spec formats:

- `rule`
- `openrouter:<model-id>`
- `ollama:<model-id>`

`rule` requires no API key and is the safest baseline.

Important:

- no premium LLM is used by default
- Claude Opus style specs are blocked by default in the runner to avoid accidental spend
- if you really want to use one, you must pass `--allow-premium-llm-models`

`openrouter:<model-id>` requires:

- `OPENROUTER_API_KEY`

`ollama:<model-id>` requires:

- a local Ollama server
- the selected local model to be pulled already

## Recommended Workflow

1. run the default rule-based ablation first
2. inspect the saved outputs
3. add LLM models through `--llm-model ...`
4. compare the `ablation_summary.csv`
5. only then increase persona count or test window if needed

## LLM Matrix Launcher

If you want one command that runs a small non-Opus LLM comparison matrix on top of the default baselines:

```powershell
$env:OPENROUTER_API_KEY = "your-key"
powershell -ExecutionPolicy Bypass -File "final phase\\run_llm_matrix.ps1"
```

The launcher defaults to:

- `google/gemini-2.5-flash`
- `anthropic/claude-sonnet-4`
- `openai/gpt-5-chat`

Useful variants:

```powershell
powershell -ExecutionPolicy Bypass -File "final phase\\run_llm_matrix.ps1" -TestDays 100 -PersonaLimit 3
powershell -ExecutionPolicy Bypass -File "final phase\\run_llm_matrix.ps1" -TestDays 30 -PersonaLimit 3 -LLMOnly
powershell -ExecutionPolicy Bypass -File "final phase\\run_llm_matrix.ps1" -RichOnly
```

## Plotting Results

To compare multiple saved runs with graphs:

```powershell
python "final phase\\plot_results.py" `
  --run-dir "final phase\\outputs\\llm_matrix_v2_30d" `
  --run-dir "final phase\\outputs\\backtest_100d" `
  --run-dir "final phase\\outputs\\llm_gemini_100d" `
  --output-dir "final phase\\plots\\comparison_latest"
```

This generates:

- an all-runs leaderboard
- a MAE-vs-direction-accuracy scatter plot
- per-run dashboards
- per-horizon rank heatmaps
- actual-vs-predicted overlays for the top experiments
- rolling absolute error comparisons

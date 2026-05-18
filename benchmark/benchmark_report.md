# LLM Benchmark Report

Built: 2026-04-19 09:37:52 UTC

## Source Runs

- Matched 30-day ablation: `final phase/outputs/llm_matrix_v2_30d`
- Long-horizon reference: `final phase/outputs/backtest_100d` plus `final phase/outputs/llm_gemini_100d`
- Probe reliability slice: all `final phase/outputs/*` LLM runs with `n_days <= 5`

## Key Findings

- The matched 30-day benchmark uses `llm_matrix_v2_30d` and the best LLM in that exact slice is `Gemini 2.5 Flash` with `MAE price = 0.443296` and `directional accuracy = 0.533`.
- `Memory Only` remains the strongest matched-window benchmark at `MAE price = 0.329545` and `directional accuracy = 0.567`. The best LLM is `+34.52%` worse on MAE price versus that baseline.
- All three stable 30-day LLM runs beat `Rule Personas` on MAE price. `Gemini 2.5 Flash` improves MAE price over `Rule Personas` by `0.047384` while matching its directional accuracy.
- Short-window probes show `GPT-5 Mini` as the least reliable available model with weighted skip rate `1.00`, while `GPT-5 Chat` is the most reliable probe model at weighted skip rate `0.00`.

## Matched 30-Day Focus Table

| label            |   n_days |   mae_price |   rmse_price |   mae_return |   directional_accuracy |   sign_f1 | skip_rate   |   mean_persona_weight | mean_vote_count   |   relative_mae_vs_memory_pct |   relative_mae_vs_rule_pct |
|:-----------------|---------:|------------:|-------------:|-------------:|-----------------------:|----------:|:------------|----------------------:|:------------------|-----------------------------:|---------------------------:|
| Memory Only      |       30 |    0.329545 |     0.396617 |   0.00365848 |               0.566667 |  0.723404 | NA          |             0         | NA                |                       0      |                  -32.8391  |
| Gemini 2.5 Flash |       30 |    0.443296 |     0.569627 |   0.00493595 |               0.533333 |  0.695652 | 0.0         |             0.265113  | 3.0               |                      34.5176 |                   -9.65673 |
| Claude Sonnet 4  |       30 |    0.44386  |     0.569759 |   0.00494229 |               0.533333 |  0.695652 | 0.0         |             0.251967  | 3.0               |                      34.6889 |                   -9.54171 |
| GPT-5 Chat       |       30 |    0.450781 |     0.581571 |   0.00501925 |               0.533333 |  0.695652 | 0.0         |             0.225729  | 3.0               |                      36.789  |                   -8.13123 |
| Rule Personas    |       30 |    0.490679 |     0.635518 |   0.00546341 |               0.533333 |  0.695652 | 0.0         |             0.0664226 | 3.0               |                      48.8961 |                    0       |

## Full 30-Day Ablation Table

| label            | family   |   mae_price |   rmse_price |   mae_return |   directional_accuracy |   sign_f1 |   rank_mae_price |   rank_directional_accuracy |
|:-----------------|:---------|------------:|-------------:|-------------:|-----------------------:|----------:|-----------------:|----------------------------:|
| Stat+ML No GDELT | stat_ml  |    0.31875  |     0.396558 |   0.00353971 |               0.5      |  0.666667 |                1 |                           7 |
| Memory Only      | memory   |    0.329545 |     0.396617 |   0.00365848 |               0.566667 |  0.723404 |                2 |                           1 |
| Gemini 2.5 Flash | llm      |    0.443296 |     0.569627 |   0.00493595 |               0.533333 |  0.695652 |                3 |                           2 |
| Claude Sonnet 4  | llm      |    0.44386  |     0.569759 |   0.00494229 |               0.533333 |  0.695652 |                4 |                           2 |
| GPT-5 Chat       | llm      |    0.450781 |     0.581571 |   0.00501925 |               0.533333 |  0.695652 |                5 |                           2 |
| Rule Personas    | rule     |    0.490679 |     0.635518 |   0.00546341 |               0.533333 |  0.695652 |                6 |                           2 |
| Stat+ML Full     | stat_ml  |    0.505987 |     0.657839 |   0.00563392 |               0.533333 |  0.695652 |                7 |                           2 |
| Stat+ML No Macro | stat_ml  |    0.641909 |     0.891364 |   0.00714186 |               0.5      |  0.666667 |                8 |                           7 |

## 100-Day Reference

| label            |   mae_price |   rmse_price |   mae_return |   directional_accuracy |   sign_f1 |
|:-----------------|------------:|-------------:|-------------:|-----------------------:|----------:|
| Memory Only      |    0.242095 |     0.328989 |   0.002718   |                   0.61 |  0.757764 |
| Stat+ML No GDELT |    0.247282 |     0.335915 |   0.00277885 |                   0.49 |  0.657718 |
| Gemini 2.5 Flash |    0.272617 |     0.389863 |   0.00306483 |                   0.58 |  0.734177 |
| Rule Personas    |    0.290083 |     0.418229 |   0.00326036 |                   0.58 |  0.734177 |
| Stat+ML Full     |    0.300455 |     0.436129 |   0.00337658 |                   0.58 |  0.734177 |
| Stat+ML No Macro |    0.352429 |     0.550743 |   0.00395451 |                   0.58 |  0.734177 |

## Probe Reliability By Model

| label            |   probe_runs |   total_probe_days |   weighted_mae_price |   weighted_rmse_price |   weighted_directional_accuracy |   weighted_skip_rate |   weighted_active_persona_rate |   weighted_mean_vote_count |   weighted_mean_persona_weight |
|:-----------------|-------------:|-------------------:|---------------------:|----------------------:|--------------------------------:|---------------------:|-------------------------------:|---------------------------:|-------------------------------:|
| GPT-5 Chat       |            1 |                  2 |             0.109429 |              0.144126 |                        1        |                    0 |                              1 |                          2 |                       0.263047 |
| Claude Sonnet 4  |            2 |                  4 |             0.111287 |              0.134851 |                        1        |                    0 |                              1 |                          2 |                       0.21712  |
| Gemini 2.5 Flash |            1 |                  2 |             0.119059 |              0.146232 |                        1        |                    0 |                              1 |                          2 |                       0.2646   |
| GPT-5 Mini       |            1 |                  2 |             0.117123 |              0.118722 |                        1        |                    1 |                              0 |                          0 |                       0        |
| Gemma 4 26B Free |            2 |                  7 |             0.470036 |              0.579776 |                        0.571429 |                    1 |                              0 |                          0 |                       0        |

## Probe Runs

| run_name                 | label            |   n_days |   mae_price |   directional_accuracy |   skip_rate |   mean_vote_count |   mean_persona_weight |
|:-------------------------|:-----------------|---------:|------------:|-----------------------:|------------:|------------------:|----------------------:|
| llm_matrix_probe_v2_2d   | Claude Sonnet 4  |        2 |    0.110018 |                    1   |           0 |                 2 |              0.26432  |
| probe_claude_sonnet_4_2d | Claude Sonnet 4  |        2 |    0.112556 |                    1   |           0 |                 2 |              0.16992  |
| llm_matrix_probe_v2_2d   | GPT-5 Chat       |        2 |    0.109429 |                    1   |           0 |                 2 |              0.263047 |
| probe_gpt_5_mini_2d      | GPT-5 Mini       |        2 |    0.117123 |                    1   |           1 |                 0 |              0        |
| llm_matrix_probe_v2_2d   | Gemini 2.5 Flash |        2 |    0.119059 |                    1   |           0 |                 2 |              0.2646   |
| llm_sanity_5d            | Gemma 4 26B Free |        5 |    0.611202 |                    0.4 |           1 |                 0 |              0        |
| llm_smoke_2d             | Gemma 4 26B Free |        2 |    0.117123 |                    1   |           1 |                 0 |              0        |

## Output Files

- `benchmark/llm_ablation_30d_full.csv`
- `benchmark/llm_ablation_30d_focus.csv`
- `benchmark/llm_reference_100d.csv`
- `benchmark/llm_probe_runs.csv`
- `benchmark/llm_probe_model_summary.csv`
- `benchmark/figures/leaderboard_mae_price_30d.png`
- `benchmark/figures/leaderboard_directional_accuracy_30d.png`
- `benchmark/figures/llm_weight_vs_mae_30d.png`
- `benchmark/figures/probe_skip_rate.png`

# TEPC Cross-Run Plot Summary

Runs compared: 2

| experiment     |   n_days |   breakout_accuracy |   macro_f1 |   mae_return |   rmse_return |   mae_volatility |   bias_return | run_name            | run_experiment                        |
|:---------------|---------:|--------------------:|-----------:|-------------:|--------------:|-----------------:|--------------:|:--------------------|:--------------------------------------|
| chaos_only     |      100 |              0.8    |   0.40398  |   0.00345361 |    0.00483634 |      0.00131974  |  -0.000696786 | livepull_100d       | livepull_100d :: chaos_only           |
| topology_chaos |      100 |              0.71   |   0.371553 |   0.00344817 |    0.00482077 |      0.00122347  |  -0.000332338 | livepull_100d       | livepull_100d :: topology_chaos       |
| topology_only  |      100 |              0.68   |   0.371438 |   0.00349007 |    0.00492029 |      0.00124144  |   1.06114e-05 | livepull_100d       | livepull_100d :: topology_only        |
| tepc_full      |      100 |              0.75   |   0.35645  |   0.00343033 |    0.00485305 |      0.00124071  |  -0.00041619  | livepull_100d       | livepull_100d :: tepc_full            |
| tepc_full      |       80 |              0.775  |   0.33701  |   0.0027458  |    0.00376496 |      0.000791117 |  -1.7559e-05  | strict_raw_2025_80d | strict_raw_2025_80d :: tepc_full      |
| topology_chaos |       80 |              0.6875 |   0.30303  |   0.00299074 |    0.0041362  |      0.000732149 |  -0.00032895  | strict_raw_2025_80d | strict_raw_2025_80d :: topology_chaos |
| macro_baseline |       80 |              0.7625 |   0.288416 |   0.00278195 |    0.00375498 |      0.000973979 |  -0.000193468 | strict_raw_2025_80d | strict_raw_2025_80d :: macro_baseline |
| chaos_only     |       80 |              0.7375 |   0.282974 |   0.00307979 |    0.00408757 |      0.000873243 |  -0.000364824 | strict_raw_2025_80d | strict_raw_2025_80d :: chaos_only     |
| macro_baseline |      100 |              0.72   |   0.282353 |   0.00362654 |    0.00518738 |      0.00134762  |  -0.000697026 | livepull_100d       | livepull_100d :: macro_baseline       |
| topology_only  |       80 |              0.6875 |   0.275689 |   0.00335159 |    0.00442237 |      0.000729716 |  -5.69276e-05 | strict_raw_2025_80d | strict_raw_2025_80d :: topology_only  |

# TEPC Cross-Run Plot Summary

Runs compared: 2

| experiment         |   n_days |   breakout_accuracy |   macro_f1 |   mae_return |   rmse_return |   mae_volatility |   bias_return | run_name                | run_experiment                                |
|:-------------------|---------:|--------------------:|-----------:|-------------:|--------------:|-----------------:|--------------:|:------------------------|:----------------------------------------------|
| tepc_persona_blend |       80 |              0.775  |   0.33701  |   0.00270421 |    0.00374137 |      0.000792922 |  -8.17389e-05 | strict_persona_2025_80d | strict_persona_2025_80d :: tepc_persona_blend |
| tepc_full          |       80 |              0.775  |   0.33701  |   0.0027458  |    0.00376496 |      0.000791117 |  -1.7559e-05  | strict_raw_2025_80d     | strict_raw_2025_80d :: tepc_full              |
| tepc_full          |       80 |              0.775  |   0.33701  |   0.0027458  |    0.00376496 |      0.000791117 |  -1.7559e-05  | strict_persona_2025_80d | strict_persona_2025_80d :: tepc_full          |
| topology_chaos     |       80 |              0.6875 |   0.30303  |   0.00299074 |    0.0041362  |      0.000732149 |  -0.00032895  | strict_raw_2025_80d     | strict_raw_2025_80d :: topology_chaos         |
| macro_baseline     |       80 |              0.7625 |   0.288416 |   0.00278195 |    0.00375498 |      0.000973979 |  -0.000193468 | strict_raw_2025_80d     | strict_raw_2025_80d :: macro_baseline         |
| chaos_only         |       80 |              0.7375 |   0.282974 |   0.00307979 |    0.00408757 |      0.000873243 |  -0.000364824 | strict_raw_2025_80d     | strict_raw_2025_80d :: chaos_only             |
| topology_only      |       80 |              0.6875 |   0.275689 |   0.00335159 |    0.00442237 |      0.000729716 |  -5.69276e-05 | strict_raw_2025_80d     | strict_raw_2025_80d :: topology_only          |
| persona_rule_only  |       80 |              0.2375 |   0.164496 |   0.00287933 |    0.00393951 |      0.00188357  |  -0.000365822 | strict_persona_2025_80d | strict_persona_2025_80d :: persona_rule_only  |

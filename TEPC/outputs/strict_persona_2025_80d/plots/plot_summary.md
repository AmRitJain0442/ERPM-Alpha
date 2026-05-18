# TEPC Plot Summary

Best experiment: `tepc_persona_blend`

| experiment         |   n_days |   breakout_accuracy |   macro_f1 |   mae_return |   rmse_return |   mae_volatility |   bias_return |
|:-------------------|---------:|--------------------:|-----------:|-------------:|--------------:|-----------------:|--------------:|
| tepc_persona_blend |       80 |              0.775  |   0.33701  |   0.00270421 |    0.00374137 |      0.000792922 |  -8.17389e-05 |
| tepc_full          |       80 |              0.775  |   0.33701  |   0.0027458  |    0.00376496 |      0.000791117 |  -1.7559e-05  |
| persona_rule_only  |       80 |              0.2375 |   0.164496 |   0.00287933 |    0.00393951 |      0.00188357  |  -0.000365822 |

```json
{
  "best_experiment": "tepc_persona_blend",
  "top_experiments": [
    "tepc_persona_blend",
    "tepc_full",
    "persona_rule_only"
  ],
  "metric_table": [
    {
      "experiment": "tepc_persona_blend",
      "n_days": 80,
      "breakout_accuracy": 0.775,
      "macro_f1": 0.3370103916866507,
      "mae_return": 0.0027042092489268,
      "rmse_return": 0.0037413741371078,
      "mae_volatility": 0.0007929219099462,
      "bias_return": -8.173889027428918e-05
    },
    {
      "experiment": "tepc_full",
      "n_days": 80,
      "breakout_accuracy": 0.775,
      "macro_f1": 0.3370103916866507,
      "mae_return": 0.0027458010725392,
      "rmse_return": 0.0037649613085827,
      "mae_volatility": 0.0007911170764962,
      "bias_return": -1.7558953559874013e-05
    },
    {
      "experiment": "persona_rule_only",
      "n_days": 80,
      "breakout_accuracy": 0.2375,
      "macro_f1": 0.1644963075708889,
      "mae_return": 0.0028793304539795,
      "rmse_return": 0.0039395083637214,
      "mae_volatility": 0.0018835666114431,
      "bias_return": -0.0003658219923007
    }
  ]
}
```

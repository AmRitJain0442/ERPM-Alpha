# TEPC Plot Summary

Best experiment: `tepc_full`

| experiment         |   n_days |   breakout_accuracy |   macro_f1 |   mae_return |   rmse_return |   mae_volatility |   bias_return |
|:-------------------|---------:|--------------------:|-----------:|-------------:|--------------:|-----------------:|--------------:|
| tepc_full          |       80 |              0.775  |   0.33701  |   0.0027458  |    0.00376496 |      0.000791117 |  -1.7559e-05  |
| persona_rule_only  |       80 |              0.8    |   0.296296 |   0.00288643 |    0.00394595 |      0.00203958  |  -0.000522834 |
| tepc_persona_blend |       80 |              0.7875 |   0.293706 |   0.0026946  |    0.00374276 |      0.000894673 |  -0.000187692 |

```json
{
  "best_experiment": "tepc_full",
  "top_experiments": [
    "tepc_full",
    "persona_rule_only",
    "tepc_persona_blend"
  ],
  "metric_table": [
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
      "breakout_accuracy": 0.8,
      "macro_f1": 0.2962962962962963,
      "mae_return": 0.0028864283356525,
      "rmse_return": 0.0039459451077767,
      "mae_volatility": 0.0020395784026567,
      "bias_return": -0.0005228337835143
    },
    {
      "experiment": "tepc_persona_blend",
      "n_days": 80,
      "breakout_accuracy": 0.7875,
      "macro_f1": 0.2937062937062937,
      "mae_return": 0.0026945951997161,
      "rmse_return": 0.0037427594111205,
      "mae_volatility": 0.0008946734172108,
      "bias_return": -0.0001876918050758
    }
  ]
}
```

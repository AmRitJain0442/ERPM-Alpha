# TEPC Plot Summary

Best experiment: `tepc_full`

| experiment     |   n_days |   breakout_accuracy |   macro_f1 |   mae_return |   rmse_return |   mae_volatility |   bias_return |
|:---------------|---------:|--------------------:|-----------:|-------------:|--------------:|-----------------:|--------------:|
| tepc_full      |       80 |              0.775  |   0.33701  |   0.0027458  |    0.00376496 |      0.000791117 |  -1.7559e-05  |
| topology_chaos |       80 |              0.6875 |   0.30303  |   0.00299074 |    0.0041362  |      0.000732149 |  -0.00032895  |
| macro_baseline |       80 |              0.7625 |   0.288416 |   0.00278195 |    0.00375498 |      0.000973979 |  -0.000193468 |
| chaos_only     |       80 |              0.7375 |   0.282974 |   0.00307979 |    0.00408757 |      0.000873243 |  -0.000364824 |
| topology_only  |       80 |              0.6875 |   0.275689 |   0.00335159 |    0.00442237 |      0.000729716 |  -5.69276e-05 |

```json
{
  "best_experiment": "tepc_full",
  "top_experiments": [
    "tepc_full",
    "topology_chaos",
    "macro_baseline"
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
      "experiment": "topology_chaos",
      "n_days": 80,
      "breakout_accuracy": 0.6875,
      "macro_f1": 0.303030303030303,
      "mae_return": 0.0029907380026549,
      "rmse_return": 0.0041361993614348,
      "mae_volatility": 0.0007321491528426,
      "bias_return": -0.0003289495644869
    },
    {
      "experiment": "macro_baseline",
      "n_days": 80,
      "breakout_accuracy": 0.7625,
      "macro_f1": 0.2884160756501182,
      "mae_return": 0.0027819465788436,
      "rmse_return": 0.003754979439603,
      "mae_volatility": 0.0009739789544345,
      "bias_return": -0.000193467531554
    },
    {
      "experiment": "chaos_only",
      "n_days": 80,
      "breakout_accuracy": 0.7375,
      "macro_f1": 0.2829736211031175,
      "mae_return": 0.0030797892276582,
      "rmse_return": 0.0040875736577718,
      "mae_volatility": 0.0008732428357801,
      "bias_return": -0.0003648242014652
    },
    {
      "experiment": "topology_only",
      "n_days": 80,
      "breakout_accuracy": 0.6875,
      "macro_f1": 0.2756892230576441,
      "mae_return": 0.0033515923207144,
      "rmse_return": 0.0044223656168784,
      "mae_volatility": 0.0007297158009067,
      "bias_return": -5.692761566197663e-05
    }
  ]
}
```

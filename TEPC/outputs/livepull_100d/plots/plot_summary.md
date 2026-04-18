# TEPC Plot Summary

Best experiment: `chaos_only`

| experiment     |   n_days |   breakout_accuracy |   macro_f1 |   mae_return |   rmse_return |   mae_volatility |   bias_return |
|:---------------|---------:|--------------------:|-----------:|-------------:|--------------:|-----------------:|--------------:|
| chaos_only     |      100 |                0.8  |   0.40398  |   0.00345361 |    0.00483634 |       0.00131974 |  -0.000696786 |
| topology_chaos |      100 |                0.71 |   0.371553 |   0.00344817 |    0.00482077 |       0.00122347 |  -0.000332338 |
| topology_only  |      100 |                0.68 |   0.371438 |   0.00349007 |    0.00492029 |       0.00124144 |   1.06114e-05 |
| tepc_full      |      100 |                0.75 |   0.35645  |   0.00343033 |    0.00485305 |       0.00124071 |  -0.00041619  |
| macro_baseline |      100 |                0.72 |   0.282353 |   0.00362654 |    0.00518738 |       0.00134762 |  -0.000697026 |

```json
{
  "best_experiment": "chaos_only",
  "top_experiments": [
    "chaos_only",
    "topology_chaos",
    "topology_only"
  ],
  "metric_table": [
    {
      "experiment": "chaos_only",
      "n_days": 100,
      "breakout_accuracy": 0.8,
      "macro_f1": 0.403980463980464,
      "mae_return": 0.0034536062100437,
      "rmse_return": 0.0048363416304379,
      "mae_volatility": 0.0013197437542048,
      "bias_return": -0.000696785871772
    },
    {
      "experiment": "topology_chaos",
      "n_days": 100,
      "breakout_accuracy": 0.71,
      "macro_f1": 0.3715528781793842,
      "mae_return": 0.0034481694788665,
      "rmse_return": 0.0048207660095876,
      "mae_volatility": 0.0012234736205273,
      "bias_return": -0.0003323379631192
    },
    {
      "experiment": "topology_only",
      "n_days": 100,
      "breakout_accuracy": 0.68,
      "macro_f1": 0.3714377865321261,
      "mae_return": 0.003490073276369,
      "rmse_return": 0.0049202942845209,
      "mae_volatility": 0.0012414411143327,
      "bias_return": 1.0611368029019265e-05
    },
    {
      "experiment": "tepc_full",
      "n_days": 100,
      "breakout_accuracy": 0.75,
      "macro_f1": 0.3564499484004127,
      "mae_return": 0.0034303341874066,
      "rmse_return": 0.0048530508676104,
      "mae_volatility": 0.0012407142415068,
      "bias_return": -0.0004161904050463
    },
    {
      "experiment": "macro_baseline",
      "n_days": 100,
      "breakout_accuracy": 0.72,
      "macro_f1": 0.2823529411764706,
      "mae_return": 0.003626537817351,
      "rmse_return": 0.0051873791604789,
      "mae_volatility": 0.0013476196403581,
      "bias_return": -0.0006970259169588
    }
  ]
}
```

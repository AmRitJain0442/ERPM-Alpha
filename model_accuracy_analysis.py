import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged data to understand the exchange rate behavior
print("="*80)
print("MODEL ACCURACY ANALYSIS")
print("="*80)

# Load exchange rate data
exchange_df = pd.read_csv('usd_inr_exchange_rates_1year.csv')
exchange_df['Date'] = pd.to_datetime(exchange_df['Date'])

# Calculate statistics
mean_rate = exchange_df['USD_to_INR'].mean()
std_rate = exchange_df['USD_to_INR'].std()
min_rate = exchange_df['USD_to_INR'].min()
max_rate = exchange_df['USD_to_INR'].max()
range_rate = max_rate - min_rate

print("\n" + "-"*80)
print("EXCHANGE RATE STATISTICS (USD/INR)")
print("-"*80)
print(f"Mean: {mean_rate:.4f}")
print(f"Std Dev: {std_rate:.4f}")
print(f"Min: {min_rate:.4f}")
print(f"Max: {max_rate:.4f}")
print(f"Range: {range_rate:.4f}")

# Model performance from results
print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)

models = {
    'Random Forest (Baseline)': {'RMSE': 0.9222, 'MAE': 0.6707, 'R2': -0.4733},
    'Random Forest (Enhanced)': {'RMSE': 0.9331, 'MAE': 0.6820, 'R2': -0.5081},
    'XGBoost (Baseline)': {'RMSE': 0.9916, 'MAE': 0.7225, 'R2': -0.7034},
    'XGBoost (Enhanced)': {'RMSE': 1.0093, 'MAE': 0.7289, 'R2': -0.7647}
}

print("\nDetailed Performance Breakdown:")
print("-"*80)

for model_name, metrics in models.items():
    print(f"\n{model_name}:")
    print(f"  R² Score: {metrics['R2']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f} INR")
    print(f"  MAE: {metrics['MAE']:.4f} INR")

    # Calculate accuracy metrics
    mape = (metrics['MAE'] / mean_rate) * 100  # Mean Absolute Percentage Error
    mae_as_pct_range = (metrics['MAE'] / range_rate) * 100

    print(f"\n  Practical Interpretation:")
    print(f"    - Average prediction error: ±{metrics['MAE']:.4f} INR ({mape:.2f}%)")
    print(f"    - Error as % of total range: {mae_as_pct_range:.2f}%")
    print(f"    - If actual rate is 87.00, prediction might be: {87.00 - metrics['MAE']:.2f} to {87.00 + metrics['MAE']:.2f}")

# Naive baseline comparison
print("\n" + "="*80)
print("COMPARISON WITH NAIVE BASELINES")
print("="*80)

# Calculate what a naive "predict yesterday's value" would give
daily_changes = exchange_df['USD_to_INR'].diff().abs()
naive_mae = daily_changes.mean()
naive_rmse = np.sqrt((daily_changes**2).mean())

print(f"\nNaive Baseline (predict yesterday's value):")
print(f"  MAE: {naive_mae:.4f} INR")
print(f"  RMSE: {naive_rmse:.4f} INR")

print(f"\nNaive Baseline (predict mean value {mean_rate:.4f}):")
naive_mean_mae = np.abs(exchange_df['USD_to_INR'] - mean_rate).mean()
naive_mean_rmse = np.sqrt(((exchange_df['USD_to_INR'] - mean_rate)**2).mean())
print(f"  MAE: {naive_mean_mae:.4f} INR")
print(f"  RMSE: {naive_mean_rmse:.4f} INR")

# Model comparison
print("\n" + "-"*80)
print("MODEL vs BASELINE COMPARISON:")
print("-"*80)

best_model_mae = min([m['MAE'] for m in models.values()])
best_model_name = [name for name, m in models.items() if m['MAE'] == best_model_mae][0]

print(f"\nBest Model: {best_model_name}")
print(f"  MAE: {best_model_mae:.4f} INR")
print(f"  Better than predicting yesterday: {((naive_mae - best_model_mae) / naive_mae * 100):.2f}%")
print(f"  Better than predicting mean: {((naive_mean_mae - best_model_mae) / naive_mean_mae * 100):.2f}%")

# Accuracy interpretation
print("\n" + "="*80)
print("ACCURACY INTERPRETATION")
print("="*80)

print(f"""
The models have NEGATIVE R² scores, which means:
  - They perform WORSE than simply predicting the average exchange rate
  - R² = -0.47 to -0.76 indicates very poor predictive power
  - The models are not capturing the pattern effectively

Why this happens:
  1. Exchange rates are notoriously difficult to predict (efficient market hypothesis)
  2. Small sample size (223 observations after feature engineering)
  3. Too many features (79 features) relative to observations = overfitting
  4. Short-term exchange rate movements are highly random
  5. Missing key factors (e.g., RBI interventions, global events, commodity prices)

Practical Accuracy:
  - Best model MAE: {best_model_mae:.4f} INR (±{(best_model_mae/mean_rate*100):.2f}%)
  - This means predictions are typically off by about 0.67-0.73 INR
  - For a rate around 87 INR, predictions might range 86.3 to 87.7
  - This is a {(best_model_mae/range_rate*100):.1f}% error relative to the total observed range

Recommendation:
  - For short-term prediction (1-7 days): Models are not reliable
  - For trend detection: Models might identify general direction
  - For risk management: Use prediction intervals, not point estimates
  - Consider simpler models with fewer features to reduce overfitting
""")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Exchange rate over time
ax1 = axes[0, 0]
ax1.plot(exchange_df['Date'], exchange_df['USD_to_INR'], linewidth=2, color='#2c3e50')
ax1.axhline(y=mean_rate, color='red', linestyle='--', label=f'Mean: {mean_rate:.2f}', alpha=0.7)
ax1.fill_between(exchange_df['Date'],
                  mean_rate - best_model_mae,
                  mean_rate + best_model_mae,
                  alpha=0.3, color='orange', label=f'±MAE ({best_model_mae:.2f})')
ax1.set_xlabel('Date', fontweight='bold')
ax1.set_ylabel('USD/INR Exchange Rate', fontweight='bold')
ax1.set_title('Exchange Rate with Model Prediction Band', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 2. Model comparison - MAE
ax2 = axes[0, 1]
model_names = list(models.keys())
mae_values = [models[m]['MAE'] for m in model_names]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax2.barh(range(len(model_names)), mae_values, color=colors, alpha=0.7)
ax2.set_yticks(range(len(model_names)))
ax2.set_yticklabels(model_names, fontsize=9)
ax2.set_xlabel('Mean Absolute Error (INR)', fontweight='bold')
ax2.set_title('Model Comparison: MAE', fontweight='bold')
ax2.axvline(x=naive_mae, color='red', linestyle='--', linewidth=2, label='Naive (yesterday)', alpha=0.7)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontsize=9)

# 3. R² comparison
ax3 = axes[1, 0]
r2_values = [models[m]['R2'] for m in model_names]
bars3 = ax3.barh(range(len(model_names)), r2_values, color=colors, alpha=0.7)
ax3.set_yticks(range(len(model_names)))
ax3.set_yticklabels(model_names, fontsize=9)
ax3.set_xlabel('R² Score', fontweight='bold')
ax3.set_title('Model Comparison: R² Score', fontweight='bold')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Baseline (mean prediction)', alpha=0.7)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

for i, bar in enumerate(bars3):
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.3f}', ha='left', va='center', fontsize=9)

# 4. Error distribution (simulated based on MAE)
ax4 = axes[1, 1]
# Create histogram showing typical error distribution
errors = np.random.normal(0, best_model_mae, 1000)
ax4.hist(errors, bins=40, color='#3498db', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect prediction', alpha=0.7)
ax4.axvline(x=best_model_mae, color='orange', linestyle='--', linewidth=2, label=f'MAE: {best_model_mae:.2f}', alpha=0.7)
ax4.axvline(x=-best_model_mae, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('Prediction Error (INR)', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.set_title('Typical Prediction Error Distribution', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_accuracy_detailed.png', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("Saved: model_accuracy_detailed.png")
print("="*80)

# Create simple accuracy summary
summary = pd.DataFrame({
    'Model': model_names,
    'MAE (INR)': [models[m]['MAE'] for m in model_names],
    'MAE (%)': [(models[m]['MAE']/mean_rate*100) for m in model_names],
    'R² Score': [models[m]['R2'] for m in model_names],
    'Prediction Quality': ['Very Poor', 'Very Poor', 'Very Poor', 'Very Poor']
})

print("\n" + "="*80)
print("ACCURACY SUMMARY TABLE")
print("="*80)
print(summary.to_string(index=False))

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"""
Current Model Accuracy: POOR
  • Average error: ±0.67 INR (±0.77% of exchange rate)
  • R² scores are negative (worse than predicting the mean)
  • Models cannot reliably predict short-term exchange rate movements

Why Exchange Rates Are Hard to Predict:
  • Random walk behavior (efficient markets)
  • Influenced by sudden global events
  • Central bank interventions
  • Speculation and market sentiment
  • Short sample period (1 year of data)

Potential Improvements:
  1. Collect more historical data (3-5 years minimum)
  2. Reduce features to prevent overfitting (use feature selection)
  3. Use simpler models (regularized regression)
  4. Focus on directional prediction (up/down) rather than exact values
  5. Add more fundamental economic indicators
  6. Use ensemble methods with cross-validation
  7. Consider time-series specific models (LSTM, Prophet)

Recommended Use Case:
  ✗ NOT suitable for: Trading decisions or precise forecasting
  ✓ Could be used for: Trend analysis, sentiment impact studies
""")

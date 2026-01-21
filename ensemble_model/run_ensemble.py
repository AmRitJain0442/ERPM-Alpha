"""
Simplified Ensemble Model - Fast Execution
==========================================

This script runs a faster version of the ensemble model by:
1. Using pre-computed model predictions
2. Optimizing weights using a simpler approach
3. Generating comparison graphs

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# Statistical Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Optimization
from scipy.optimize import minimize

np.random.seed(42)
tf.random.set_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load exchange rate and GDELT data."""
    print("Loading data...")

    # Try to load 10-year data first
    data_path = os.path.join(OUTPUT_DIR, 'usd_inr_10year.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'usd_inr_exchange_rates_1year.csv')

    df = pd.read_csv(data_path, parse_dates=['Date'])
    print(f"  Loaded {len(df)} observations")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Load GDELT features
    gdelt_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'Phase-B', 'merged_training_data.csv')
    gdelt_df = None
    if os.path.exists(gdelt_path):
        gdelt_df = pd.read_csv(gdelt_path, parse_dates=['Date'])
        print(f"  Loaded GDELT: {len(gdelt_df)} observations")

    return df, gdelt_df


def vmd_decompose_simple(signal, K=3):
    """Simple signal decomposition using rolling means and differences."""
    signal = np.array(signal)

    # Trend: Long-term moving average
    trend = pd.Series(signal).rolling(window=50, min_periods=1).mean().values

    # Seasonality: Medium-term pattern
    detrended = signal - trend
    seasonality = pd.Series(detrended).rolling(window=10, min_periods=1).mean().values

    # Noise: Residual
    noise = detrended - seasonality

    return {'trend': trend, 'seasonality': seasonality, 'noise': noise}


def train_lstm(series, lookback=20, epochs=20):
    """Train LSTM model on series."""
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.1),
        LSTM(16),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1,
              callbacks=[EarlyStopping(patience=5)], verbose=0)

    # Get fitted values
    fitted = np.full(len(series), np.nan)
    preds = model.predict(X, verbose=0)
    fitted[lookback:] = scaler.inverse_transform(preds).flatten()

    return fitted, model, scaler


def train_garch(returns):
    """Train GARCH model on returns."""
    returns_scaled = returns * 100
    model = arch_model(returns_scaled, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    result = model.fit(disp='off', show_warning=False)

    # Get fitted mean and conditional volatility
    fitted_mean = result.params['mu'] / 100
    cond_vol = result.conditional_volatility / 100

    return fitted_mean, cond_vol, result


def run_monte_carlo(current_price, mean_return, volatility, horizon=30, n_sims=1000):
    """Run Monte Carlo simulation."""
    paths = np.zeros((n_sims, horizon + 1))
    paths[:, 0] = current_price

    for t in range(1, horizon + 1):
        Z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mean_return - 0.5 * volatility**2) + volatility * Z
        )

    return {
        'mean': np.mean(paths, axis=0),
        'median': np.median(paths, axis=0),
        'p5': np.percentile(paths, 5, axis=0),
        'p25': np.percentile(paths, 25, axis=0),
        'p75': np.percentile(paths, 75, axis=0),
        'p95': np.percentile(paths, 95, axis=0),
        'paths': paths
    }


def optimize_weights(predictions, actual):
    """Optimize ensemble weights using scipy minimize."""
    def objective(w):
        w = np.array(w)
        w = w / w.sum()  # Normalize
        pred = sum(w[i] * predictions[i] for i in range(len(predictions)))
        return mean_squared_error(actual, pred)

    n_models = len(predictions)
    x0 = [1/n_models] * n_models
    bounds = [(0, 1)] * n_models
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x / result.x.sum()


def main():
    print("=" * 70)
    print("  ENSEMBLE USD/INR EXCHANGE RATE MODEL")
    print("=" * 70)

    # Load data
    df, gdelt_df = load_data()

    price_series = pd.Series(df['USD_to_INR'].values, index=df['Date'])
    returns = price_series.pct_change().dropna()

    # Use last 2 years for modeling (more stable)
    cutoff_date = price_series.index.max() - pd.Timedelta(days=730)
    recent_prices = price_series[price_series.index >= cutoff_date]
    recent_returns = recent_prices.pct_change().dropna()

    print(f"\nUsing {len(recent_prices)} recent observations for modeling")

    # Split into train/validation
    train_size = int(len(recent_prices) * 0.85)
    train_prices = recent_prices.iloc[:train_size]
    val_prices = recent_prices.iloc[train_size:]
    val_dates = recent_prices.index[train_size:]

    print(f"Training: {len(train_prices)}, Validation: {len(val_prices)}")

    # ==========================================
    # MODEL 1: VMD + Trend Extrapolation
    # ==========================================
    print("\n[1/5] VMD Decomposition + Trend Extrapolation...")
    components = vmd_decompose_simple(recent_prices.values)

    # Extrapolate trend using Holt-Winters
    trend_series = pd.Series(components['trend'][:-len(val_prices)],
                             index=train_prices.index)
    hw_model = ExponentialSmoothing(trend_series, trend='add', seasonal=None).fit()
    trend_forecast = hw_model.forecast(len(val_prices))

    # Add back average seasonality
    avg_seasonality = np.mean(components['seasonality'])
    vmd_predictions = trend_forecast.values + avg_seasonality

    # ==========================================
    # MODEL 2: LSTM
    # ==========================================
    print("[2/5] LSTM Model...")
    lstm_fitted, lstm_model, lstm_scaler = train_lstm(train_prices, lookback=20, epochs=15)

    # Predict validation period
    lstm_predictions = []
    input_seq = lstm_scaler.transform(train_prices.values[-20:].reshape(-1, 1)).flatten()
    for _ in range(len(val_prices)):
        X_pred = input_seq[-20:].reshape(1, 20, 1)
        pred = lstm_model.predict(X_pred, verbose=0)[0, 0]
        lstm_predictions.append(lstm_scaler.inverse_transform([[pred]])[0, 0])
        input_seq = np.append(input_seq, pred)
    lstm_predictions = np.array(lstm_predictions)

    # ==========================================
    # MODEL 3: GARCH
    # ==========================================
    print("[3/5] GARCH Model...")
    train_returns = train_prices.pct_change().dropna()
    garch_mean, garch_vol, garch_result = train_garch(train_returns)

    # Forecast returns using mean return
    last_price = train_prices.iloc[-1]
    garch_predictions = [last_price]
    for i in range(len(val_prices)):
        garch_predictions.append(garch_predictions[-1] * (1 + garch_mean))
    garch_predictions = np.array(garch_predictions[1:])

    # ==========================================
    # MODEL 4: Monte Carlo
    # ==========================================
    print("[4/5] Monte Carlo Simulation...")
    mc_result = run_monte_carlo(
        train_prices.iloc[-1],
        train_returns.mean(),
        train_returns.std(),
        horizon=len(val_prices),
        n_sims=5000
    )
    mc_predictions = mc_result['median'][1:]

    # ==========================================
    # MODEL 5: GDELT Regression (if available)
    # ==========================================
    print("[5/5] GDELT Regression Model...")
    if gdelt_df is not None and len(gdelt_df) > 50:
        # Merge GDELT with prices
        merged = pd.merge(
            recent_prices.reset_index().rename(columns={'index': 'Date', 0: 'Price'}),
            gdelt_df,
            on='Date',
            how='inner'
        )

        if len(merged) > 50:
            feature_cols = [c for c in gdelt_df.columns if c != 'Date']
            X = merged[feature_cols].fillna(0).values
            y = merged['Price'].values

            # Train Ridge regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            ridge = Ridge(alpha=1.0).fit(X_scaled[:-len(val_prices)], y[:-len(val_prices)])

            # Predict
            gdelt_predictions = ridge.predict(X_scaled[-len(val_prices):])
        else:
            gdelt_predictions = np.full(len(val_prices), train_prices.mean())
    else:
        gdelt_predictions = np.full(len(val_prices), train_prices.mean())

    # ==========================================
    # OPTIMIZE WEIGHTS
    # ==========================================
    print("\nOptimizing ensemble weights...")

    actual = val_prices.values
    predictions = [vmd_predictions, lstm_predictions, garch_predictions, mc_predictions, gdelt_predictions]
    model_names = ['VMD', 'LSTM', 'GARCH', 'Monte Carlo', 'GDELT']

    # Handle length mismatches
    min_len = min(len(actual), min(len(p) for p in predictions))
    actual = actual[:min_len]
    predictions = [p[:min_len] for p in predictions]

    weights = optimize_weights(predictions, actual)

    # Ensemble prediction
    ensemble_pred = sum(w * p for w, p in zip(weights, predictions))

    # ==========================================
    # RESULTS
    # ==========================================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\nOptimal Weights:")
    for name, w in zip(model_names, weights):
        print(f"  {name}: {w:.4f}")

    print("\nModel Performance (Validation):")
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 45)

    for name, pred in zip(model_names, predictions):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        print(f"{name:<15} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")

    # Ensemble metrics
    ens_rmse = np.sqrt(mean_squared_error(actual, ensemble_pred))
    ens_mae = mean_absolute_error(actual, ensemble_pred)
    ens_r2 = r2_score(actual, ensemble_pred)
    print("-" * 45)
    print(f"{'ENSEMBLE':<15} {ens_rmse:<10.4f} {ens_mae:<10.4f} {ens_r2:<10.4f}")

    # ==========================================
    # VISUALIZATIONS
    # ==========================================
    print("\nGenerating visualizations...")

    # 1. VMD Decomposition
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    axes[0].plot(recent_prices.index, recent_prices.values, 'b-', linewidth=1)
    axes[0].set_title('Original Exchange Rate (USD/INR)', fontweight='bold')
    axes[0].set_ylabel('INR')

    axes[1].plot(recent_prices.index, components['trend'], 'g-', linewidth=1)
    axes[1].set_title('Trend Component', fontweight='bold')

    axes[2].plot(recent_prices.index, components['seasonality'], 'orange', linewidth=1)
    axes[2].set_title('Seasonality Component', fontweight='bold')

    axes[3].plot(recent_prices.index, components['noise'], 'r-', linewidth=0.5, alpha=0.7)
    axes[3].set_title('Noise Component', fontweight='bold')
    axes[3].set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vmd_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: vmd_decomposition.png")

    # 2. Model Weights
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = ax.bar(model_names, weights, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Weight', fontweight='bold')
    ax.set_title('Optimized Ensemble Weights', fontweight='bold', fontsize=14)
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{w:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ensemble_weights.png")

    # 3. Validation Predictions
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    val_dates_arr = val_dates[:min_len]
    ax1.plot(val_dates_arr, actual, 'k-', label='Actual', linewidth=2)
    ax1.plot(val_dates_arr, ensemble_pred, 'r--', label='Ensemble', linewidth=2)
    for name, pred, color in zip(model_names, predictions, colors):
        ax1.plot(val_dates_arr, pred, '--', label=name, alpha=0.5, linewidth=1, color=color)
    ax1.set_title('Validation: Model Predictions vs Actual', fontweight='bold')
    ax1.set_ylabel('USD/INR')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    error = actual - ensemble_pred
    ax2.fill_between(val_dates_arr, error, 0, alpha=0.3, color='red')
    ax2.plot(val_dates_arr, error, 'r-', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-')
    ax2.set_title('Ensemble Prediction Error', fontweight='bold')
    ax2.set_ylabel('Error (INR)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'validation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: validation_comparison.png")

    # 4. Model Performance Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rmse_vals = [np.sqrt(mean_squared_error(actual, p)) for p in predictions]
    rmse_vals.append(ens_rmse)
    mae_vals = [mean_absolute_error(actual, p) for p in predictions]
    mae_vals.append(ens_mae)
    labels = model_names + ['ENSEMBLE']

    ax1 = axes[0]
    bars1 = ax1.bar(labels, rmse_vals, color=colors + ['red'], edgecolor='black', alpha=0.8)
    ax1.set_ylabel('RMSE', fontweight='bold')
    ax1.set_title('RMSE Comparison', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars1, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=8)

    ax2 = axes[1]
    bars2 = ax2.bar(labels, mae_vals, color=colors + ['red'], edgecolor='black', alpha=0.8)
    ax2.set_ylabel('MAE', fontweight='bold')
    ax2.set_title('MAE Comparison', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars2, mae_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: model_comparison.png")

    # 5. 30-Day Forecast with Confidence Intervals
    print("\nGenerating 30-day forecast...")
    current_price = recent_prices.iloc[-1]
    mc_forecast = run_monte_carlo(
        current_price,
        recent_returns.mean(),
        recent_returns.std(),
        horizon=30,
        n_sims=10000
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    # Historical (last 90 days)
    hist_days = 90
    hist_idx = recent_prices.index[-hist_days:]
    hist_vals = recent_prices.values[-hist_days:]
    ax.plot(hist_idx, hist_vals, 'b-', linewidth=2, label='Historical')

    # Forecast
    last_date = recent_prices.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=31, freq='B')
    ax.plot(forecast_dates, mc_forecast['median'], 'r-', linewidth=2, label='Forecast (Median)')
    ax.fill_between(forecast_dates, mc_forecast['p5'], mc_forecast['p95'],
                    alpha=0.2, color='red', label='90% CI')
    ax.fill_between(forecast_dates, mc_forecast['p25'], mc_forecast['p75'],
                    alpha=0.3, color='red', label='50% CI')
    ax.axvline(x=last_date, color='gray', linestyle='--', linewidth=1)

    ax.set_title('USD/INR Exchange Rate: 30-Day Forecast', fontweight='bold', fontsize=14)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Exchange Rate (INR)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'forecast_30day.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: forecast_30day.png")

    # 6. Summary Dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Full price history
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(price_series.index, price_series.values, 'b-', linewidth=0.8)
    ax1.set_title('USD/INR Exchange Rate (Full History)', fontweight='bold')
    ax1.set_ylabel('INR')
    ax1.grid(True, alpha=0.3)

    # Weights pie
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.pie(weights, labels=model_names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Ensemble Weights', fontweight='bold')

    # VMD components
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(components['trend'], 'g-', linewidth=1)
    ax3.set_title('Trend', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(components['seasonality'], 'orange', linewidth=1)
    ax4.set_title('Seasonality', fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(components['noise'], 'r-', linewidth=0.5, alpha=0.7)
    ax5.set_title('Noise', fontweight='bold')

    # Validation plot
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.plot(val_dates_arr, actual, 'k-', label='Actual', linewidth=2)
    ax6.plot(val_dates_arr, ensemble_pred, 'r--', label='Ensemble', linewidth=2)
    ax6.set_title('Validation: Ensemble vs Actual', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Metrics box
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    metrics_text = f"""
    ENSEMBLE PERFORMANCE
    ════════════════════════

    RMSE:  {ens_rmse:.4f}
    MAE:   {ens_mae:.4f}
    R²:    {ens_r2:.4f}

    ════════════════════════

    30-DAY FORECAST
    Current: {current_price:.2f} INR
    Median:  {mc_forecast['median'][-1]:.2f} INR
    90% CI:  [{mc_forecast['p5'][-1]:.2f}, {mc_forecast['p95'][-1]:.2f}]
    """
    ax7.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
            verticalalignment='center', transform=ax7.transAxes)

    plt.suptitle('Ensemble Exchange Rate Model - Summary Dashboard',
                fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ensemble_dashboard.png")

    # Save results to CSV
    weights_df = pd.DataFrame({
        'Model': model_names,
        'Weight': weights
    })
    weights_df.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_weights.csv'), index=False)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2'],
        'Value': [ens_rmse, ens_mae, ens_r2]
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_metrics.csv'), index=False)

    forecast_df = pd.DataFrame({
        'Day': range(31),
        'Median': mc_forecast['median'],
        'P5': mc_forecast['p5'],
        'P25': mc_forecast['p25'],
        'P75': mc_forecast['p75'],
        'P95': mc_forecast['p95']
    })
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_forecast.csv'), index=False)

    print("\n" + "=" * 70)
    print("  ENSEMBLE MODEL COMPLETE!")
    print("=" * 70)
    print(f"\nOptimal Weights:")
    for name, w in zip(model_names, weights):
        print(f"  {name}: {w:.4f}")
    print(f"\nValidation RMSE: {ens_rmse:.4f}")
    print(f"Validation R²: {ens_r2:.4f}")
    print(f"\nCurrent rate: {current_price:.4f} INR")
    print(f"30-day forecast: {mc_forecast['median'][-1]:.4f} INR")
    print(f"90% CI: [{mc_forecast['p5'][-1]:.4f}, {mc_forecast['p95'][-1]:.4f}]")
    print(f"\nOutput saved to: {OUTPUT_DIR}")

    return weights, metrics_df


if __name__ == "__main__":
    weights, metrics = main()

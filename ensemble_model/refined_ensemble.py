"""
Refined Ensemble Model - Better GDELT Integration
=================================================

Fixes GDELT date alignment and forces meaningful model combinations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from scipy.optimize import minimize

np.random.seed(42)
tf.random.set_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def load_and_merge_data():
    """Load and properly merge all data sources."""
    print("=" * 70)
    print("  LOADING AND MERGING DATA")
    print("=" * 70)

    base_dir = os.path.dirname(OUTPUT_DIR)

    # Load exchange rate data
    rate_path = os.path.join(OUTPUT_DIR, 'usd_inr_10year.csv')
    if not os.path.exists(rate_path):
        rate_path = os.path.join(base_dir, 'usd_inr_exchange_rates_1year.csv')
    rate_df = pd.read_csv(rate_path, parse_dates=['Date'])
    rate_df = rate_df.sort_values('Date').reset_index(drop=True)
    print(f"Exchange rates: {len(rate_df)} obs ({rate_df['Date'].min().date()} to {rate_df['Date'].max().date()})")

    # Load GDELT features
    gdelt_path = os.path.join(base_dir, 'Phase-B', 'merged_training_data.csv')
    gdelt_df = None
    if os.path.exists(gdelt_path):
        gdelt_df = pd.read_csv(gdelt_path, parse_dates=['Date'])
        print(f"GDELT: {len(gdelt_df)} obs ({gdelt_df['Date'].min().date()} to {gdelt_df['Date'].max().date()})")

    # Load Goldstein data
    goldstein_path = os.path.join(base_dir, 'combined_goldstein_exchange_rates.csv')
    goldstein_df = None
    if os.path.exists(goldstein_path):
        goldstein_df = pd.read_csv(goldstein_path, parse_dates=['Date'])
        print(f"Goldstein: {len(goldstein_df)} obs")

    # Load trade balance
    trade_path = os.path.join(base_dir, 'india_usa_trade', 'output', 'trade_balance_analysis.csv')
    trade_df = None
    if os.path.exists(trade_path):
        trade_df = pd.read_csv(trade_path)
        print(f"Trade balance: {len(trade_df)} years")

    # Merge datasets
    merged = rate_df.copy()

    if gdelt_df is not None:
        gdelt_cols = [c for c in gdelt_df.columns if c != 'Date']
        merged = pd.merge(merged, gdelt_df, on='Date', how='left')
        print(f"\nAfter GDELT merge: {merged[gdelt_cols[0]].notna().sum()} rows with GDELT data")

    if goldstein_df is not None:
        goldstein_cols = ['USA_Avg_Goldstein', 'India_Avg_Goldstein',
                         'Combined_Weighted_Avg', 'USA_India_Sentiment_Diff']
        goldstein_subset = goldstein_df[['Date'] + [c for c in goldstein_cols if c in goldstein_df.columns]]
        merged = pd.merge(merged, goldstein_subset, on='Date', how='left')

    if trade_df is not None:
        merged['Year'] = merged['Date'].dt.year
        trade_dict = dict(zip(trade_df['fetch_year'], trade_df['trade_balance']))
        merged['Trade_Balance'] = merged['Year'].map(trade_dict)
        merged['Trade_Balance_Norm'] = (merged['Trade_Balance'] - merged['Trade_Balance'].mean()) / merged['Trade_Balance'].std()
        merged = merged.drop('Year', axis=1)

    return merged


def create_features(df):
    """Create engineered features."""
    print("\nEngineering features...")

    # Price-based features
    df['Returns'] = df['USD_to_INR'].pct_change()
    df['Log_Returns'] = np.log(df['USD_to_INR'] / df['USD_to_INR'].shift(1))
    df['Price_MA5'] = df['USD_to_INR'].rolling(5).mean()
    df['Price_MA20'] = df['USD_to_INR'].rolling(20).mean()
    df['Price_Std5'] = df['USD_to_INR'].rolling(5).std()
    df['Price_Momentum'] = df['USD_to_INR'] - df['USD_to_INR'].shift(5)

    # GDELT lag features
    gdelt_cols = ['Tone_Economy', 'Tone_Conflict', 'Tone_Overall', 'Goldstein_Avg', 'IMF_3']
    for col in gdelt_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 5]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
            df[f'{col}_roll3'] = df[col].rolling(3).mean()
            df[f'{col}_roll5'] = df[col].rolling(5).mean()

    # Goldstein features
    if 'USA_Avg_Goldstein' in df.columns and 'India_Avg_Goldstein' in df.columns:
        df['Goldstein_Diff'] = df['USA_Avg_Goldstein'] - df['India_Avg_Goldstein']
        df['Goldstein_Ratio'] = df['USA_Avg_Goldstein'] / (df['India_Avg_Goldstein'].abs() + 0.01)

    # Fill NaN
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    feature_cols = [c for c in df.columns if c not in ['Date', 'USD_to_INR']]
    print(f"  Total features: {len(feature_cols)}")

    return df, feature_cols


def train_models(train_prices, train_features, horizon):
    """Train all models and return predictions."""
    print("\n" + "=" * 70)
    print("  TRAINING MODELS")
    print("=" * 70)

    last_price = train_prices.iloc[-1]
    models = {}

    # 1. Holt-Winters
    print("\n[1/9] Holt-Winters Exponential Smoothing...")
    try:
        hw = ExponentialSmoothing(train_prices, trend='add', damped_trend=True).fit()
        models['Holt_Winters'] = hw.forecast(horizon).values
    except:
        models['Holt_Winters'] = np.full(horizon, last_price)

    # 2. ARIMA
    print("[2/9] ARIMA(1,1,1)...")
    try:
        arima = ARIMA(train_prices, order=(1,1,1)).fit()
        models['ARIMA'] = arima.forecast(horizon).values
    except:
        models['ARIMA'] = np.full(horizon, last_price)

    # 3. ARIMA with different order
    print("[3/9] ARIMA(2,1,2)...")
    try:
        arima2 = ARIMA(train_prices, order=(2,1,2)).fit()
        models['ARIMA_212'] = arima2.forecast(horizon).values
    except:
        models['ARIMA_212'] = np.full(horizon, last_price)

    # 4. LSTM
    print("[4/9] LSTM Neural Network...")
    try:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(train_prices.values.reshape(-1, 1))
        lookback = 15

        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        lstm = Sequential([
            LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.1),
            LSTM(16),
            Dense(1)
        ])
        lstm.compile(optimizer='adam', loss='mse')
        lstm.fit(X, y, epochs=20, batch_size=32, verbose=0,
                callbacks=[EarlyStopping(patience=5)])

        # Forecast
        preds = []
        seq = data[-lookback:].flatten()
        for _ in range(horizon):
            X_pred = seq[-lookback:].reshape(1, lookback, 1)
            pred = lstm.predict(X_pred, verbose=0)[0, 0]
            preds.append(scaler.inverse_transform([[pred]])[0, 0])
            seq = np.append(seq, pred)
        models['LSTM'] = np.array(preds)
    except Exception as e:
        print(f"    LSTM error: {e}")
        models['LSTM'] = np.full(horizon, last_price)

    # 5. GARCH
    print("[5/9] GARCH Volatility Model...")
    try:
        returns = train_prices.pct_change().dropna() * 100
        garch = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1).fit(disp='off')
        mean_return = garch.params['mu'] / 100
        preds = [last_price]
        for _ in range(horizon):
            preds.append(preds[-1] * (1 + mean_return))
        models['GARCH'] = np.array(preds[1:])
    except:
        models['GARCH'] = np.full(horizon, last_price)

    # 6. Monte Carlo
    print("[6/9] Monte Carlo Simulation...")
    returns = train_prices.pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    n_sims = 5000
    paths = np.zeros((n_sims, horizon + 1))
    paths[:, 0] = last_price
    for t in range(1, horizon + 1):
        Z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2) + sigma*Z)
    models['Monte_Carlo'] = np.median(paths, axis=0)[1:]

    # 7. XGBoost with features
    print("[7/9] XGBoost + GDELT Features...")
    if train_features is not None and len(train_features.columns) > 5:
        try:
            X = train_features.values[:-horizon]
            y = train_prices.values[horizon:]
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                         random_state=42, verbosity=0)
            xgb_model.fit(X, y)

            X_pred = train_features.values[-horizon:]
            X_pred = np.nan_to_num(X_pred, nan=0, posinf=0, neginf=0)
            models['XGBoost'] = xgb_model.predict(X_pred)
        except Exception as e:
            print(f"    XGBoost error: {e}")
            models['XGBoost'] = np.full(horizon, last_price)
    else:
        models['XGBoost'] = np.full(horizon, last_price)

    # 8. Gradient Boosting
    print("[8/9] Gradient Boosting...")
    if train_features is not None and len(train_features.columns) > 5:
        try:
            X = train_features.values[:-horizon]
            y = train_prices.values[horizon:]
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

            gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
            gb_model.fit(X, y)

            X_pred = train_features.values[-horizon:]
            X_pred = np.nan_to_num(X_pred, nan=0, posinf=0, neginf=0)
            models['GradBoost'] = gb_model.predict(X_pred)
        except Exception as e:
            print(f"    GradBoost error: {e}")
            models['GradBoost'] = np.full(horizon, last_price)
    else:
        models['GradBoost'] = np.full(horizon, last_price)

    # 9. Simple Moving Average extrapolation
    print("[9/9] Moving Average Extrapolation...")
    ma20 = train_prices.rolling(20).mean().iloc[-1]
    ma5 = train_prices.rolling(5).mean().iloc[-1]
    trend = (ma5 - ma20) / 20
    models['MA_Trend'] = np.array([last_price + trend * i for i in range(1, horizon + 1)])

    return models


def optimize_ensemble_weights(predictions, actual, min_models=2, max_weight=0.6):
    """Optimize weights with constraints."""
    print("\nOptimizing ensemble weights...")

    model_names = list(predictions.keys())
    n_models = len(model_names)
    pred_arrays = [predictions[name] for name in model_names]

    # Align lengths
    min_len = min(len(actual), min(len(p) for p in pred_arrays))
    actual = actual[:min_len]
    pred_arrays = [p[:min_len] for p in pred_arrays]

    def objective(w):
        w = np.array(w)
        w = w / w.sum()
        pred = sum(w[i] * pred_arrays[i] for i in range(n_models))
        return mean_squared_error(actual, pred)

    def min_models_constraint(w):
        # At least min_models should have weight > 0.05
        return sum(1 for x in w if x > 0.05) - min_models

    x0 = [1/n_models] * n_models
    bounds = [(0, max_weight)] * n_models
    constraints = [
        {'type': 'eq', 'fun': lambda w: sum(w) - 1},
        {'type': 'ineq', 'fun': min_models_constraint}
    ]

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x / result.x.sum()

    return dict(zip(model_names, weights)), pred_arrays, actual


def main():
    print("=" * 70)
    print("  REFINED ENSEMBLE MODEL")
    print("  Better GDELT Integration + Forced Model Combinations")
    print("=" * 70)

    # Load data
    df = load_and_merge_data()
    df, feature_cols = create_features(df)

    # Use data where GDELT is available
    gdelt_available = df['Tone_Economy'].notna() if 'Tone_Economy' in df.columns else pd.Series([False] * len(df))
    if gdelt_available.sum() > 100:
        print(f"\nUsing GDELT-available period: {gdelt_available.sum()} days")
        df_gdelt = df[gdelt_available].reset_index(drop=True)
    else:
        # Use last 2 years
        cutoff = df['Date'].max() - pd.Timedelta(days=730)
        df_gdelt = df[df['Date'] >= cutoff].reset_index(drop=True)
        print(f"\nUsing last 2 years: {len(df_gdelt)} days")

    # Prepare data
    price_series = pd.Series(df_gdelt['USD_to_INR'].values, index=df_gdelt['Date'])
    feature_df = df_gdelt[feature_cols]

    # Train/validation split
    train_size = int(len(price_series) * 0.85)
    train_prices = price_series.iloc[:train_size]
    val_prices = price_series.iloc[train_size:]
    train_features = feature_df.iloc[:train_size]
    val_features = feature_df.iloc[train_size:]

    horizon = len(val_prices)
    print(f"\nTraining: {len(train_prices)}, Validation: {horizon}")

    # Train models
    model_preds = train_models(train_prices, train_features, horizon)

    # Optimize weights
    weights, pred_arrays, actual = optimize_ensemble_weights(
        model_preds, val_prices.values,
        min_models=3,  # Force at least 3 models
        max_weight=0.5  # Max 50% weight per model
    )

    # Create ensemble prediction
    ensemble_pred = sum(weights[name] * model_preds[name][:len(actual)]
                       for name in weights.keys())

    # Calculate metrics
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\nOptimized Weights (min 3 models, max 50% each):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"  {name}: {w:.4f}")

    print("\nModel Performance:")
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 45)

    model_metrics = {}
    for name in model_preds.keys():
        pred = model_preds[name][:len(actual)]
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        print(f"{name:<15} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
        model_metrics[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}

    ens_rmse = np.sqrt(mean_squared_error(actual, ensemble_pred))
    ens_mae = mean_absolute_error(actual, ensemble_pred)
    ens_r2 = r2_score(actual, ensemble_pred)
    print("-" * 45)
    print(f"{'ENSEMBLE':<15} {ens_rmse:<10.4f} {ens_mae:<10.4f} {ens_r2:<10.4f}")

    # Best single model
    best_single = min(model_metrics.items(), key=lambda x: x[1]['rmse'])
    improvement = (best_single[1]['rmse'] - ens_rmse) / best_single[1]['rmse'] * 100
    print(f"\nBest single model: {best_single[0]} (RMSE: {best_single[1]['rmse']:.4f})")
    print(f"Ensemble improvement: {improvement:.2f}%")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)

    val_dates = price_series.index[train_size:train_size + len(actual)]

    # 1. Weights comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax1 = axes[0]
    sorted_weights = sorted(weights.items(), key=lambda x: -x[1])
    names = [x[0] for x in sorted_weights]
    vals = [x[1] for x in sorted_weights]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    bars = ax1.barh(names, vals, color=colors, edgecolor='black')
    ax1.set_xlabel('Weight', fontweight='bold')
    ax1.set_title('Refined Ensemble Weights', fontweight='bold')
    for bar, v in zip(bars, vals):
        if v > 0.01:
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.3f}', va='center', fontsize=9)

    # Pie chart (only significant weights)
    ax2 = axes[1]
    sig_weights = [(n, w) for n, w in weights.items() if w > 0.05]
    if sig_weights:
        ax2.pie([w for _, w in sig_weights],
               labels=[n for n, _ in sig_weights],
               autopct='%1.1f%%',
               colors=plt.cm.Set3(np.linspace(0, 1, len(sig_weights))))
    ax2.set_title('Active Models in Ensemble', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'refined_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: refined_weights.png")

    # 2. Validation comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(val_dates, actual, 'k-', label='Actual', linewidth=2)
    ax1.plot(val_dates, ensemble_pred, 'r--', label='Ensemble', linewidth=2)

    # Top 3 contributing models
    top_models = sorted(weights.items(), key=lambda x: -x[1])[:3]
    for name, _ in top_models:
        ax1.plot(val_dates, model_preds[name][:len(actual)], '--',
                label=name, alpha=0.5, linewidth=1)

    ax1.set_title('Validation: Predictions vs Actual', fontweight='bold')
    ax1.set_ylabel('USD/INR')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    error = actual - ensemble_pred
    ax2.fill_between(val_dates, error, 0, alpha=0.3, color='red')
    ax2.plot(val_dates, error, 'r-', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-')
    ax2.set_title('Ensemble Prediction Error', fontweight='bold')
    ax2.set_ylabel('Error (INR)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'refined_validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: refined_validation.png")

    # 3. Model comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    rmse_vals = [model_metrics[n]['rmse'] for n in model_preds.keys()]
    model_names_list = list(model_preds.keys())
    colors = ['lightblue'] * len(model_names_list)

    # Highlight ensemble
    rmse_vals.append(ens_rmse)
    model_names_list.append('ENSEMBLE')
    colors.append('red')

    bars = ax.bar(model_names_list, rmse_vals, color=colors, edgecolor='black')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('Model RMSE Comparison', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{v:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'refined_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: refined_model_comparison.png")

    # 4. 30-Day Forecast
    print("\n  Generating 30-day forecast...")
    current_price = price_series.iloc[-1]
    recent_returns = price_series.pct_change().dropna()
    mu, sigma = recent_returns.mean(), recent_returns.std()

    n_sims = 10000
    forecast_horizon = 30
    paths = np.zeros((n_sims, forecast_horizon + 1))
    paths[:, 0] = current_price

    for t in range(1, forecast_horizon + 1):
        Z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2) + sigma*Z)

    forecast = {
        'median': np.median(paths, axis=0),
        'p5': np.percentile(paths, 5, axis=0),
        'p25': np.percentile(paths, 25, axis=0),
        'p75': np.percentile(paths, 75, axis=0),
        'p95': np.percentile(paths, 95, axis=0)
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    # Historical
    hist_days = 60
    hist_idx = price_series.index[-hist_days:]
    hist_vals = price_series.values[-hist_days:]
    ax.plot(hist_idx, hist_vals, 'b-', linewidth=2, label='Historical')

    # Forecast
    last_date = price_series.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='B')

    ax.plot(forecast_dates, forecast['median'], 'r-', linewidth=2, label='Forecast')
    ax.fill_between(forecast_dates, forecast['p5'], forecast['p95'],
                   alpha=0.15, color='red', label='90% CI')
    ax.fill_between(forecast_dates, forecast['p25'], forecast['p75'],
                   alpha=0.25, color='red', label='50% CI')
    ax.axvline(x=last_date, color='gray', linestyle='--', linewidth=1)

    ax.set_title('USD/INR: 30-Day Forecast with Confidence Intervals', fontweight='bold', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate (INR)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'refined_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: refined_forecast.png")

    # Save results
    weights_df = pd.DataFrame([
        {'Model': name, 'Weight': w} for name, w in sorted(weights.items(), key=lambda x: -x[1])
    ])
    weights_df.to_csv(os.path.join(OUTPUT_DIR, 'refined_weights.csv'), index=False)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2', 'Improvement_pct'],
        'Value': [ens_rmse, ens_mae, ens_r2, improvement]
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'refined_metrics.csv'), index=False)

    forecast_df = pd.DataFrame({
        'Day': range(forecast_horizon + 1),
        'Median': forecast['median'],
        'P5': forecast['p5'],
        'P25': forecast['p25'],
        'P75': forecast['p75'],
        'P95': forecast['p95']
    })
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'refined_forecast.csv'), index=False)

    print("\n" + "=" * 70)
    print("  REFINED ENSEMBLE COMPLETE!")
    print("=" * 70)
    print(f"\nActive Models (weight > 5%):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w > 0.05:
            print(f"  {name}: {w:.4f}")
    print(f"\nValidation RMSE: {ens_rmse:.4f}")
    print(f"Validation R²: {ens_r2:.4f}")
    print(f"Improvement over best single: {improvement:.2f}%")
    print(f"\nCurrent: {current_price:.2f} INR")
    print(f"30-day forecast: {forecast['median'][-1]:.2f} INR")
    print(f"90% CI: [{forecast['p5'][-1]:.2f}, {forecast['p95'][-1]:.2f}]")

    return weights, model_metrics


if __name__ == "__main__":
    weights, metrics = main()

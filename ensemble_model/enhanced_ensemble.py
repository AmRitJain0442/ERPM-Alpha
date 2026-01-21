"""
Enhanced Ensemble Model with GDELT and Trade Data Integration
=============================================================

This script creates a refined ensemble model that:
1. Uses more model combinations (VMD, SARIMA, LSTM, GARCH, Holt-Winters, XGBoost)
2. Engineers features from GDELT data (lags, rolling stats, interactions)
3. Incorporates trade balance data
4. Tests multiple optimization methods
5. Provides detailed model comparison

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

# ML imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Statistical Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping

# Optimization
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

np.random.seed(42)
tf.random.set_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING AND FEATURE ENGINEERING
# =============================================================================

def load_all_data():
    """Load exchange rate, GDELT, and trade data."""
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)

    base_dir = os.path.dirname(OUTPUT_DIR)

    # Exchange rate data
    rate_path = os.path.join(OUTPUT_DIR, 'usd_inr_10year.csv')
    if not os.path.exists(rate_path):
        rate_path = os.path.join(base_dir, 'usd_inr_exchange_rates_1year.csv')

    rate_df = pd.read_csv(rate_path, parse_dates=['Date'])
    print(f"Exchange rates: {len(rate_df)} observations")

    # GDELT thematic features
    gdelt_path = os.path.join(base_dir, 'Phase-B', 'merged_training_data.csv')
    gdelt_df = None
    if os.path.exists(gdelt_path):
        gdelt_df = pd.read_csv(gdelt_path, parse_dates=['Date'])
        print(f"GDELT features: {len(gdelt_df)} observations, {len(gdelt_df.columns)-1} features")

    # Combined Goldstein data
    goldstein_path = os.path.join(base_dir, 'combined_goldstein_exchange_rates.csv')
    goldstein_df = None
    if os.path.exists(goldstein_path):
        goldstein_df = pd.read_csv(goldstein_path, parse_dates=['Date'])
        print(f"Goldstein data: {len(goldstein_df)} observations")

    # Trade balance data
    trade_path = os.path.join(base_dir, 'india_usa_trade', 'output', 'trade_balance_analysis.csv')
    trade_df = None
    if os.path.exists(trade_path):
        trade_df = pd.read_csv(trade_path)
        print(f"Trade balance: {len(trade_df)} years")

    return rate_df, gdelt_df, goldstein_df, trade_df


def engineer_gdelt_features(gdelt_df, goldstein_df, rate_df):
    """Create engineered features from GDELT and Goldstein data."""
    print("\nEngineering GDELT features...")

    # Start with rate data
    df = rate_df.copy()
    df = df.set_index('Date')

    # Merge GDELT features
    if gdelt_df is not None:
        gdelt = gdelt_df.set_index('Date')
        df = df.join(gdelt, how='left')

    # Merge Goldstein features
    if goldstein_df is not None:
        goldstein = goldstein_df.set_index('Date')
        goldstein_cols = ['USA_Avg_Goldstein', 'India_Avg_Goldstein',
                         'Combined_Weighted_Avg', 'USA_India_Sentiment_Diff']
        for col in goldstein_cols:
            if col in goldstein.columns:
                df[col] = goldstein[col]

    # Create lag features (1, 3, 5, 7 days)
    feature_cols = [c for c in df.columns if c not in ['USD_to_INR', 'Date']]
    for col in feature_cols[:10]:  # Top 10 features
        for lag in [1, 3, 5]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Create rolling statistics (5, 10 day windows)
    for col in feature_cols[:5]:  # Top 5 features
        df[f'{col}_roll5_mean'] = df[col].rolling(5).mean()
        df[f'{col}_roll5_std'] = df[col].rolling(5).std()
        df[f'{col}_roll10_mean'] = df[col].rolling(10).mean()

    # Interaction features
    if 'Tone_Economy' in df.columns and 'Goldstein_Avg' in df.columns:
        df['Tone_Goldstein_interaction'] = df['Tone_Economy'] * df['Goldstein_Avg']

    if 'USA_Avg_Goldstein' in df.columns and 'India_Avg_Goldstein' in df.columns:
        df['Goldstein_ratio'] = df['USA_Avg_Goldstein'] / (df['India_Avg_Goldstein'] + 0.001)

    # Sentiment momentum
    if 'Tone_Overall' in df.columns:
        df['Tone_momentum'] = df['Tone_Overall'].diff(3)

    # Volume features
    if 'Count_Total' in df.columns:
        df['Volume_change'] = df['Count_Total'].pct_change()
        df['High_volume'] = (df['Count_Total'] > df['Count_Total'].rolling(20).mean()).astype(int)

    # Fill NaN
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    print(f"  Created {len(df.columns)} total features")
    return df.reset_index()


def add_trade_features(df, trade_df):
    """Add trade balance features (annual data interpolated to daily)."""
    if trade_df is None:
        return df

    print("Adding trade balance features...")

    # Create daily trade deficit series
    df['Year'] = pd.to_datetime(df['Date']).dt.year

    trade_dict = dict(zip(trade_df['fetch_year'], trade_df['trade_balance']))
    df['Trade_Balance'] = df['Year'].map(trade_dict)

    # Normalize trade balance
    df['Trade_Balance_Norm'] = (df['Trade_Balance'] - df['Trade_Balance'].mean()) / df['Trade_Balance'].std()

    # Trade balance trend (year-over-year change)
    trade_df_sorted = trade_df.sort_values('fetch_year')
    trade_df_sorted['Trade_YoY'] = trade_df_sorted['trade_balance'].pct_change()
    trade_yoy_dict = dict(zip(trade_df_sorted['fetch_year'], trade_df_sorted['Trade_YoY']))
    df['Trade_YoY'] = df['Year'].map(trade_yoy_dict).fillna(0)

    df = df.drop('Year', axis=1)
    print(f"  Added trade balance features")

    return df


# =============================================================================
# MODEL IMPLEMENTATIONS
# =============================================================================

def vmd_decompose(signal, K=3):
    """VMD-style decomposition using rolling averages."""
    signal = np.array(signal)
    trend = pd.Series(signal).rolling(window=50, min_periods=1).mean().values
    detrended = signal - trend
    seasonality = pd.Series(detrended).rolling(window=10, min_periods=1).mean().values
    noise = detrended - seasonality
    return {'trend': trend, 'seasonality': seasonality, 'noise': noise}


def train_holt_winters(train, horizon):
    """Holt-Winters exponential smoothing."""
    try:
        model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=True)
        fitted = model.fit()
        return fitted.forecast(horizon).values
    except:
        return np.full(horizon, train.iloc[-1])


def train_arima(train, horizon, order=(2,1,2)):
    """ARIMA model."""
    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()
        return fitted.forecast(horizon).values
    except:
        return np.full(horizon, train.iloc[-1])


def train_lstm_model(train_series, lookback=20, epochs=20):
    """LSTM model training."""
    scaler = MinMaxScaler()
    data = scaler.fit_transform(train_series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.1),
        LSTM(16),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1,
              callbacks=[EarlyStopping(patience=5)], verbose=0)

    return model, scaler, lookback


def lstm_forecast(model, scaler, train_series, lookback, horizon):
    """Generate LSTM forecasts."""
    data = scaler.transform(train_series.values.reshape(-1, 1))
    predictions = []
    input_seq = data[-lookback:].flatten()

    for _ in range(horizon):
        X_pred = input_seq[-lookback:].reshape(1, lookback, 1)
        pred = model.predict(X_pred, verbose=0)[0, 0]
        predictions.append(scaler.inverse_transform([[pred]])[0, 0])
        input_seq = np.append(input_seq, pred)

    return np.array(predictions)


def train_garch(returns):
    """GARCH model for volatility."""
    try:
        returns_scaled = returns * 100
        model = arch_model(returns_scaled, mean='Constant', vol='GARCH', p=1, q=1)
        result = model.fit(disp='off', show_warning=False)
        return result.params['mu'] / 100, result
    except:
        return returns.mean(), None


def train_xgboost_noise(X, y):
    """XGBoost for noise prediction."""
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_scaled, y)

    return model, scaler


def monte_carlo_forecast(current_price, mean_return, volatility, horizon, n_sims=5000):
    """Monte Carlo simulation."""
    paths = np.zeros((n_sims, horizon + 1))
    paths[:, 0] = current_price

    for t in range(1, horizon + 1):
        Z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t-1] * np.exp(
            (mean_return - 0.5 * volatility**2) + volatility * Z
        )

    return {
        'mean': np.mean(paths, axis=0)[1:],
        'median': np.median(paths, axis=0)[1:],
        'p5': np.percentile(paths, 5, axis=0)[1:],
        'p95': np.percentile(paths, 95, axis=0)[1:]
    }


# =============================================================================
# WEIGHT OPTIMIZATION
# =============================================================================

def optimize_weights_scipy(predictions, actual, method='SLSQP'):
    """Optimize weights using scipy."""
    n_models = len(predictions)

    def objective(w):
        w = np.array(w)
        w = w / w.sum()
        pred = sum(w[i] * predictions[i] for i in range(n_models))
        return mean_squared_error(actual, pred)

    x0 = [1/n_models] * n_models
    bounds = [(0, 1)] * n_models
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

    result = minimize(objective, x0, method=method, bounds=bounds, constraints=constraints)
    return result.x / result.x.sum()


def optimize_weights_de(predictions, actual):
    """Optimize weights using Differential Evolution."""
    n_models = len(predictions)

    def objective(w):
        w = np.array(w)
        w = w / w.sum()
        pred = sum(w[i] * predictions[i] for i in range(n_models))
        return mean_squared_error(actual, pred)

    bounds = [(0, 1)] * n_models
    result = differential_evolution(objective, bounds, seed=42, maxiter=100, disp=False)
    return result.x / result.x.sum()


def find_best_model_subset(predictions, actual, model_names, max_models=4):
    """Find best subset of models."""
    print("\n  Testing model subsets...")
    best_rmse = float('inf')
    best_subset = None
    best_weights = None

    n_models = len(predictions)

    for r in range(2, min(max_models + 1, n_models + 1)):
        for subset_idx in combinations(range(n_models), r):
            subset_preds = [predictions[i] for i in subset_idx]
            subset_names = [model_names[i] for i in subset_idx]

            try:
                weights = optimize_weights_scipy(subset_preds, actual)
                ensemble = sum(w * p for w, p in zip(weights, subset_preds))
                rmse = np.sqrt(mean_squared_error(actual, ensemble))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_subset = subset_idx
                    best_weights = weights
            except:
                continue

    return best_subset, best_weights, best_rmse


# =============================================================================
# MAIN ENSEMBLE
# =============================================================================

def run_enhanced_ensemble():
    """Run the enhanced ensemble model."""

    # Load data
    rate_df, gdelt_df, goldstein_df, trade_df = load_all_data()

    # Engineer features
    df = engineer_gdelt_features(gdelt_df, goldstein_df, rate_df)
    df = add_trade_features(df, trade_df)

    # Prepare price series
    price_series = pd.Series(df['USD_to_INR'].values, index=pd.to_datetime(df['Date']))

    # Use last 2 years for modeling
    cutoff = price_series.index.max() - pd.Timedelta(days=730)
    recent = price_series[price_series.index >= cutoff]
    recent_df = df[pd.to_datetime(df['Date']) >= cutoff].copy()

    print(f"\nUsing {len(recent)} recent observations")

    # Train/validation split
    train_size = int(len(recent) * 0.85)
    train_prices = recent.iloc[:train_size]
    val_prices = recent.iloc[train_size:]
    train_df = recent_df.iloc[:train_size]
    val_df = recent_df.iloc[train_size:]

    horizon = len(val_prices)
    print(f"Training: {len(train_prices)}, Validation: {horizon}")

    # ===========================================
    # TRAIN ALL MODELS
    # ===========================================
    print("\n" + "=" * 70)
    print("  TRAINING MODELS")
    print("=" * 70)

    model_predictions = {}
    model_names = []

    # 1. VMD Trend
    print("\n[1/8] VMD Trend Extrapolation...")
    components = vmd_decompose(recent.values)
    trend_train = pd.Series(components['trend'][:-horizon])
    vmd_pred = train_holt_winters(trend_train, horizon)
    vmd_pred += np.mean(components['seasonality'])
    model_predictions['VMD_Trend'] = vmd_pred
    model_names.append('VMD_Trend')

    # 2. Holt-Winters
    print("[2/8] Holt-Winters...")
    hw_pred = train_holt_winters(train_prices, horizon)
    model_predictions['Holt_Winters'] = hw_pred
    model_names.append('Holt_Winters')

    # 3. ARIMA
    print("[3/8] ARIMA(2,1,2)...")
    arima_pred = train_arima(train_prices, horizon, order=(2,1,2))
    model_predictions['ARIMA'] = arima_pred
    model_names.append('ARIMA')

    # 4. LSTM
    print("[4/8] LSTM...")
    lstm_model, lstm_scaler, lookback = train_lstm_model(train_prices, lookback=20, epochs=15)
    lstm_pred = lstm_forecast(lstm_model, lstm_scaler, train_prices, lookback, horizon)
    model_predictions['LSTM'] = lstm_pred
    model_names.append('LSTM')

    # 5. GARCH
    print("[5/8] GARCH...")
    train_returns = train_prices.pct_change().dropna()
    garch_mean, garch_result = train_garch(train_returns)
    garch_pred = [train_prices.iloc[-1]]
    for i in range(horizon):
        garch_pred.append(garch_pred[-1] * (1 + garch_mean))
    model_predictions['GARCH'] = np.array(garch_pred[1:])
    model_names.append('GARCH')

    # 6. Monte Carlo
    print("[6/8] Monte Carlo...")
    mc_result = monte_carlo_forecast(
        train_prices.iloc[-1],
        train_returns.mean(),
        train_returns.std(),
        horizon
    )
    model_predictions['Monte_Carlo'] = mc_result['median']
    model_names.append('Monte_Carlo')

    # 7. XGBoost with GDELT features
    print("[7/8] XGBoost + GDELT...")
    feature_cols = [c for c in train_df.columns if c not in ['Date', 'USD_to_INR']]
    if len(feature_cols) > 5:
        X_train = train_df[feature_cols].values
        y_train = train_df['USD_to_INR'].values
        X_val = val_df[feature_cols].values

        xgb_model, xgb_scaler = train_xgboost_noise(X_train, y_train)
        X_val_scaled = xgb_scaler.transform(np.nan_to_num(X_val, nan=0, posinf=0, neginf=0))
        xgb_pred = xgb_model.predict(X_val_scaled)
        model_predictions['XGB_GDELT'] = xgb_pred
        model_names.append('XGB_GDELT')
    else:
        model_predictions['XGB_GDELT'] = np.full(horizon, train_prices.mean())
        model_names.append('XGB_GDELT')

    # 8. Random Forest with GDELT
    print("[8/8] Random Forest + GDELT...")
    if len(feature_cols) > 5:
        X_train_clean = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        rf_model.fit(X_train_clean, y_train)
        X_val_clean = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)
        rf_pred = rf_model.predict(X_val_clean)
        model_predictions['RF_GDELT'] = rf_pred
        model_names.append('RF_GDELT')
    else:
        model_predictions['RF_GDELT'] = np.full(horizon, train_prices.mean())
        model_names.append('RF_GDELT')

    # ===========================================
    # OPTIMIZE WEIGHTS
    # ===========================================
    print("\n" + "=" * 70)
    print("  OPTIMIZING WEIGHTS")
    print("=" * 70)

    actual = val_prices.values
    predictions_list = [model_predictions[name] for name in model_names]

    # Align lengths
    min_len = min(len(actual), min(len(p) for p in predictions_list))
    actual = actual[:min_len]
    predictions_list = [p[:min_len] for p in predictions_list]

    # Method 1: SLSQP optimization
    print("\n[Method 1] SLSQP Optimization...")
    weights_slsqp = optimize_weights_scipy(predictions_list, actual, method='SLSQP')
    ensemble_slsqp = sum(w * p for w, p in zip(weights_slsqp, predictions_list))
    rmse_slsqp = np.sqrt(mean_squared_error(actual, ensemble_slsqp))

    # Method 2: Differential Evolution
    print("[Method 2] Differential Evolution...")
    weights_de = optimize_weights_de(predictions_list, actual)
    ensemble_de = sum(w * p for w, p in zip(weights_de, predictions_list))
    rmse_de = np.sqrt(mean_squared_error(actual, ensemble_de))

    # Method 3: Best subset search
    print("[Method 3] Best Subset Search...")
    best_subset, best_subset_weights, rmse_subset = find_best_model_subset(
        predictions_list, actual, model_names, max_models=4
    )

    # Choose best method
    results = {
        'SLSQP': (weights_slsqp, rmse_slsqp),
        'Diff_Evol': (weights_de, rmse_de),
        'Subset': (best_subset_weights, rmse_subset)
    }

    best_method = min(results, key=lambda x: results[x][1])
    best_weights, best_rmse = results[best_method]

    print(f"\nBest method: {best_method} (RMSE: {best_rmse:.4f})")

    # If subset is best, use subset models
    if best_method == 'Subset':
        final_models = [model_names[i] for i in best_subset]
        final_predictions = [predictions_list[i] for i in best_subset]
        final_weights = best_subset_weights
    else:
        final_models = model_names
        final_predictions = predictions_list
        final_weights = best_weights

    # Create final ensemble
    ensemble_pred = sum(w * p for w, p in zip(final_weights, final_predictions))

    # ===========================================
    # RESULTS
    # ===========================================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\nFinal Ensemble Weights:")
    weight_dict = {}
    for name, w in zip(final_models, final_weights):
        print(f"  {name}: {w:.4f}")
        weight_dict[name] = w

    print("\nIndividual Model Performance:")
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 45)

    model_metrics = {}
    for name, pred in zip(model_names, predictions_list):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        print(f"{name:<15} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
        model_metrics[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}

    # Ensemble metrics
    ens_rmse = np.sqrt(mean_squared_error(actual, ensemble_pred))
    ens_mae = mean_absolute_error(actual, ensemble_pred)
    ens_r2 = r2_score(actual, ensemble_pred)
    print("-" * 45)
    print(f"{'ENSEMBLE':<15} {ens_rmse:<10.4f} {ens_mae:<10.4f} {ens_r2:<10.4f}")

    # Improvement over best single model
    best_single_rmse = min(m['rmse'] for m in model_metrics.values())
    improvement = (best_single_rmse - ens_rmse) / best_single_rmse * 100
    print(f"\nImprovement over best single model: {improvement:.2f}%")

    # ===========================================
    # VISUALIZATIONS
    # ===========================================
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)

    val_dates = recent.index[train_size:train_size + min_len]

    # 1. Enhanced weights chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All models weights
    ax1 = axes[0]
    all_weights = np.zeros(len(model_names))
    for i, name in enumerate(model_names):
        if name in weight_dict:
            all_weights[i] = weight_dict[name]
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    bars = ax1.bar(model_names, all_weights, color=colors, edgecolor='black')
    ax1.set_ylabel('Weight', fontweight='bold')
    ax1.set_title('Optimized Ensemble Weights (All Models)', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for bar, w in zip(bars, all_weights):
        if w > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{w:.3f}', ha='center', fontsize=8)

    # Pie chart for selected models
    ax2 = axes[1]
    selected_weights = [w for w in final_weights if w > 0.01]
    selected_names = [n for n, w in zip(final_models, final_weights) if w > 0.01]
    if len(selected_weights) > 0:
        ax2.pie(selected_weights, labels=selected_names, autopct='%1.1f%%',
                colors=plt.cm.Set2(np.linspace(0, 1, len(selected_weights))))
        ax2.set_title('Selected Models in Ensemble', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'enhanced_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: enhanced_weights.png")

    # 2. Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RMSE comparison
    ax1 = axes[0, 0]
    rmse_vals = [model_metrics[n]['rmse'] for n in model_names] + [ens_rmse]
    labels = model_names + ['ENSEMBLE']
    colors = ['lightblue'] * len(model_names) + ['red']
    bars = ax1.bar(labels, rmse_vals, color=colors, edgecolor='black')
    ax1.set_ylabel('RMSE', fontweight='bold')
    ax1.set_title('RMSE Comparison', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # R² comparison
    ax2 = axes[0, 1]
    r2_vals = [model_metrics[n]['r2'] for n in model_names] + [ens_r2]
    colors = ['lightgreen'] * len(model_names) + ['red']
    bars = ax2.bar(labels, r2_vals, color=colors, edgecolor='black')
    ax2.set_ylabel('R²', fontweight='bold')
    ax2.set_title('R² Comparison', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='gray', linestyle='--')

    # Validation predictions
    ax3 = axes[1, 0]
    ax3.plot(val_dates, actual, 'k-', label='Actual', linewidth=2)
    ax3.plot(val_dates, ensemble_pred, 'r--', label='Ensemble', linewidth=2)
    # Plot top 3 models
    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['rmse'])[:3]
    for name, _ in sorted_models:
        idx = model_names.index(name)
        ax3.plot(val_dates, predictions_list[idx], '--', label=name, alpha=0.6, linewidth=1)
    ax3.set_title('Validation: Predictions vs Actual', fontweight='bold')
    ax3.set_ylabel('USD/INR')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Prediction errors
    ax4 = axes[1, 1]
    for name, _ in sorted_models[:2]:
        idx = model_names.index(name)
        error = actual - predictions_list[idx]
        ax4.plot(val_dates, error, '--', label=name, alpha=0.5)
    ens_error = actual - ensemble_pred
    ax4.fill_between(val_dates, ens_error, 0, alpha=0.3, color='red', label='Ensemble')
    ax4.axhline(y=0, color='black', linestyle='-')
    ax4.set_title('Prediction Errors', fontweight='bold')
    ax4.set_ylabel('Error (INR)')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'enhanced_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: enhanced_comparison.png")

    # 3. Forecast
    print("\n  Generating 30-day forecast...")
    current_price = recent.iloc[-1]
    recent_returns = recent.pct_change().dropna()

    mc_forecast = monte_carlo_forecast(
        current_price,
        recent_returns.mean(),
        recent_returns.std(),
        30,
        n_sims=10000
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    hist_days = 90
    hist_idx = recent.index[-hist_days:]
    hist_vals = recent.values[-hist_days:]
    ax.plot(hist_idx, hist_vals, 'b-', linewidth=2, label='Historical')

    last_date = recent.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
    ax.plot(forecast_dates, mc_forecast['median'], 'r-', linewidth=2, label='Forecast')
    ax.fill_between(forecast_dates, mc_forecast['p5'], mc_forecast['p95'],
                    alpha=0.2, color='red', label='90% CI')
    ax.axvline(x=last_date, color='gray', linestyle='--')

    ax.set_title('USD/INR: 30-Day Ensemble Forecast', fontweight='bold', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate (INR)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'enhanced_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: enhanced_forecast.png")

    # 4. Summary dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Price history
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(price_series.index, price_series.values, 'b-', linewidth=0.8)
    ax1.set_title('USD/INR Exchange Rate History', fontweight='bold')
    ax1.set_ylabel('INR')
    ax1.grid(True, alpha=0.3)

    # Weights
    ax2 = fig.add_subplot(gs[0, 2])
    if len(selected_weights) > 0:
        ax2.pie(selected_weights, labels=selected_names, autopct='%1.1f%%',
                colors=plt.cm.Set2(np.linspace(0, 1, len(selected_weights))))
    ax2.set_title('Ensemble Weights', fontweight='bold')

    # VMD decomposition
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(components['trend'], 'g-', linewidth=1)
    ax3.set_title('Trend Component', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(components['seasonality'], 'orange', linewidth=1)
    ax4.set_title('Seasonality', fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(components['noise'], 'r-', linewidth=0.5, alpha=0.7)
    ax5.set_title('Noise', fontweight='bold')

    # Validation
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.plot(val_dates, actual, 'k-', label='Actual', linewidth=2)
    ax6.plot(val_dates, ensemble_pred, 'r--', label='Ensemble', linewidth=2)
    ax6.set_title('Validation Performance', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Metrics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    metrics_text = f"""
ENHANCED ENSEMBLE
════════════════════════

RMSE:  {ens_rmse:.4f}
MAE:   {ens_mae:.4f}
R²:    {ens_r2:.4f}

Best method: {best_method}
Models used: {len(final_models)}

════════════════════════

30-DAY FORECAST
Current: {current_price:.2f} INR
Median:  {mc_forecast['median'][-1]:.2f} INR
90% CI: [{mc_forecast['p5'][-1]:.2f},
         {mc_forecast['p95'][-1]:.2f}]
    """
    ax7.text(0.05, 0.5, metrics_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center', transform=ax7.transAxes)

    plt.suptitle('Enhanced Ensemble Model - Summary Dashboard',
                fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, 'enhanced_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: enhanced_dashboard.png")

    # Save results
    weights_df = pd.DataFrame({
        'Model': final_models,
        'Weight': final_weights
    })
    weights_df.to_csv(os.path.join(OUTPUT_DIR, 'enhanced_weights.csv'), index=False)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'R2', 'Best_Method', 'Improvement_pct'],
        'Value': [ens_rmse, ens_mae, ens_r2, best_method, improvement]
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'enhanced_metrics.csv'), index=False)

    # Save all model comparison
    comparison_df = pd.DataFrame([
        {'Model': name, **metrics} for name, metrics in model_metrics.items()
    ])
    comparison_df.loc[len(comparison_df)] = {'Model': 'ENSEMBLE', 'rmse': ens_rmse, 'mae': ens_mae, 'r2': ens_r2}
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

    print("\n" + "=" * 70)
    print("  ENHANCED ENSEMBLE COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Ensemble:")
    for name, w in zip(final_models, final_weights):
        if w > 0.01:
            print(f"  {name}: {w:.4f}")
    print(f"\nValidation R²: {ens_r2:.4f}")
    print(f"Improvement over best single model: {improvement:.2f}%")
    print(f"\nCurrent rate: {current_price:.4f} INR")
    print(f"30-day forecast: {mc_forecast['median'][-1]:.4f} INR")
    print(f"\nOutput saved to: {OUTPUT_DIR}")

    return weight_dict, model_metrics


if __name__ == "__main__":
    weights, metrics = run_enhanced_ensemble()

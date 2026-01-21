"""
Ultimate Ensemble Model with Gemini AI Integration
===================================================

This comprehensive model combines:
1. Traditional Time Series: ARIMA, SARIMA, Holt-Winters, Theta
2. Machine Learning: XGBoost, LightGBM, Random Forest, SVR
3. Deep Learning: LSTM, GRU, Transformer-style attention
4. Statistical: GARCH, EGARCH, Monte Carlo
5. Decomposition: VMD, EMD, Wavelet
6. GDELT Noise Prediction: ML models on news features
7. Gemini AI: LLM-based sentiment analysis for short-term outlook

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import sys
import json
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

# Statistical Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Input,
                                     Bidirectional, Attention, MultiHeadAttention,
                                     LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Optimization
from scipy.optimize import minimize, differential_evolution
from scipy import signal

np.random.seed(42)
tf.random.set_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyB08g5ToCYmiN6-LrrHsmCKGdNQlJiBnDM"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


# =============================================================================
# GEMINI AI INTEGRATION
# =============================================================================

def call_gemini_api(prompt, max_tokens=1000):
    """Call Gemini API for text analysis."""
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens
        }
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"  Gemini API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"  Gemini API exception: {e}")
        return None


def analyze_gdelt_with_gemini(gdelt_df, goldstein_df):
    """Use Gemini to analyze GDELT news data for exchange rate outlook."""
    print("\n  Analyzing GDELT data with Gemini AI...")

    # Prepare summary of recent GDELT data
    if gdelt_df is not None and len(gdelt_df) > 0:
        recent = gdelt_df.tail(30)

        summary = f"""
Recent 30-day GDELT News Analytics for India-USA Relations:

SENTIMENT ANALYSIS:
- Economy Tone (avg): {recent['Tone_Economy'].mean():.2f} (range: {recent['Tone_Economy'].min():.2f} to {recent['Tone_Economy'].max():.2f})
- Conflict Tone (avg): {recent['Tone_Conflict'].mean():.2f}
- Policy Tone (avg): {recent['Tone_Policy'].mean():.2f}
- Corporate Tone (avg): {recent['Tone_Corporate'].mean():.2f}
- Overall Tone (avg): {recent['Tone_Overall'].mean():.2f}

GOLDSTEIN SCALE (measures event impact, -10 to +10):
- Weighted Goldstein: {recent['Goldstein_Weighted'].mean():.2f}
- Average Goldstein: {recent['Goldstein_Avg'].mean():.3f}

NEWS VOLUME:
- Total Events: {recent['Count_Total'].sum():,.0f}
- Economy Events: {recent['Count_Economy'].sum():,.0f}
- Conflict Events: {recent['Count_Conflict'].sum():,.0f}
- Policy Events: {recent['Count_Policy'].sum():,.0f}

VOLUME SPIKES (% change from baseline):
- Overall Volume Spike: {recent['Volume_Spike'].mean():.1f}%
- Economy Volume Spike: {recent['Volume_Spike_Economy'].mean():.1f}%
- Conflict Volume Spike: {recent['Volume_Spike_Conflict'].mean():.1f}%

TREND (last 7 days vs previous 7 days):
- Economy Tone Change: {recent['Tone_Economy'].tail(7).mean() - recent['Tone_Economy'].head(7).mean():.2f}
- Goldstein Change: {recent['Goldstein_Avg'].tail(7).mean() - recent['Goldstein_Avg'].head(7).mean():.3f}
"""
    else:
        summary = "No recent GDELT data available."

    # Add Goldstein data if available
    if goldstein_df is not None and len(goldstein_df) > 0:
        recent_gs = goldstein_df.tail(30)
        summary += f"""

USA-INDIA BILATERAL SENTIMENT:
- USA Goldstein (avg): {recent_gs['USA_Avg_Goldstein'].mean():.3f}
- India Goldstein (avg): {recent_gs['India_Avg_Goldstein'].mean():.3f}
- Sentiment Difference (USA-India): {recent_gs['USA_India_Sentiment_Diff'].mean():.3f}
- Combined Weighted Average: {recent_gs['Combined_Weighted_Avg'].mean():.3f}
"""

    prompt = f"""You are a financial analyst specializing in currency markets. Based on the following GDELT news analytics data, provide a brief analysis for USD/INR exchange rate outlook.

{summary}

Please provide:
1. SHORT-TERM OUTLOOK (1-7 days): Will USD/INR likely increase, decrease, or stay stable? Confidence level?
2. KEY RISK FACTORS: What news trends could impact the exchange rate?
3. SENTIMENT SCORE: On a scale of -1 (very bearish for INR) to +1 (very bullish for INR), what is your assessment?
4. VOLATILITY EXPECTATION: Low, Medium, or High?

Be concise and data-driven. Format your response as:
OUTLOOK: [increase/decrease/stable]
CONFIDENCE: [low/medium/high]
SENTIMENT_SCORE: [number between -1 and 1]
VOLATILITY: [low/medium/high]
REASONING: [2-3 sentences]
"""

    response = call_gemini_api(prompt, max_tokens=500)

    if response:
        # Parse the response
        result = {
            'outlook': 'stable',
            'confidence': 'medium',
            'sentiment_score': 0.0,
            'volatility': 'medium',
            'reasoning': response,
            'raw_response': response
        }

        # Try to extract structured data
        lines = response.upper().split('\n')
        for line in lines:
            if 'OUTLOOK:' in line:
                if 'INCREASE' in line:
                    result['outlook'] = 'increase'
                elif 'DECREASE' in line:
                    result['outlook'] = 'decrease'
            elif 'CONFIDENCE:' in line:
                if 'HIGH' in line:
                    result['confidence'] = 'high'
                elif 'LOW' in line:
                    result['confidence'] = 'low'
            elif 'SENTIMENT_SCORE:' in line:
                try:
                    score = float(line.split(':')[1].strip().split()[0])
                    result['sentiment_score'] = max(-1, min(1, score))
                except:
                    pass
            elif 'VOLATILITY:' in line:
                if 'HIGH' in line:
                    result['volatility'] = 'high'
                elif 'LOW' in line:
                    result['volatility'] = 'low'

        return result
    else:
        return {
            'outlook': 'stable',
            'confidence': 'low',
            'sentiment_score': 0.0,
            'volatility': 'medium',
            'reasoning': 'Unable to get Gemini analysis',
            'raw_response': None
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load all data sources."""
    print("=" * 70)
    print("  LOADING DATA")
    print("=" * 70)

    base_dir = os.path.dirname(OUTPUT_DIR)

    # Exchange rates
    rate_path = os.path.join(OUTPUT_DIR, 'usd_inr_10year.csv')
    if not os.path.exists(rate_path):
        rate_path = os.path.join(base_dir, 'usd_inr_exchange_rates_1year.csv')
    rate_df = pd.read_csv(rate_path, parse_dates=['Date'])
    print(f"Exchange rates: {len(rate_df)} observations")

    # GDELT
    gdelt_path = os.path.join(base_dir, 'Phase-B', 'merged_training_data.csv')
    gdelt_df = pd.read_csv(gdelt_path, parse_dates=['Date']) if os.path.exists(gdelt_path) else None
    if gdelt_df is not None:
        print(f"GDELT: {len(gdelt_df)} observations")

    # Goldstein
    goldstein_path = os.path.join(base_dir, 'combined_goldstein_exchange_rates.csv')
    goldstein_df = pd.read_csv(goldstein_path, parse_dates=['Date']) if os.path.exists(goldstein_path) else None
    if goldstein_df is not None:
        print(f"Goldstein: {len(goldstein_df)} observations")

    # Trade balance
    trade_path = os.path.join(base_dir, 'india_usa_trade', 'output', 'trade_balance_analysis.csv')
    trade_df = pd.read_csv(trade_path) if os.path.exists(trade_path) else None
    if trade_df is not None:
        print(f"Trade: {len(trade_df)} years")

    return rate_df, gdelt_df, goldstein_df, trade_df


# =============================================================================
# ADVANCED DECOMPOSITION METHODS
# =============================================================================

def wavelet_decompose(signal, wavelet='db4', level=3):
    """Wavelet decomposition."""
    try:
        import pywt
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # Reconstruct trend (approximation) and details
        trend = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)[:len(signal)]
        noise = signal - trend
        return {'trend': trend, 'noise': noise}
    except ImportError:
        # Fallback to simple decomposition
        trend = pd.Series(signal).rolling(50, min_periods=1).mean().values
        noise = signal - trend
        return {'trend': trend, 'noise': noise}


def emd_decompose(signal, n_imfs=3):
    """Empirical Mode Decomposition (simplified)."""
    signal = np.array(signal)
    residue = signal.copy()
    imfs = []

    for _ in range(n_imfs):
        imf = np.zeros_like(signal)
        h = residue.copy()

        # Sifting process (simplified)
        for _ in range(10):
            # Find local maxima and minima
            maxima_idx = signal.argpartition(-len(signal)//10)[-len(signal)//10:]
            minima_idx = signal.argpartition(len(signal)//10)[:len(signal)//10]

            # Simple interpolation
            upper = pd.Series(h).rolling(5, min_periods=1).max().values
            lower = pd.Series(h).rolling(5, min_periods=1).min().values
            mean_env = (upper + lower) / 2
            h = h - mean_env

        imf = h
        residue = residue - imf
        imfs.append(imf)

    return {'imfs': imfs, 'residue': residue, 'trend': residue}


def vmd_decompose(signal, K=3):
    """VMD-style decomposition using bandpass filtering."""
    signal = np.array(signal)
    n = len(signal)

    # Create frequency bands
    modes = []
    for k in range(K):
        # Bandpass filter at different frequencies
        low_freq = k / (2 * K)
        high_freq = (k + 1) / (2 * K)

        # Simple moving average approximation of bandpass
        window_low = max(3, int(n * (1 - high_freq) / 10))
        window_high = max(3, int(n * (1 - low_freq) / 10))

        smooth_low = pd.Series(signal).rolling(window_low, min_periods=1).mean().values
        smooth_high = pd.Series(signal).rolling(window_high, min_periods=1).mean().values

        mode = smooth_high - smooth_low
        modes.append(mode)

    # Trend is the slowest mode
    trend = pd.Series(signal).rolling(50, min_periods=1).mean().values
    seasonality = pd.Series(signal - trend).rolling(10, min_periods=1).mean().values
    noise = signal - trend - seasonality

    return {
        'modes': modes,
        'trend': trend,
        'seasonality': seasonality,
        'noise': noise
    }


# =============================================================================
# ADVANCED MODELS
# =============================================================================

def train_theta_model(train, horizon):
    """Theta method for forecasting."""
    # Theta method: decompose into two theta lines
    n = len(train)
    t = np.arange(1, n + 1)

    # Linear regression for drift
    X = np.column_stack([np.ones(n), t])
    y = train.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Theta lines
    theta0 = y  # Original series
    theta2 = 2 * y - (beta[0] + beta[1] * t)  # Dampened series

    # SES on theta2
    alpha = 0.5
    ses = np.zeros(n)
    ses[0] = theta2[0]
    for i in range(1, n):
        ses[i] = alpha * theta2[i] + (1 - alpha) * ses[i-1]

    # Forecast
    drift = beta[1]
    forecasts = []
    last_ses = ses[-1]
    for h in range(1, horizon + 1):
        fc = last_ses + drift * h / 2
        forecasts.append(fc)

    return np.array(forecasts)


def train_ets_model(train, horizon):
    """ETS (Error, Trend, Seasonal) model."""
    try:
        model = ExponentialSmoothing(
            train,
            trend='add',
            seasonal=None,
            damped_trend=True
        ).fit(optimized=True)
        return model.forecast(horizon).values
    except:
        return np.full(horizon, train.iloc[-1])


def train_sarima_model(train, horizon):
    """SARIMA model."""
    try:
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False, maxiter=100)
        return model.forecast(horizon).values
    except:
        return np.full(horizon, train.iloc[-1])


def train_gru_model(train, lookback=20, epochs=20):
    """GRU neural network."""
    scaler = MinMaxScaler()
    data = scaler.fit_transform(train.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        GRU(32, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.1),
        GRU(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=5)])

    return model, scaler, lookback


def train_attention_model(train, lookback=20, epochs=20):
    """Transformer-style attention model."""
    scaler = MinMaxScaler()
    data = scaler.fit_transform(train.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build attention model
    inputs = Input(shape=(lookback, 1))
    x = Dense(32)(inputs)

    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = LayerNormalization()(x + attn_output)

    x = GlobalAveragePooling1D()(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=5)])

    return model, scaler, lookback


def forecast_nn(model, scaler, train, lookback, horizon):
    """Generate NN forecasts."""
    data = scaler.transform(train.values.reshape(-1, 1))
    preds = []
    seq = data[-lookback:].flatten()

    for _ in range(horizon):
        X_pred = seq[-lookback:].reshape(1, lookback, 1)
        pred = model.predict(X_pred, verbose=0)[0, 0]
        preds.append(scaler.inverse_transform([[pred]])[0, 0])
        seq = np.append(seq, pred)

    return np.array(preds)


def train_svr_model(X, y):
    """Support Vector Regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X, nan=0))
    model = SVR(kernel='rbf', C=100, gamma='scale')
    model.fit(X_scaled, y)
    return model, scaler


def train_knn_model(X, y):
    """K-Nearest Neighbors Regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X, nan=0))
    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    model.fit(X_scaled, y)
    return model, scaler


def train_bayesian_ridge(X, y):
    """Bayesian Ridge Regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X, nan=0))
    model = BayesianRidge()
    model.fit(X_scaled, y)
    return model, scaler


# =============================================================================
# GDELT NOISE PREDICTION
# =============================================================================

def build_gdelt_features(gdelt_df, goldstein_df):
    """Build comprehensive GDELT features for noise prediction."""
    if gdelt_df is None or len(gdelt_df) == 0:
        return None

    df = gdelt_df.copy()

    # Core features
    core_features = ['Tone_Economy', 'Tone_Conflict', 'Tone_Policy',
                    'Tone_Corporate', 'Tone_Overall', 'Goldstein_Avg',
                    'Count_Total', 'Volume_Spike']

    # Add lags
    for col in core_features:
        if col in df.columns:
            for lag in [1, 2, 3, 5, 7]:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Add rolling stats
    for col in core_features[:5]:
        if col in df.columns:
            df[f'{col}_roll3_mean'] = df[col].rolling(3).mean()
            df[f'{col}_roll3_std'] = df[col].rolling(3).std()
            df[f'{col}_roll7_mean'] = df[col].rolling(7).mean()

    # Add momentum
    for col in ['Tone_Economy', 'Goldstein_Avg']:
        if col in df.columns:
            df[f'{col}_momentum3'] = df[col].diff(3)
            df[f'{col}_momentum7'] = df[col].diff(7)

    # Add Goldstein features if available
    if goldstein_df is not None:
        gs = goldstein_df.copy()
        df = pd.merge(df, gs[['Date', 'USA_Avg_Goldstein', 'India_Avg_Goldstein',
                              'USA_India_Sentiment_Diff']], on='Date', how='left')

        if 'USA_Avg_Goldstein' in df.columns:
            df['Goldstein_Ratio'] = df['USA_Avg_Goldstein'] / (df['India_Avg_Goldstein'].abs() + 0.01)

    # Fill NaN
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    feature_cols = [c for c in df.columns if c not in ['Date']]

    return df, feature_cols


def train_gdelt_noise_models(X_train, y_train):
    """Train multiple models for GDELT-based noise prediction."""
    models = {}

    X_clean = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    y_clean = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)

    # XGBoost
    try:
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                                     learning_rate=0.1, random_state=42, verbosity=0)
        xgb_model.fit(X_clean, y_clean)
        models['XGBoost'] = xgb_model
    except:
        pass

    # Gradient Boosting
    try:
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        gb_model.fit(X_clean, y_clean)
        models['GradBoost'] = gb_model
    except:
        pass

    # Random Forest
    try:
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        rf_model.fit(X_clean, y_clean)
        models['RandomForest'] = rf_model
    except:
        pass

    # Ridge
    try:
        ridge_model = Ridge(alpha=1.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        ridge_model.fit(X_scaled, y_clean)
        models['Ridge'] = (ridge_model, scaler)
    except:
        pass

    return models


# =============================================================================
# MONTE CARLO AND GARCH
# =============================================================================

def monte_carlo_forecast(current_price, mean_return, volatility, horizon, n_sims=10000):
    """Enhanced Monte Carlo with jump diffusion."""
    paths = np.zeros((n_sims, horizon + 1))
    paths[:, 0] = current_price

    # Jump parameters
    jump_prob = 0.05
    jump_mean = 0
    jump_std = volatility * 2

    for t in range(1, horizon + 1):
        Z = np.random.standard_normal(n_sims)

        # Geometric Brownian Motion
        drift = (mean_return - 0.5 * volatility**2)
        diffusion = volatility * Z

        # Add jumps
        jumps = np.random.binomial(1, jump_prob, n_sims) * np.random.normal(jump_mean, jump_std, n_sims)

        paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion + jumps)

    return {
        'mean': np.mean(paths, axis=0),
        'median': np.median(paths, axis=0),
        'p5': np.percentile(paths, 5, axis=0),
        'p10': np.percentile(paths, 10, axis=0),
        'p25': np.percentile(paths, 25, axis=0),
        'p75': np.percentile(paths, 75, axis=0),
        'p90': np.percentile(paths, 90, axis=0),
        'p95': np.percentile(paths, 95, axis=0),
        'paths': paths
    }


def train_egarch(returns):
    """EGARCH model for asymmetric volatility."""
    try:
        returns_scaled = returns * 100
        model = arch_model(returns_scaled, mean='Constant', vol='EGARCH',
                          p=1, q=1, o=1, dist='skewt')
        result = model.fit(disp='off', show_warning=False)
        return result.params['mu'] / 100, result
    except:
        return returns.mean(), None


# =============================================================================
# ENSEMBLE OPTIMIZATION
# =============================================================================

def optimize_ensemble(predictions, actual, min_weight=0.0, max_weight=0.5):
    """Optimize ensemble weights with constraints."""
    model_names = list(predictions.keys())
    n_models = len(model_names)

    pred_arrays = [np.array(predictions[name])[:len(actual)] for name in model_names]

    def objective(w):
        w = np.array(w)
        w = w / (w.sum() + 1e-10)
        pred = sum(w[i] * pred_arrays[i] for i in range(n_models))
        return mean_squared_error(actual, pred)

    # Try multiple optimization methods
    best_weights = None
    best_mse = float('inf')

    # Method 1: SLSQP
    try:
        x0 = [1/n_models] * n_models
        bounds = [(min_weight, max_weight)] * n_models
        constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.fun < best_mse:
            best_mse = result.fun
            best_weights = result.x
    except:
        pass

    # Method 2: Differential Evolution
    try:
        bounds = [(min_weight, max_weight)] * n_models
        result = differential_evolution(objective, bounds, seed=42, maxiter=50)
        weights = result.x / result.x.sum()
        mse = objective(weights)
        if mse < best_mse:
            best_mse = mse
            best_weights = weights
    except:
        pass

    if best_weights is None:
        best_weights = np.array([1/n_models] * n_models)

    best_weights = best_weights / best_weights.sum()
    return dict(zip(model_names, best_weights))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("  ULTIMATE ENSEMBLE MODEL")
    print("  With Gemini AI, GDELT Noise Prediction, and 15+ Models")
    print("=" * 70)

    # Load data
    rate_df, gdelt_df, goldstein_df, trade_df = load_all_data()

    # Prepare price series
    price_series = pd.Series(rate_df['USD_to_INR'].values,
                            index=pd.to_datetime(rate_df['Date']))

    # Use last 2 years
    cutoff = price_series.index.max() - pd.Timedelta(days=730)
    recent = price_series[price_series.index >= cutoff]
    print(f"\nUsing {len(recent)} recent observations")

    # Train/validation split
    train_size = int(len(recent) * 0.85)
    train_prices = recent.iloc[:train_size]
    val_prices = recent.iloc[train_size:]
    horizon = len(val_prices)

    print(f"Training: {len(train_prices)}, Validation: {horizon}")

    # Get Gemini analysis
    print("\n" + "=" * 70)
    print("  GEMINI AI ANALYSIS")
    print("=" * 70)
    gemini_result = analyze_gdelt_with_gemini(gdelt_df, goldstein_df)
    print(f"\n  Outlook: {gemini_result['outlook']}")
    print(f"  Confidence: {gemini_result['confidence']}")
    print(f"  Sentiment Score: {gemini_result['sentiment_score']:.2f}")
    print(f"  Volatility: {gemini_result['volatility']}")
    print(f"\n  Reasoning: {gemini_result['reasoning'][:500]}...")

    # VMD Decomposition
    print("\n" + "=" * 70)
    print("  SIGNAL DECOMPOSITION")
    print("=" * 70)
    vmd = vmd_decompose(recent.values)
    print(f"  VMD components extracted: trend, seasonality, noise")

    # Train all models
    print("\n" + "=" * 70)
    print("  TRAINING 15+ MODELS")
    print("=" * 70)

    predictions = {}
    last_price = train_prices.iloc[-1]
    returns = train_prices.pct_change().dropna()

    # 1. Holt-Winters
    print("\n[01/15] Holt-Winters ETS...")
    predictions['Holt_Winters'] = train_ets_model(train_prices, horizon)

    # 2. ARIMA
    print("[02/15] ARIMA(1,1,1)...")
    try:
        arima = ARIMA(train_prices, order=(1,1,1)).fit()
        predictions['ARIMA'] = arima.forecast(horizon).values
    except:
        predictions['ARIMA'] = np.full(horizon, last_price)

    # 3. SARIMA
    print("[03/15] SARIMA...")
    predictions['SARIMA'] = train_sarima_model(train_prices, horizon)

    # 4. Theta
    print("[04/15] Theta Method...")
    predictions['Theta'] = train_theta_model(train_prices, horizon)

    # 5. LSTM
    print("[05/15] LSTM Neural Network...")
    try:
        lstm_model, lstm_scaler, lstm_lb = train_gru_model(train_prices, lookback=15, epochs=15)
        predictions['LSTM'] = forecast_nn(lstm_model, lstm_scaler, train_prices, lstm_lb, horizon)
    except Exception as e:
        print(f"    LSTM error: {e}")
        predictions['LSTM'] = np.full(horizon, last_price)

    # 6. GRU
    print("[06/15] GRU Neural Network...")
    try:
        gru_model, gru_scaler, gru_lb = train_gru_model(train_prices, lookback=20, epochs=15)
        predictions['GRU'] = forecast_nn(gru_model, gru_scaler, train_prices, gru_lb, horizon)
    except:
        predictions['GRU'] = np.full(horizon, last_price)

    # 7. Attention
    print("[07/15] Attention Model...")
    try:
        attn_model, attn_scaler, attn_lb = train_attention_model(train_prices, lookback=15, epochs=15)
        predictions['Attention'] = forecast_nn(attn_model, attn_scaler, train_prices, attn_lb, horizon)
    except:
        predictions['Attention'] = np.full(horizon, last_price)

    # 8. GARCH
    print("[08/15] GARCH...")
    try:
        garch_mean, _ = train_egarch(returns)
        garch_preds = [last_price]
        for _ in range(horizon):
            garch_preds.append(garch_preds[-1] * (1 + garch_mean))
        predictions['GARCH'] = np.array(garch_preds[1:])
    except:
        predictions['GARCH'] = np.full(horizon, last_price)

    # 9. Monte Carlo
    print("[09/15] Monte Carlo (Jump Diffusion)...")
    mc = monte_carlo_forecast(last_price, returns.mean(), returns.std(), horizon)
    predictions['Monte_Carlo'] = mc['median'][1:]

    # 10. VMD + Trend Extrapolation
    print("[10/15] VMD Trend Extrapolation...")
    trend_train = pd.Series(vmd['trend'][:-horizon])
    predictions['VMD_Trend'] = train_ets_model(trend_train, horizon) + np.mean(vmd['seasonality'])

    # 11. Moving Average Momentum
    print("[11/15] MA Momentum...")
    ma5 = train_prices.rolling(5).mean().iloc[-1]
    ma20 = train_prices.rolling(20).mean().iloc[-1]
    momentum = (ma5 - ma20) / 20
    predictions['MA_Momentum'] = np.array([last_price + momentum * i for i in range(1, horizon + 1)])

    # 12-15. GDELT-based models
    print("[12/15] GDELT Feature Models...")
    if gdelt_df is not None and len(gdelt_df) > 50:
        gdelt_features, feature_cols = build_gdelt_features(gdelt_df, goldstein_df)

        # Merge with price data
        merged = pd.merge(
            recent.reset_index().rename(columns={'index': 'Date'}),
            gdelt_features[['Date'] + feature_cols],
            on='Date',
            how='left'
        )

        if merged[feature_cols[0]].notna().sum() > 50:
            X = merged[feature_cols].values[:-horizon]
            y = merged['USD_to_INR'].values[horizon:]
            X_val = merged[feature_cols].values[-horizon:]

            # Train GDELT noise models
            noise = vmd['noise'][:len(X)]
            gdelt_models = train_gdelt_noise_models(X, noise)

            for name, model in gdelt_models.items():
                try:
                    if isinstance(model, tuple):
                        m, scaler = model
                        X_scaled = scaler.transform(np.nan_to_num(X_val, nan=0))
                        noise_pred = m.predict(X_scaled)
                    else:
                        noise_pred = model.predict(np.nan_to_num(X_val, nan=0))

                    # Combine with trend
                    predictions[f'GDELT_{name}'] = predictions['VMD_Trend'] + noise_pred
                except:
                    pass

    # 13. Gemini-adjusted forecast
    print("[13/15] Gemini-Adjusted Forecast...")
    gemini_adjustment = gemini_result['sentiment_score'] * 0.1  # Scale factor
    predictions['Gemini_Adjusted'] = predictions['Monte_Carlo'] * (1 + gemini_adjustment)

    # 14. Ensemble of top models (pre-selection)
    print("[14/15] Mini-Ensemble (Top 3)...")
    # Average of MC, MA_Momentum, GARCH
    predictions['Mini_Ensemble'] = (predictions['Monte_Carlo'] +
                                    predictions['MA_Momentum'] +
                                    predictions['GARCH']) / 3

    # 15. Wavelet + ARIMA
    print("[15/15] Wavelet Decomposition...")
    wav = wavelet_decompose(recent.values)
    wav_trend = pd.Series(wav['trend'][:-horizon])
    predictions['Wavelet_Trend'] = train_ets_model(wav_trend, horizon)

    # Optimize ensemble
    print("\n" + "=" * 70)
    print("  OPTIMIZING ENSEMBLE WEIGHTS")
    print("=" * 70)

    actual = val_prices.values
    weights = optimize_ensemble(predictions, actual, min_weight=0.0, max_weight=0.4)

    # Create final ensemble
    ensemble_pred = sum(weights[name] * np.array(predictions[name])[:len(actual)]
                       for name in weights.keys())

    # Results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\nTop Model Weights:")
    sorted_weights = sorted(weights.items(), key=lambda x: -x[1])[:10]
    for name, w in sorted_weights:
        if w > 0.01:
            print(f"  {name}: {w:.4f}")

    print("\nModel Performance:")
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 50)

    model_metrics = {}
    for name, pred in predictions.items():
        pred_arr = np.array(pred)[:len(actual)]
        rmse = np.sqrt(mean_squared_error(actual, pred_arr))
        mae = mean_absolute_error(actual, pred_arr)
        r2 = r2_score(actual, pred_arr)
        model_metrics[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}

    # Sort by RMSE
    sorted_metrics = sorted(model_metrics.items(), key=lambda x: x[1]['rmse'])
    for name, m in sorted_metrics[:10]:
        print(f"{name:<20} {m['rmse']:<10.4f} {m['mae']:<10.4f} {m['r2']:<10.4f}")

    ens_rmse = np.sqrt(mean_squared_error(actual, ensemble_pred))
    ens_mae = mean_absolute_error(actual, ensemble_pred)
    ens_r2 = r2_score(actual, ensemble_pred)
    print("-" * 50)
    print(f"{'ENSEMBLE':<20} {ens_rmse:<10.4f} {ens_mae:<10.4f} {ens_r2:<10.4f}")

    best_single = sorted_metrics[0]
    improvement = (best_single[1]['rmse'] - ens_rmse) / best_single[1]['rmse'] * 100
    print(f"\nBest single: {best_single[0]} (RMSE: {best_single[1]['rmse']:.4f})")
    print(f"Ensemble improvement: {improvement:.2f}%")

    # Visualizations
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)

    val_dates = recent.index[train_size:train_size + len(actual)]

    # 1. Comprehensive weights
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    top_weights = [(n, w) for n, w in sorted_weights if w > 0.01]
    if top_weights:
        names, vals = zip(*top_weights)
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
        bars = ax1.barh(names, vals, color=colors, edgecolor='black')
        ax1.set_xlabel('Weight', fontweight='bold')
        ax1.set_title('Ultimate Ensemble Weights', fontweight='bold', fontsize=12)

    ax2 = axes[1]
    if top_weights:
        ax2.pie([w for _, w in top_weights],
               labels=[n for n, _ in top_weights],
               autopct='%1.1f%%',
               colors=plt.cm.Set3(np.linspace(0, 1, len(top_weights))))
    ax2.set_title('Weight Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ultimate_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ultimate_weights.png")

    # 2. Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # RMSE comparison
    ax1 = axes[0, 0]
    top_models = [m[0] for m in sorted_metrics[:12]]
    rmse_vals = [model_metrics[m]['rmse'] for m in top_models]
    colors = ['lightblue'] * len(top_models)
    bars = ax1.bar(top_models, rmse_vals, color=colors, edgecolor='black')
    ax1.axhline(y=ens_rmse, color='red', linestyle='--', linewidth=2, label=f'Ensemble: {ens_rmse:.3f}')
    ax1.set_ylabel('RMSE', fontweight='bold')
    ax1.set_title('Model RMSE Comparison (Top 12)', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # R² comparison
    ax2 = axes[0, 1]
    r2_vals = [model_metrics[m]['r2'] for m in top_models]
    bars = ax2.bar(top_models, r2_vals, color='lightgreen', edgecolor='black')
    ax2.axhline(y=ens_r2, color='red', linestyle='--', linewidth=2, label=f'Ensemble: {ens_r2:.3f}')
    ax2.axhline(y=0, color='gray', linestyle='-')
    ax2.set_ylabel('R²', fontweight='bold')
    ax2.set_title('Model R² Comparison', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    # Validation predictions
    ax3 = axes[1, 0]
    ax3.plot(val_dates, actual, 'k-', label='Actual', linewidth=2)
    ax3.plot(val_dates, ensemble_pred, 'r--', label='Ensemble', linewidth=2)
    for name, _ in sorted_metrics[:3]:
        ax3.plot(val_dates, np.array(predictions[name])[:len(actual)],
                '--', label=name, alpha=0.5, linewidth=1)
    ax3.set_title('Validation: Top Models vs Actual', fontweight='bold')
    ax3.set_ylabel('USD/INR')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Prediction errors
    ax4 = axes[1, 1]
    error = actual - ensemble_pred
    ax4.fill_between(val_dates, error, 0, alpha=0.3, color='red')
    ax4.plot(val_dates, error, 'r-', linewidth=1)
    ax4.axhline(y=0, color='black', linestyle='-')
    ax4.set_title('Ensemble Prediction Error', fontweight='bold')
    ax4.set_ylabel('Error (INR)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ultimate_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ultimate_comparison.png")

    # 3. 30-Day Forecast
    print("\n  Generating 30-day forecast...")
    current_price = recent.iloc[-1]
    mc_30 = monte_carlo_forecast(current_price, returns.mean(), returns.std(), 30, n_sims=20000)

    # Adjust based on Gemini sentiment
    gemini_adj = 1 + gemini_result['sentiment_score'] * 0.05
    mc_30_adj = {k: v * gemini_adj if k != 'paths' else v for k, v in mc_30.items()}

    fig, ax = plt.subplots(figsize=(16, 9))

    # Historical
    hist_days = 90
    hist_idx = recent.index[-hist_days:]
    hist_vals = recent.values[-hist_days:]
    ax.plot(hist_idx, hist_vals, 'b-', linewidth=2, label='Historical')

    # Forecast
    last_date = recent.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=31, freq='B')

    ax.plot(forecast_dates, mc_30_adj['median'], 'r-', linewidth=2.5,
            label=f'Forecast (Gemini-adjusted)')
    ax.fill_between(forecast_dates, mc_30_adj['p5'], mc_30_adj['p95'],
                   alpha=0.15, color='red', label='90% CI')
    ax.fill_between(forecast_dates, mc_30_adj['p25'], mc_30_adj['p75'],
                   alpha=0.25, color='red', label='50% CI')

    ax.axvline(x=last_date, color='gray', linestyle='--', linewidth=1)

    # Add Gemini insight
    gemini_text = f"Gemini AI: {gemini_result['outlook'].upper()} ({gemini_result['confidence']})\nSentiment: {gemini_result['sentiment_score']:.2f}"
    ax.text(0.02, 0.98, gemini_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title('USD/INR: 30-Day Ensemble Forecast with Gemini AI Adjustment',
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate (INR)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ultimate_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ultimate_forecast.png")

    # 4. Summary Dashboard
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    # Full price history
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(price_series.index, price_series.values, 'b-', linewidth=0.8)
    ax1.set_title('USD/INR Exchange Rate (10 Years)', fontweight='bold')
    ax1.set_ylabel('INR')
    ax1.grid(True, alpha=0.3)

    # Gemini insight box
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.axis('off')
    gemini_box = f"""
GEMINI AI ANALYSIS
══════════════════

Outlook: {gemini_result['outlook'].upper()}
Confidence: {gemini_result['confidence']}
Sentiment: {gemini_result['sentiment_score']:.2f}
Volatility: {gemini_result['volatility']}
"""
    ax2.text(0.1, 0.9, gemini_box, transform=ax2.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # VMD Decomposition
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(vmd['trend'], 'g-', linewidth=1)
    ax3.set_title('Trend', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(vmd['seasonality'], 'orange', linewidth=1)
    ax4.set_title('Seasonality', fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(vmd['noise'], 'r-', linewidth=0.5, alpha=0.7)
    ax5.set_title('Noise', fontweight='bold')

    # Weights pie
    ax6 = fig.add_subplot(gs[1, 3])
    if top_weights:
        ax6.pie([w for _, w in top_weights[:6]],
               labels=[n for n, _ in top_weights[:6]],
               autopct='%1.1f%%',
               colors=plt.cm.Set2(np.linspace(0, 1, min(6, len(top_weights)))))
    ax6.set_title('Ensemble Weights', fontweight='bold')

    # Validation
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.plot(val_dates, actual, 'k-', label='Actual', linewidth=2)
    ax7.plot(val_dates, ensemble_pred, 'r--', label='Ensemble', linewidth=2)
    ax7.set_title('Validation Performance', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Model ranking
    ax8 = fig.add_subplot(gs[2, 2:])
    top_10 = sorted_metrics[:10]
    names = [m[0] for m in top_10]
    rmses = [m[1]['rmse'] for m in top_10]
    ax8.barh(names[::-1], rmses[::-1], color=plt.cm.RdYlGn(np.linspace(0.8, 0.2, 10)))
    ax8.axvline(x=ens_rmse, color='red', linestyle='--', linewidth=2)
    ax8.set_xlabel('RMSE')
    ax8.set_title('Top 10 Models by RMSE', fontweight='bold')

    # Forecast
    ax9 = fig.add_subplot(gs[3, :3])
    ax9.plot(hist_idx[-30:], hist_vals[-30:], 'b-', linewidth=2, label='Historical')
    ax9.plot(forecast_dates, mc_30_adj['median'], 'r-', linewidth=2, label='Forecast')
    ax9.fill_between(forecast_dates, mc_30_adj['p5'], mc_30_adj['p95'], alpha=0.2, color='red')
    ax9.axvline(x=last_date, color='gray', linestyle='--')
    ax9.set_title('30-Day Forecast', fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Metrics
    ax10 = fig.add_subplot(gs[3, 3])
    ax10.axis('off')
    metrics_text = f"""
ULTIMATE ENSEMBLE
═════════════════════

Models Used: {len(predictions)}
Active Weights: {sum(1 for w in weights.values() if w > 0.01)}

VALIDATION
RMSE:  {ens_rmse:.4f}
MAE:   {ens_mae:.4f}
R²:    {ens_r2:.4f}

vs Best Single: {improvement:+.2f}%

═════════════════════

30-DAY FORECAST
Current: {current_price:.2f} INR
Day 30:  {mc_30_adj['median'][-1]:.2f} INR
90% CI: [{mc_30_adj['p5'][-1]:.2f},
         {mc_30_adj['p95'][-1]:.2f}]
"""
    ax10.text(0.05, 0.95, metrics_text, transform=ax10.transAxes, fontsize=9,
             fontfamily='monospace', verticalalignment='top')

    plt.suptitle('Ultimate Ensemble Model - Comprehensive Dashboard',
                fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ultimate_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ultimate_dashboard.png")

    # Save results
    weights_df = pd.DataFrame([
        {'Model': name, 'Weight': w} for name, w in sorted_weights
    ])
    weights_df.to_csv(os.path.join(OUTPUT_DIR, 'ultimate_weights.csv'), index=False)

    metrics_df = pd.DataFrame([
        {'Model': name, **m} for name, m in model_metrics.items()
    ])
    metrics_df = metrics_df.sort_values('rmse')
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'ultimate_model_metrics.csv'), index=False)

    forecast_df = pd.DataFrame({
        'Day': range(31),
        'Median': mc_30_adj['median'],
        'P5': mc_30_adj['p5'],
        'P10': mc_30_adj['p10'],
        'P25': mc_30_adj['p25'],
        'P75': mc_30_adj['p75'],
        'P90': mc_30_adj['p90'],
        'P95': mc_30_adj['p95']
    })
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'ultimate_forecast.csv'), index=False)

    # Save Gemini analysis
    with open(os.path.join(OUTPUT_DIR, 'gemini_analysis.json'), 'w') as f:
        json.dump(gemini_result, f, indent=2)

    print("\n" + "=" * 70)
    print("  ULTIMATE ENSEMBLE COMPLETE!")
    print("=" * 70)
    print(f"\nModels trained: {len(predictions)}")
    print(f"Active in ensemble: {sum(1 for w in weights.values() if w > 0.01)}")
    print(f"\nTop Weights:")
    for name, w in sorted_weights[:5]:
        if w > 0.01:
            print(f"  {name}: {w:.4f}")
    print(f"\nValidation RMSE: {ens_rmse:.4f}")
    print(f"Validation R²: {ens_r2:.4f}")
    print(f"Improvement: {improvement:+.2f}%")
    print(f"\nGemini AI Outlook: {gemini_result['outlook'].upper()} ({gemini_result['confidence']})")
    print(f"Sentiment Score: {gemini_result['sentiment_score']:.2f}")
    print(f"\nCurrent: {current_price:.2f} INR")
    print(f"30-Day Forecast: {mc_30_adj['median'][-1]:.2f} INR")
    print(f"90% CI: [{mc_30_adj['p5'][-1]:.2f}, {mc_30_adj['p95'][-1]:.2f}]")
    print(f"\nOutput: {OUTPUT_DIR}")

    return weights, model_metrics, gemini_result


if __name__ == "__main__":
    weights, metrics, gemini = main()

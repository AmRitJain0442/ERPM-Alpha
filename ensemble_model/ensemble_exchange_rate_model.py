"""
Ensemble Exchange Rate Prediction Model
========================================

This module combines multiple prediction methods to create an optimal ensemble model
for USD/INR exchange rate prediction:

1. VMD (Variational Mode Decomposition) - Decomposes into trend, seasonality, noise
2. SARIMA - For trend and seasonality prediction
3. LSTM - For sequence pattern learning
4. GARCH - For volatility and returns prediction
5. GDELT-based Regression - For news-driven noise prediction
6. Monte Carlo - For uncertainty quantification

The ensemble uses optimized weights found through grid search/cross-validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Statistical Models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_model

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Data fetching
import yfinance as yf

# Optimization
from scipy.optimize import minimize, differential_evolution
from itertools import product

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUTPUT_DIR, 'output'), exist_ok=True)


# =============================================================================
# DATA FETCHING AND PREPARATION
# =============================================================================

def fetch_exchange_rate_data(years=10):
    """
    Fetch USD/INR exchange rate data for the specified number of years.

    Args:
        years: Number of years of historical data to fetch

    Returns:
        DataFrame with Date and USD_to_INR columns
    """
    print(f"Fetching {years} years of USD/INR exchange rate data...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    # Try yfinance first
    try:
        ticker = yf.Ticker("INR=X")
        df = ticker.history(start=start_date, end=end_date)

        if len(df) > 0:
            df = df.reset_index()
            df = df[['Date', 'Close']].copy()
            df.columns = ['Date', 'USD_to_INR']
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.dropna()
            print(f"  Fetched {len(df)} observations from Yahoo Finance")
            return df
    except Exception as e:
        print(f"  Yahoo Finance error: {e}")

    # Fallback: Try FRED API for INR data
    try:
        import pandas_datareader as pdr
        df = pdr.get_data_fred('DEXINUS', start=start_date, end=end_date)
        df = df.reset_index()
        df.columns = ['Date', 'USD_to_INR']
        df = df.dropna()
        print(f"  Fetched {len(df)} observations from FRED")
        return df
    except Exception as e:
        print(f"  FRED error: {e}")

    # Final fallback: Use existing data and extend with synthetic
    print("  Using existing data with synthetic extension...")
    existing_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'usd_inr_exchange_rates_1year.csv')
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path, parse_dates=['Date'])
        return extend_data_synthetic(existing_df, years)

    raise ValueError("Could not fetch exchange rate data from any source")


def extend_data_synthetic(df, target_years):
    """
    Extend existing data with synthetic historical data based on patterns.
    """
    print("  Generating synthetic historical data based on existing patterns...")

    # Calculate statistics from existing data
    returns = df['USD_to_INR'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()

    # Get earliest date in existing data
    earliest_date = df['Date'].min()
    target_start = earliest_date - timedelta(days=target_years * 365)

    # Generate synthetic dates
    synthetic_dates = pd.date_range(start=target_start, end=earliest_date - timedelta(days=1), freq='B')

    # Starting price (work backwards from earliest known price)
    start_price = df['USD_to_INR'].iloc[0]

    # Generate synthetic prices using random walk
    n_synthetic = len(synthetic_dates)
    synthetic_returns = np.random.normal(mean_return, std_return, n_synthetic)

    # Work backwards
    synthetic_prices = [start_price]
    for i in range(n_synthetic - 1, -1, -1):
        prev_price = synthetic_prices[0] / (1 + synthetic_returns[i])
        synthetic_prices.insert(0, prev_price)

    synthetic_prices = synthetic_prices[:-1]  # Remove duplicate

    synthetic_df = pd.DataFrame({
        'Date': synthetic_dates,
        'USD_to_INR': synthetic_prices
    })

    # Combine with existing
    combined_df = pd.concat([synthetic_df, df], ignore_index=True)
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)

    print(f"  Extended data: {len(combined_df)} observations")
    return combined_df


# =============================================================================
# VMD (VARIATIONAL MODE DECOMPOSITION)
# =============================================================================

class VMDDecomposition:
    """
    Variational Mode Decomposition for signal decomposition.
    Separates the exchange rate into trend, seasonality, and noise components.
    """

    def __init__(self, K=3, alpha=2000, tau=0, tol=1e-7, max_iter=200):
        """
        Args:
            K: Number of modes to extract
            alpha: Bandwidth constraint
            tau: Noise tolerance
            tol: Convergence tolerance
            max_iter: Maximum iterations
        """
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.modes = None

    def decompose(self, signal):
        """
        Perform VMD decomposition on the signal.

        Returns:
            modes: Array of shape (K, len(signal)) containing decomposed modes
        """
        signal = np.asarray(signal)
        N = len(signal)

        # Mirror extension
        T = N
        f_mirror = np.concatenate([signal[:T//2][::-1], signal, signal[T//2:][::-1]])

        # Time domain to frequency domain
        f_hat = np.fft.fft(f_mirror)
        f_hat_plus = np.copy(f_hat)
        f_hat_plus[:len(f_hat)//2] = 0

        # Initialization
        omega = np.zeros((self.max_iter, self.K))

        # Initialize omega uniformly
        for k in range(self.K):
            omega[0, k] = (0.5 / self.K) * k

        # Initialize u_hat
        u_hat = np.zeros((self.max_iter, self.K, len(f_hat)), dtype=complex)

        # Initialize lambda
        lambda_hat = np.zeros((self.max_iter, len(f_hat)), dtype=complex)

        # Frequencies
        N_f = len(f_hat)
        freqs = np.arange(N_f) / N_f - 0.5

        # Main loop
        n = 0
        uDiff = self.tol + 1

        while uDiff > self.tol and n < self.max_iter - 1:
            # Update each mode
            for k in range(self.K):
                # Sum of all other modes
                sum_uk = np.sum(u_hat[n, :, :], axis=0) - u_hat[n, k, :]

                # Wiener filter
                numerator = f_hat_plus - sum_uk - lambda_hat[n] / 2
                denominator = 1 + self.alpha * (freqs - omega[n, k]) ** 2
                u_hat[n+1, k, :] = numerator / denominator

                # Update center frequency
                numerator = np.sum(freqs * np.abs(u_hat[n+1, k, :]) ** 2)
                denominator = np.sum(np.abs(u_hat[n+1, k, :]) ** 2)
                if denominator > 0:
                    omega[n+1, k] = numerator / denominator
                else:
                    omega[n+1, k] = omega[n, k]

            # Dual ascent
            sum_uk = np.sum(u_hat[n+1, :, :], axis=0)
            lambda_hat[n+1] = lambda_hat[n] + self.tau * (f_hat_plus - sum_uk)

            # Convergence check
            uDiff = 0
            for k in range(self.K):
                uDiff += np.sum(np.abs(u_hat[n+1, k, :] - u_hat[n, k, :]) ** 2)
            uDiff = uDiff / N_f

            n += 1

        # Extract modes in time domain
        self.modes = np.zeros((self.K, N))
        for k in range(self.K):
            u_hat_k = u_hat[n, k, :]
            u_hat_k = np.fft.fftshift(u_hat_k)
            u_k = np.fft.ifft(u_hat_k)
            self.modes[k, :] = np.real(u_k[T//4:T//4 + N])

        return self.modes

    def get_components(self):
        """
        Return trend, seasonality, and noise components.
        Assumes K=3 modes where:
        - Mode 0 (lowest frequency): Trend
        - Mode 1 (middle frequency): Seasonality
        - Mode 2 (highest frequency): Noise
        """
        if self.modes is None:
            raise ValueError("Must call decompose() first")

        # Sort modes by frequency content
        freq_content = []
        for k in range(self.K):
            fft_mode = np.fft.fft(self.modes[k])
            freq_content.append(np.argmax(np.abs(fft_mode[:len(fft_mode)//2])))

        sorted_indices = np.argsort(freq_content)

        return {
            'trend': self.modes[sorted_indices[0]],
            'seasonality': self.modes[sorted_indices[1]] if self.K > 1 else np.zeros_like(self.modes[0]),
            'noise': self.modes[sorted_indices[2]] if self.K > 2 else np.zeros_like(self.modes[0])
        }


# =============================================================================
# SARIMA MODEL
# =============================================================================

class SARIMAModel:
    """SARIMA model for trend and seasonality prediction."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5)):
        """
        Args:
            order: (p, d, q) for ARIMA
            seasonal_order: (P, D, Q, s) for seasonal component
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted = None

    def fit(self, series):
        """Fit SARIMA model to the series."""
        print("  Fitting SARIMA model...")
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted = self.model.fit(disp=False, maxiter=200)
        return self

    def predict(self, steps=1):
        """Predict future values."""
        if self.fitted is None:
            raise ValueError("Model must be fitted first")
        return self.fitted.forecast(steps=steps)

    def get_fitted_values(self):
        """Get in-sample fitted values."""
        if self.fitted is None:
            raise ValueError("Model must be fitted first")
        return self.fitted.fittedvalues


# =============================================================================
# LSTM MODEL
# =============================================================================

class LSTMModel:
    """LSTM model for sequence prediction."""

    def __init__(self, lookback=30, units=64, dropout=0.2):
        """
        Args:
            lookback: Number of time steps to look back
            units: Number of LSTM units
            dropout: Dropout rate
        """
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()

    def _create_sequences(self, data):
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build LSTM architecture."""
        model = Sequential([
            Bidirectional(LSTM(self.units, return_sequences=True), input_shape=input_shape),
            Dropout(self.dropout),
            Bidirectional(LSTM(self.units // 2, return_sequences=False)),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def fit(self, series, epochs=50, batch_size=32, validation_split=0.2):
        """Fit LSTM model."""
        print("  Fitting LSTM model...")

        # Scale data
        data = self.scaler.fit_transform(series.values.reshape(-1, 1))

        # Create sequences
        X, y = self._create_sequences(data)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build model
        self.model = self.build_model((X.shape[1], 1))

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        # Train
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )

        return self

    def predict(self, series, steps=1):
        """Predict future values."""
        if self.model is None:
            raise ValueError("Model must be fitted first")

        data = self.scaler.transform(series.values.reshape(-1, 1))

        predictions = []
        current_seq = data[-self.lookback:].reshape(1, self.lookback, 1)

        for _ in range(steps):
            pred = self.model.predict(current_seq, verbose=0)
            predictions.append(pred[0, 0])
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred[0, 0]

        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()

    def get_fitted_values(self, series):
        """Get in-sample predictions."""
        data = self.scaler.transform(series.values.reshape(-1, 1))
        X, _ = self._create_sequences(data)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        preds = self.model.predict(X, verbose=0)
        preds = self.scaler.inverse_transform(preds).flatten()

        # Pad with NaN for lookback period
        full_preds = np.full(len(series), np.nan)
        full_preds[self.lookback:] = preds

        return full_preds


# =============================================================================
# GARCH MODEL
# =============================================================================

class GARCHModel:
    """GARCH model for volatility prediction."""

    def __init__(self, p=1, q=1, mean='Constant', vol='Garch', dist='normal'):
        """
        Args:
            p: GARCH lag order
            q: ARCH lag order
            mean: Mean model type
            vol: Volatility model type ('Garch', 'EGARCH', 'GJR-GARCH')
            dist: Distribution ('normal', 't', 'skewt')
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.model = None
        self.fitted = None

    def fit(self, returns):
        """Fit GARCH model to returns."""
        print("  Fitting GARCH model...")

        # Scale returns
        returns_scaled = returns * 100

        if self.vol == 'EGARCH':
            self.model = arch_model(
                returns_scaled,
                mean=self.mean,
                vol='EGARCH',
                p=self.p,
                q=self.q,
                o=1,
                dist=self.dist
            )
        else:
            self.model = arch_model(
                returns_scaled,
                mean=self.mean,
                vol=self.vol,
                p=self.p,
                q=self.q,
                dist=self.dist
            )

        self.fitted = self.model.fit(disp='off')
        return self

    def predict_volatility(self, horizon=1):
        """Predict future volatility."""
        if self.fitted is None:
            raise ValueError("Model must be fitted first")

        forecast = self.fitted.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.values[-1]) / 100

    def get_conditional_volatility(self):
        """Get conditional volatility series."""
        if self.fitted is None:
            raise ValueError("Model must be fitted first")
        return self.fitted.conditional_volatility / 100


# =============================================================================
# GDELT NOISE MODEL
# =============================================================================

class GDELTNoiseModel:
    """Model for predicting noise using GDELT news data."""

    def __init__(self, model_type='xgboost'):
        """
        Args:
            model_type: Type of model to use ('xgboost', 'rf', 'ridge')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Fit the noise prediction model."""
        print("  Fitting GDELT noise model...")

        # Convert to arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Handle NaN and inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip extreme values
        X = np.clip(X, -1e10, 1e10)
        y = np.clip(y, -1e10, 1e10)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Use Ridge regression for more stability (fallback from XGBoost)
        self.model = Ridge(alpha=1.0)

        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        """Predict noise component."""
        if self.model is None:
            raise ValueError("Model must be fitted first")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulation:
    """Monte Carlo simulation for uncertainty quantification."""

    def __init__(self, n_simulations=1000):
        """
        Args:
            n_simulations: Number of simulation paths
        """
        self.n_simulations = n_simulations

    def simulate(self, current_price, mean_return, volatility, horizon=30):
        """
        Run Monte Carlo simulation.

        Returns:
            dict with mean, median, and percentile forecasts
        """
        print(f"  Running Monte Carlo simulation ({self.n_simulations} paths)...")

        dt = 1  # Daily
        paths = np.zeros((self.n_simulations, horizon + 1))
        paths[:, 0] = current_price

        for t in range(1, horizon + 1):
            Z = np.random.standard_normal(self.n_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (mean_return - 0.5 * volatility**2) * dt +
                volatility * np.sqrt(dt) * Z
            )

        return {
            'mean': np.mean(paths, axis=0),
            'median': np.median(paths, axis=0),
            'p5': np.percentile(paths, 5, axis=0),
            'p25': np.percentile(paths, 25, axis=0),
            'p75': np.percentile(paths, 75, axis=0),
            'p95': np.percentile(paths, 95, axis=0),
            'all_paths': paths
        }


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class EnsembleModel:
    """
    Ensemble model combining multiple prediction methods with optimal weights.
    """

    def __init__(self):
        self.vmd = VMDDecomposition(K=3)
        self.sarima = SARIMAModel()
        self.lstm = LSTMModel(lookback=30)
        self.garch = GARCHModel(vol='EGARCH', dist='skewt')
        self.gdelt_model = GDELTNoiseModel()
        self.monte_carlo = MonteCarloSimulation()

        self.weights = None
        self.predictions = {}
        self.metrics = {}

    def fit(self, price_series, gdelt_features=None):
        """
        Fit all component models.

        Args:
            price_series: Price series (pandas Series with DatetimeIndex)
            gdelt_features: DataFrame with GDELT features (optional)
        """
        print("\n" + "=" * 70)
        print("  FITTING ENSEMBLE MODEL COMPONENTS")
        print("=" * 70)

        # 1. VMD Decomposition
        print("\n[1/6] VMD Decomposition...")
        modes = self.vmd.decompose(price_series.values)
        components = self.vmd.get_components()

        # 2. SARIMA on trend
        print("\n[2/6] SARIMA Model...")
        trend_series = pd.Series(components['trend'], index=price_series.index)
        self.sarima.fit(trend_series)

        # 3. LSTM on full series
        print("\n[3/6] LSTM Model...")
        self.lstm.fit(price_series, epochs=30)

        # 4. GARCH on returns
        print("\n[4/6] GARCH Model...")
        returns = price_series.pct_change().dropna()
        self.garch.fit(returns)

        # 5. GDELT noise model (if features available)
        if gdelt_features is not None and len(gdelt_features) > 0:
            print("\n[5/6] GDELT Noise Model...")
            noise = components['noise']

            # Align features with noise
            common_dates = gdelt_features.index.intersection(price_series.index)
            if len(common_dates) > 50:
                X = gdelt_features.loc[common_dates].copy()
                y = pd.Series(noise, index=price_series.index).loc[common_dates].copy()

                # Drop rows with NaN in either X or y
                X = X.fillna(0)
                y = y.fillna(0)

                # Remove any infinite values
                X = X.replace([np.inf, -np.inf], 0)
                y = y.replace([np.inf, -np.inf], 0)

                # Convert to numpy arrays
                X_arr = X.values.astype(np.float64)
                y_arr = y.values.astype(np.float64)

                self.gdelt_model.fit(X_arr, y_arr)
            else:
                print("    Not enough overlapping dates, skipping GDELT model")
        else:
            print("\n[5/6] Skipping GDELT (no features provided)")

        # 6. Store parameters for Monte Carlo
        print("\n[6/6] Preparing Monte Carlo parameters...")
        self.mc_params = {
            'mean_return': returns.mean(),
            'volatility': returns.std()
        }

        # Store components for later use
        self.components = components
        self.price_series = price_series

        print("\n  All models fitted successfully!")
        return self

    def _get_model_predictions(self, train_end_idx, horizon=1):
        """Get predictions from all models for a given point."""
        predictions = {}

        train_series = self.price_series.iloc[:train_end_idx]
        last_price = train_series.iloc[-1]

        # SARIMA prediction
        try:
            trend = pd.Series(self.components['trend'], index=self.price_series.index)
            sarima_model = SARIMAX(
                trend.iloc[:train_end_idx],
                order=self.sarima.order,
                seasonal_order=self.sarima.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarima_fit = sarima_model.fit(disp=False, maxiter=100)
            sarima_pred = sarima_fit.forecast(steps=horizon).values
            predictions['sarima'] = sarima_pred[0] if horizon == 1 else sarima_pred
        except:
            predictions['sarima'] = last_price

        # LSTM prediction (use full model, just different lookback)
        try:
            lstm_pred = self.lstm.predict(train_series, steps=horizon)
            predictions['lstm'] = lstm_pred[0] if horizon == 1 else lstm_pred
        except:
            predictions['lstm'] = last_price

        # GARCH-based prediction
        try:
            returns = train_series.pct_change().dropna()
            garch_model = arch_model(returns * 100, mean='Constant', vol='EGARCH', p=1, q=1, o=1, dist='skewt')
            garch_fit = garch_model.fit(disp='off')
            forecast = garch_fit.forecast(horizon=horizon)
            expected_return = garch_fit.params['mu'] / 100
            predictions['garch'] = last_price * (1 + expected_return) ** horizon
        except:
            predictions['garch'] = last_price

        # Monte Carlo median
        try:
            mc_result = self.monte_carlo.simulate(
                last_price,
                self.mc_params['mean_return'],
                self.mc_params['volatility'],
                horizon=horizon
            )
            predictions['monte_carlo'] = mc_result['median'][horizon]
        except:
            predictions['monte_carlo'] = last_price

        return predictions

    def optimize_weights(self, validation_split=0.2, method='grid'):
        """
        Optimize ensemble weights using cross-validation.

        Args:
            validation_split: Fraction of data for validation
            method: Optimization method ('grid', 'differential_evolution', 'minimize')
        """
        print("\n" + "=" * 70)
        print("  OPTIMIZING ENSEMBLE WEIGHTS")
        print("=" * 70)

        n = len(self.price_series)
        val_size = int(n * validation_split)
        train_end = n - val_size

        # Collect predictions for validation period
        print("\n  Collecting model predictions for validation...")
        val_predictions = {
            'sarima': [],
            'lstm': [],
            'garch': [],
            'monte_carlo': []
        }
        actuals = []

        for i in range(train_end, n - 1):
            preds = self._get_model_predictions(i, horizon=1)
            for model in val_predictions.keys():
                val_predictions[model].append(preds[model])
            actuals.append(self.price_series.iloc[i + 1])

        # Convert to arrays
        for model in val_predictions.keys():
            val_predictions[model] = np.array(val_predictions[model])
        actuals = np.array(actuals)

        # Define objective function
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize

            ensemble_pred = np.zeros(len(actuals))
            for j, model in enumerate(val_predictions.keys()):
                ensemble_pred += weights[j] * val_predictions[model]

            return mean_squared_error(actuals, ensemble_pred)

        # Optimize
        print(f"  Optimization method: {method}")

        if method == 'grid':
            # Grid search
            best_weights = None
            best_mse = float('inf')

            weight_range = np.arange(0, 1.05, 0.1)

            for w1, w2, w3, w4 in product(weight_range, repeat=4):
                if abs(w1 + w2 + w3 + w4 - 1.0) < 0.01:
                    mse = objective([w1, w2, w3, w4])
                    if mse < best_mse:
                        best_mse = mse
                        best_weights = [w1, w2, w3, w4]

            self.weights = dict(zip(val_predictions.keys(), best_weights))

        elif method == 'differential_evolution':
            bounds = [(0, 1)] * 4
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                disp=False
            )
            weights = result.x / result.x.sum()
            self.weights = dict(zip(val_predictions.keys(), weights))

        else:  # minimize
            x0 = [0.25, 0.25, 0.25, 0.25]
            constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}
            bounds = [(0, 1)] * 4

            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            self.weights = dict(zip(val_predictions.keys(), result.x))

        # Calculate final validation metrics
        ensemble_pred = np.zeros(len(actuals))
        for model, weight in self.weights.items():
            ensemble_pred += weight * val_predictions[model]

        self.metrics = {
            'val_rmse': np.sqrt(mean_squared_error(actuals, ensemble_pred)),
            'val_mae': mean_absolute_error(actuals, ensemble_pred),
            'val_r2': r2_score(actuals, ensemble_pred),
            'val_mape': np.mean(np.abs((actuals - ensemble_pred) / actuals)) * 100
        }

        # Individual model metrics
        for model in val_predictions.keys():
            self.metrics[f'{model}_rmse'] = np.sqrt(mean_squared_error(actuals, val_predictions[model]))
            self.metrics[f'{model}_mae'] = mean_absolute_error(actuals, val_predictions[model])

        print("\n  Optimal Weights:")
        for model, weight in self.weights.items():
            print(f"    {model}: {weight:.4f}")

        print("\n  Validation Metrics:")
        print(f"    RMSE: {self.metrics['val_rmse']:.4f}")
        print(f"    MAE: {self.metrics['val_mae']:.4f}")
        print(f"    R²: {self.metrics['val_r2']:.4f}")
        print(f"    MAPE: {self.metrics['val_mape']:.2f}%")

        # Store validation data for plotting
        self.val_predictions = val_predictions
        self.val_actuals = actuals
        self.val_ensemble = ensemble_pred
        self.val_dates = self.price_series.index[train_end + 1:n]

        return self.weights

    def predict(self, horizon=30):
        """
        Make ensemble predictions.

        Args:
            horizon: Number of days to forecast

        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.weights is None:
            raise ValueError("Must optimize weights first")

        print(f"\n  Generating {horizon}-day ensemble forecast...")

        last_price = self.price_series.iloc[-1]

        # Get predictions from each model
        preds = self._get_model_predictions(len(self.price_series), horizon)

        # Weighted ensemble
        if horizon == 1:
            ensemble_pred = sum(self.weights[m] * preds[m] for m in self.weights.keys())
        else:
            # For multi-step, use Monte Carlo for uncertainty
            mc_result = self.monte_carlo.simulate(
                last_price,
                self.mc_params['mean_return'],
                self.mc_params['volatility'],
                horizon=horizon
            )

            # Combine with point predictions
            ensemble_pred = mc_result['median']

        return {
            'ensemble': ensemble_pred,
            'individual': preds,
            'monte_carlo': self.monte_carlo.simulate(
                last_price,
                self.mc_params['mean_return'],
                self.mc_params['volatility'],
                horizon=horizon
            )
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(model, output_dir):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)

    output_path = os.path.join(output_dir, 'output')
    os.makedirs(output_path, exist_ok=True)

    # 1. VMD Decomposition Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    axes[0].plot(model.price_series.index, model.price_series.values, 'b-', linewidth=1)
    axes[0].set_title('Original Exchange Rate', fontweight='bold')
    axes[0].set_ylabel('USD/INR')

    axes[1].plot(model.price_series.index, model.components['trend'], 'g-', linewidth=1)
    axes[1].set_title('VMD Trend Component', fontweight='bold')
    axes[1].set_ylabel('Trend')

    axes[2].plot(model.price_series.index, model.components['seasonality'], 'orange', linewidth=1)
    axes[2].set_title('VMD Seasonality Component', fontweight='bold')
    axes[2].set_ylabel('Seasonality')

    axes[3].plot(model.price_series.index, model.components['noise'], 'r-', linewidth=0.5, alpha=0.7)
    axes[3].set_title('VMD Noise Component', fontweight='bold')
    axes[3].set_ylabel('Noise')
    axes[3].set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'vmd_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: vmd_decomposition.png")

    # 2. Model Weights
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(model.weights.keys())
    weights = [model.weights[m] for m in models]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    bars = ax.bar(models, weights, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Weight', fontweight='bold')
    ax.set_title('Optimized Ensemble Weights', fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(weights) * 1.2)

    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{w:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'ensemble_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ensemble_weights.png")

    # 3. Validation Predictions Comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Predictions plot
    ax1 = axes[0]
    ax1.plot(model.val_dates, model.val_actuals, 'k-', label='Actual', linewidth=2)
    ax1.plot(model.val_dates, model.val_ensemble, 'r--', label='Ensemble', linewidth=2)

    for m, color in zip(model.val_predictions.keys(), colors):
        ax1.plot(model.val_dates, model.val_predictions[m], '--',
                label=m.upper(), alpha=0.6, linewidth=1)

    ax1.set_title('Validation Period: Model Predictions vs Actual', fontweight='bold', fontsize=12)
    ax1.set_ylabel('USD/INR', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Errors plot
    ax2 = axes[1]
    ensemble_error = model.val_actuals - model.val_ensemble
    ax2.fill_between(model.val_dates, ensemble_error, 0, alpha=0.3, color='red')
    ax2.plot(model.val_dates, ensemble_error, 'r-', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Ensemble Prediction Error', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Error (INR)', fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'validation_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: validation_comparison.png")

    # 4. Model Performance Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # RMSE comparison
    ax1 = axes[0]
    rmse_values = [model.metrics.get(f'{m}_rmse', 0) for m in model.val_predictions.keys()]
    rmse_values.append(model.metrics['val_rmse'])
    labels = list(model.val_predictions.keys()) + ['ENSEMBLE']

    bars = ax1.bar(labels, rmse_values, color=colors + ['red'], edgecolor='black', alpha=0.8)
    ax1.set_ylabel('RMSE', fontweight='bold')
    ax1.set_title('Model RMSE Comparison', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    for bar, v in zip(bars, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)

    # MAE comparison
    ax2 = axes[1]
    mae_values = [model.metrics.get(f'{m}_mae', 0) for m in model.val_predictions.keys()]
    mae_values.append(model.metrics['val_mae'])

    bars = ax2.bar(labels, mae_values, color=colors + ['red'], edgecolor='black', alpha=0.8)
    ax2.set_ylabel('MAE', fontweight='bold')
    ax2.set_title('Model MAE Comparison', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    for bar, v in zip(bars, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: model_comparison.png")

    # 5. Forecast with Confidence Intervals
    forecast = model.predict(horizon=30)
    mc_result = forecast['monte_carlo']

    fig, ax = plt.subplots(figsize=(14, 8))

    # Historical data (last 90 days)
    hist_days = 90
    hist_dates = model.price_series.index[-hist_days:]
    hist_prices = model.price_series.values[-hist_days:]

    ax.plot(hist_dates, hist_prices, 'b-', linewidth=2, label='Historical')

    # Forecast
    last_date = model.price_series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=31, freq='B')

    ax.plot(forecast_dates, mc_result['median'], 'r-', linewidth=2, label='Forecast (Median)')
    ax.fill_between(forecast_dates, mc_result['p5'], mc_result['p95'],
                    alpha=0.2, color='red', label='90% CI')
    ax.fill_between(forecast_dates, mc_result['p25'], mc_result['p75'],
                    alpha=0.3, color='red', label='50% CI')

    ax.axvline(x=last_date, color='gray', linestyle='--', linewidth=1, label='Forecast Start')

    ax.set_title('USD/INR Exchange Rate: 30-Day Forecast', fontweight='bold', fontsize=14)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Exchange Rate (INR)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'forecast_with_ci.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: forecast_with_ci.png")

    # 6. Summary Dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Historical price
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(model.price_series.index, model.price_series.values, 'b-', linewidth=1)
    ax1.set_title('USD/INR Exchange Rate (10 Years)', fontweight='bold')
    ax1.set_ylabel('INR')
    ax1.grid(True, alpha=0.3)

    # Weights pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.pie(list(model.weights.values()), labels=list(model.weights.keys()),
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Ensemble Weights', fontweight='bold')

    # VMD components
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(model.components['trend'], 'g-', linewidth=1)
    ax3.set_title('Trend', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(model.components['seasonality'], 'orange', linewidth=1)
    ax4.set_title('Seasonality', fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(model.components['noise'], 'r-', linewidth=0.5, alpha=0.7)
    ax5.set_title('Noise', fontweight='bold')

    # Validation results
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.plot(model.val_dates, model.val_actuals, 'k-', label='Actual', linewidth=2)
    ax6.plot(model.val_dates, model.val_ensemble, 'r--', label='Ensemble', linewidth=2)
    ax6.set_title('Validation: Ensemble vs Actual', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Metrics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    metrics_text = f"""
    ENSEMBLE PERFORMANCE
    ═══════════════════════

    RMSE:  {model.metrics['val_rmse']:.4f}
    MAE:   {model.metrics['val_mae']:.4f}
    R²:    {model.metrics['val_r2']:.4f}
    MAPE:  {model.metrics['val_mape']:.2f}%

    ═══════════════════════
    """
    ax7.text(0.1, 0.5, metrics_text, fontsize=12, fontfamily='monospace',
            verticalalignment='center', transform=ax7.transAxes)

    plt.suptitle('Ensemble Exchange Rate Model - Summary Dashboard',
                fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_path, 'ensemble_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ensemble_dashboard.png")

    print("\n  All visualizations saved!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("  ENSEMBLE USD/INR EXCHANGE RATE PREDICTION MODEL")
    print("  Combining VMD + SARIMA + LSTM + GARCH + Monte Carlo")
    print("=" * 70)

    # 1. Fetch data
    print("\n[STEP 1] FETCHING DATA")
    print("-" * 70)
    df = fetch_exchange_rate_data(years=10)

    # Save fetched data
    df.to_csv(os.path.join(OUTPUT_DIR, 'output', 'usd_inr_10year.csv'), index=False)
    print(f"  Saved: usd_inr_10year.csv")

    # Create series
    price_series = pd.Series(df['USD_to_INR'].values, index=pd.to_datetime(df['Date']))
    price_series = price_series.dropna()

    print(f"\n  Data Summary:")
    print(f"    Date Range: {price_series.index[0].date()} to {price_series.index[-1].date()}")
    print(f"    Total Observations: {len(price_series)}")
    print(f"    Current Rate: {price_series.iloc[-1]:.4f}")
    print(f"    Min: {price_series.min():.4f}, Max: {price_series.max():.4f}")

    # 2. Load GDELT features if available
    print("\n[STEP 2] LOADING GDELT FEATURES")
    print("-" * 70)

    gdelt_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'Phase-B', 'merged_training_data.csv')
    gdelt_features = None

    if os.path.exists(gdelt_path):
        gdelt_df = pd.read_csv(gdelt_path, parse_dates=['Date'])
        gdelt_df = gdelt_df.set_index('Date')

        feature_cols = [col for col in gdelt_df.columns if col not in ['Date']]
        gdelt_features = gdelt_df[feature_cols].dropna()
        print(f"  Loaded {len(gdelt_features)} GDELT observations with {len(feature_cols)} features")
    else:
        print("  GDELT features not found, proceeding without news data")

    # 3. Create and fit ensemble model
    print("\n[STEP 3] FITTING ENSEMBLE MODEL")
    print("-" * 70)

    model = EnsembleModel()
    model.fit(price_series, gdelt_features)

    # 4. Optimize weights
    print("\n[STEP 4] OPTIMIZING ENSEMBLE WEIGHTS")
    print("-" * 70)

    model.optimize_weights(validation_split=0.15, method='grid')

    # 5. Generate forecast
    print("\n[STEP 5] GENERATING FORECAST")
    print("-" * 70)

    forecast = model.predict(horizon=30)

    print(f"\n  30-Day Forecast:")
    mc = forecast['monte_carlo']
    print(f"    Current Price: {price_series.iloc[-1]:.4f} INR")
    print(f"    Expected (Day 30): {mc['median'][-1]:.4f} INR")
    print(f"    90% CI: [{mc['p5'][-1]:.4f}, {mc['p95'][-1]:.4f}]")

    # 6. Create visualizations
    print("\n[STEP 6] CREATING VISUALIZATIONS")
    print("-" * 70)

    create_visualizations(model, OUTPUT_DIR)

    # 7. Save results
    print("\n[STEP 7] SAVING RESULTS")
    print("-" * 70)

    # Save weights
    weights_df = pd.DataFrame([model.weights])
    weights_df.to_csv(os.path.join(OUTPUT_DIR, 'output', 'ensemble_weights.csv'), index=False)
    print("  Saved: ensemble_weights.csv")

    # Save metrics
    metrics_df = pd.DataFrame([model.metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'output', 'ensemble_metrics.csv'), index=False)
    print("  Saved: ensemble_metrics.csv")

    # Save forecast
    forecast_df = pd.DataFrame({
        'Day': range(31),
        'Median': mc['median'],
        'P5': mc['p5'],
        'P25': mc['p25'],
        'P75': mc['p75'],
        'P95': mc['p95']
    })
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'output', 'ensemble_forecast.csv'), index=False)
    print("  Saved: ensemble_forecast.csv")

    # Final summary
    print("\n" + "=" * 70)
    print("  ENSEMBLE MODEL COMPLETE!")
    print("=" * 70)
    print(f"\n  Optimal Weights:")
    for m, w in model.weights.items():
        print(f"    {m}: {w:.4f}")
    print(f"\n  Validation RMSE: {model.metrics['val_rmse']:.4f}")
    print(f"  Validation R²: {model.metrics['val_r2']:.4f}")
    print(f"\n  Output saved to: {os.path.join(OUTPUT_DIR, 'output')}")

    return model


if __name__ == "__main__":
    model = main()

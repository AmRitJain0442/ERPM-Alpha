import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CUSTOM TECHNICAL INDICATORS (No pandas_ta needed)
# ==========================================
def compute_rsi(series, period=14):
    """Calculate RSI manually"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bbands(series, period=20, std_dev=2):
    """Calculate Bollinger Bands manually"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = 'Super_Master_Dataset.csv'  # Your SQL + YFinance merged file
TEST_DAYS = 60                          # Number of days to simulate "Live Trading"
REGIMES = 3                             # Number of market states (Calm, Panic, Shift)

# ==========================================
# 1. FEATURE ENGINEERING ENGINE
# ==========================================
def engineer_features(df):
    """
    Takes raw data and adds 'Quant' features:
    - Technicals (RSI, Bollinger, Lags)
    - Regime Drivers (Volatility, Yield Velocity)
    """
    print("--- STEP 1: GENERATING QUANT FEATURES ---")
    df = df.copy()
    
    # A. Target Variable (Predict Tomorrow's Price)
    df['Target_Price'] = df['INR'].shift(-1)
    
    # B. Autoregression (The #1 Predictor)
    # Price yesterday is the best predictor of price today
    df['Lag_Price_1'] = df['INR'].shift(1)
    df['Lag_Price_2'] = df['INR'].shift(2)
    df['Lag_Price_5'] = df['INR'].shift(5)
    
    # C. Technical Indicators (Momentum)
    # RSI (Relative Strength Index)
    df['RSI'] = compute_rsi(df['INR'], period=14)
    # Bollinger Bands (Volatility)
    df['BBU'], df['BBM'], df['BBL'] = compute_bbands(df['INR'], period=20, std_dev=2)
    # SMA (Trend)
    df['SMA_20'] = df['INR'].rolling(20).mean()
    df['Dist_from_SMA'] = df['INR'] - df['SMA_20']
    
    # D. Regime Detectors (For GMM)
    # 1. Realized Volatility (Standard Deviation of returns)
    df['Regime_Vol'] = df['INR'].pct_change().rolling(10).std()
    # 2. Global Fear (Average Panic Index)
    df['Regime_Fear'] = (df['IN_Panic_Index'] + df['US_Panic_Index']) / 2
    # 3. Yield Velocity (How fast are US rates moving?)
    df['Regime_Yield_Vel'] = df['US10Y'].diff()
    
    # Cleanup
    df.dropna(inplace=True)
    print(f"Features Generated. Dataset Size: {len(df)} rows.")
    return df

# ==========================================
# 2. REGIME DETECTION ENGINE (GMM)
# ==========================================
def detect_regimes(df):
    """
    Uses Unsupervised Learning to cluster days into 'States'
    """
    print("\n--- STEP 2: DETECTING MARKET REGIMES ---")
    
    # Features that define the "State of the World"
    regime_features = ['Regime_Vol', 'Regime_Fear', 'Regime_Yield_Vel']
    X_regime = df[regime_features]
    
    # Fit GMM
    gmm = GaussianMixture(n_components=REGIMES, covariance_type='full', random_state=42)
    gmm.fit(X_regime)
    
    # Assign States
    df['Regime'] = gmm.predict(X_regime)
    
    # Print Stats
    print("Regime Profiles:")
    print(df.groupby('Regime')[['Regime_Vol', 'Regime_Fear', 'INR']].mean())
    
    return df, gmm

# ==========================================
# 3. MIXTURE OF EXPERTS TRAINING
# ==========================================
def train_experts(train_df):
    """
    Trains a separate XGBoost model for each Regime
    """
    print("\n--- STEP 3: TRAINING EXPERTS ---")
    
    experts = {}
    
    # Features to train on (Exclude Targets & Labels)
    drop_cols = ['Target_Price', 'Regime', 'Regime_Vol', 'Regime_Fear', 'Regime_Yield_Vel']
    features = [c for c in train_df.columns if c not in drop_cols]
    
    for state in range(REGIMES):
        # Filter data for this state
        state_data = train_df[train_df['Regime'] == state]
        
        if len(state_data) < 30:
            print(f"  -> WARNING: Regime {state} has insufficient data. Skipping.")
            continue
            
        # Train Expert
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            objective='reg:squarederror',
            n_jobs=-1
        )
        model.fit(state_data[features], state_data['Target_Price'])
        experts[state] = model
        print(f"  -> Expert {state} Trained (Rows: {len(state_data)})")
        
    return experts, features

# ==========================================
# 4. SOFT VOTING PREDICTION
# ==========================================
def predict_soft_voting(test_df, experts, gmm, features):
    """
    Predicts price using probability-weighted blending
    """
    print("\n--- STEP 4: RUNNING SOFT VOTING INFERENCE ---")
    
    predictions = []
    actuals = test_df['Target_Price'].values
    
    # Get Probabilities for Test Data
    # (How much does this day belong to Regime 0, 1, or 2?)
    regime_cols = ['Regime_Vol', 'Regime_Fear', 'Regime_Yield_Vel']
    probs = gmm.predict_proba(test_df[regime_cols])
    
    for i in range(len(test_df)):
        row = test_df.iloc[[i]]
        current_probs = probs[i] # e.g. [0.1, 0.8, 0.1]
        
        weighted_pred = 0
        
        for state in range(REGIMES):
            if state in experts:
                # Ask Expert
                expert_pred = experts[state].predict(row[features])[0]
                # Weight by Probability
                weighted_pred += expert_pred * current_probs[state]
            else:
                # Fallback: Use Naive Last Price
                weighted_pred += row['INR'].values[0] * current_probs[state]
                
        predictions.append(weighted_pred)
        
    return predictions, actuals

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # 1. Load
    print(f"Loading {DATA_FILE}...")
    try:
        raw_df = pd.read_csv(DATA_FILE, parse_dates=['Date'], index_col='Date')
    except FileNotFoundError:
        print("ERROR: File not found. Please run the SQL Fetcher script first.")
        exit()

    # 2. Engineer
    df = engineer_features(raw_df)

    # 3. Split (Walk-Forward)
    train_df = df.iloc[:-TEST_DAYS].copy()
    test_df = df.iloc[-TEST_DAYS:].copy()

    # 4. Detect Regimes (Learn from Train, Apply to Test)
    train_df, gmm_model = detect_regimes(train_df)
    
    # Apply GMM to Test (Don't refit! Use the trained logic)
    test_df['Regime'] = gmm_model.predict(test_df[['Regime_Vol', 'Regime_Fear', 'Regime_Yield_Vel']])

    # 5. Train Experts
    expert_models, feature_cols = train_experts(train_df)

    # 6. Predict
    preds, y_true = predict_soft_voting(test_df, expert_models, gmm_model, feature_cols)

    # 7. Evaluate
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)

    print(f"\n=== FINAL REPORT ===")
    print(f"RMSE: {rmse:.4f} INR")
    print(f"MAE:  {mae:.4f} INR")

    # 8. Visualize
    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index, y_true, label='Actual Price', color='black', alpha=0.5, linewidth=2)
    plt.plot(test_df.index, preds, label='Soft Voting AI', color='purple', linestyle='--', linewidth=2)
    
    # Color background by Regime
    # (Visual flair to show which regime was dominant)
    y_min, y_max = min(y_true), max(y_true)
    for i in range(len(test_df)):
        regime = test_df['Regime'].iloc[i]
        color = ['green', 'orange', 'red'][regime % 3]
        plt.fill_between([test_df.index[i], test_df.index[i]], y_min, y_max, color=color, alpha=0.1)

    plt.title(f"Hybrid Quant Model Forecast (RMSE: {rmse:.2f})\nBackground Color = Market Regime")
    plt.legend()
    plt.show()
    
    # 9. Save Forecasts
    results = pd.DataFrame({'Actual': y_true, 'Predicted': preds}, index=test_df.index)
    results.to_csv('final_forecasts.csv')
    print("Forecasts saved to 'final_forecasts.csv'")
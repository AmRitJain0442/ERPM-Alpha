import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
# Assuming your CSV is named 'data.csv'
# Parse dates to ensure time-series continuity
df = pd.read_csv(r'C:\Users\amrit\Desktop\gdelt_india\Phase-B\merged_training_data.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

print(f"Loaded {len(df)} rows of data.")

# --- 2. FEATURE ENGINEERING (APPLYING THE LAGS) ---
# We use the specific lags you discovered in your correlation analysis.

# Feature 1: Economy Tone (Impacts market 3 days later)
df['Feat_Tone_Econ_Lag3'] = df['Tone_Economy'].shift(3)

# Feature 2: Goldstein Stability (Impacts market 4 days later)
df['Feat_Goldstein_Lag4'] = df['Goldstein_Weighted'].shift(4)

# Feature 3: Volume Spike (Immediate impact? Let's try Lag 1 just to be safe/realistic)
df['Feat_Vol_Spike_Lag1'] = df['Volume_Spike'].shift(1)

# Drop the initial rows that contain NaNs due to shifting
df.dropna(inplace=True)

print(f"Data after Lagging & Cleaning: {len(df)} rows.")

# --- 3. SPLIT DATA (Walk-Forward) ---
# We do NOT shuffle. The past must predict the future.
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# --- 4. MODEL A: XGBOOST (PREDICTING THE NOISE) ---
# Target: 'IMF_3' (The red noise line)
features = ['Feat_Tone_Econ_Lag3', 'Feat_Goldstein_Lag4', 'Feat_Vol_Spike_Lag1']
target = 'IMF_3'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

print("\nTraining XGBoost on Noise (IMF_3)...")
reg = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,    # Slower learning prevents overfitting on small data
    max_depth=3,           # Shallow trees avoid memorizing noise
    objective='reg:squarederror',
    n_jobs=-1
)
reg.fit(X_train, y_train)

# Predict Noise
noise_pred_train = reg.predict(X_train)
noise_pred_test = reg.predict(X_test)

# --- 5. MODEL B: ARIMA (PREDICTING THE TREND - OPTIONAL) ---
# *Note: Since you only provided the Noise (IMF_3) and News in the CSV, 
# I will focus this evaluation PURELY on the Noise Prediction capability.*
# *If you have the Trend (IMF_1) in the CSV, uncomment the lines below.*

# trend_model = ARIMA(train_data['IMF_1'], order=(1,1,0)).fit()
# trend_pred = trend_model.forecast(steps=len(test_data))

# --- 6. VISUALIZE RESULTS ---
plt.figure(figsize=(15, 6))
plt.plot(test_data.index, y_test, label='Actual Volatility (IMF 3)', color='black', alpha=0.6)
plt.plot(test_data.index, noise_pred_test, label='Predicted Volatility (News-Based)', color='red', linestyle='--')

plt.title('Alpha Check: Can News Signals Predict Exchange Rate Volatility?')
plt.xlabel('Date')
plt.ylabel('Price Deviation (IMF 3)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 7. FEATURE IMPORTANCE ---
# This tells you WHICH news signal actually mattered
xgb.plot_importance(reg)
plt.title("What Drives the Market? (Feature Importance)")
plt.show()

# --- 8. PERFORMANCE METRICS ---
rmse = np.sqrt(mean_squared_error(y_test, noise_pred_test))
mae = mean_absolute_error(y_test, noise_pred_test)
print(f"\nTest RMSE: {rmse:.5f}")
print(f"Test MAE:  {mae:.5f}")
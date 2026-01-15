import xgboost as xgb
from sklearn.metrics import mean_squared_error

# --- SETUP PREDICTION TARGET ---
# We predict the Next Day's Price
df['Target_Price'] = df['INR'].shift(-1)
data = df.dropna().copy()

# Features (Everything except Target and Regime helper cols)
features = [c for c in data.columns if c not in ['Target_Price', 'Regime', 'Realized_Vol']]

# Split Time-Series (Last 100 days for test)
train = data.iloc[:-100]
test = data.iloc[-100:]

# --- TRAIN THE EXPERTS ---
models = {}
print("\nTraining Mixture of Experts...")

for state in [0, 1, 2]:
    # 1. Filter data for this specific state
    state_train = train[train['Regime'] == state]
    
    if len(state_train) < 50: 
        print(f"Warning: Regime {state} has too little data. Skipping.")
        continue

    # 2. Train a specialized XGBoost
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, objective='reg:squarederror')
    model.fit(state_train[features], state_train['Target_Price'])
    models[state] = model
    print(f"  -> Expert Model {state} Trained (Rows: {len(state_train)})")

# --- PREDICT (DYNAMIC SWITCHING) ---
print("\nRunning Dynamic Prediction...")
predictions = []
actuals = []

for index, row in test.iterrows():
    current_regime = int(row['Regime']) # GMM told us which state we are in
    
    # Select the expert for this state
    if current_regime in models:
        pred = models[current_regime].predict(pd.DataFrame([row[features]]))[0]
    else:
        # Fallback (Average of all if state is rare)
        pred = row['INR'] 
    
    predictions.append(pred)
    actuals.append(row['Target_Price'])

# --- EVALUATE ---
rmse = np.sqrt(mean_squared_error(actuals, predictions))
print(f"Mixture of Experts RMSE: {rmse:.4f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(actuals, label='Actual', color='black', alpha=0.5)
plt.plot(predictions, label='Regime-Switched Prediction', color='blue')
plt.title(f'Dynamic Regime Prediction (RMSE: {rmse:.2f})')
plt.legend()
plt.show()
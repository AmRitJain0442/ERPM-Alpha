import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from vmdpy import VMD

# --- 1. GET DATA (Using yfinance as placeholder for your dataset) ---
print("Fetching INR/USD data...")
# 'INR=X' is the ticker for USD/INR exchange rate
data = yf.download('INR=X', period='2y', interval='1d') 

# Handle MultiIndex if present (yfinance update)
if isinstance(data.columns, pd.MultiIndex):
    # Assuming 'INR=X' is the ticker, or just take the first column
    prices_series = data['Close'].iloc[:, 0]
else:
    prices_series = data['Close']

# Clean data (remove NaNs)
prices_series = prices_series.dropna()
dates = prices_series.index
prices = prices_series.values

# --- 2. CONFIGURING VMD (The "Knobs") ---
dates = data.index

# --- 2. CONFIGURING VMD (The "Knobs") ---
# These are the expert settings for Financial Time Series
alpha = 2000       # Bandwidth constraint. 
                   # Higher = Stricter separation (good for separating Trend from Noise).
                   # Lower = Looser (allows modes to overlap).
                   
tau = 0.           # Noise-tolerance (0 = Strict fidelity to original signal).

K = 3              # Number of modes.
                   # Mode 1: Long-term Trend (Macro)
                   # Mode 2: Medium-term Cycle (Business cycles)
                   # Mode 3: High-frequency Residuals (News/Noise)

DC = 0             # No DC part imposed
init = 1           # Initialize omegas uniformly
tol = 1e-7         # Tolerance for convergence

# --- 3. RUNNING VMD ---
print("Decomposing signal...")
# u is the decomposed modes (shape: K x N)
u, u_hat, omega = VMD(prices, alpha, tau, K, DC, init, tol)

# --- 4. VISUALIZATION ---
# This visual check is CRITICAL. 
plt.figure(figsize=(12, 10))

# Plot Original Price
plt.subplot(K+1, 1, 1)
plt.plot(prices, color='black')
plt.title('Original INR/USD Exchange Rate')
plt.grid(True)

# Plot the Decomposed Modes (IMFs)
mode_names = ['Trend (IMF 1)', 'Cycle (IMF 2)', 'Noise (IMF 3)']
colors = ['blue', 'green', 'red']

for i in range(K):
    plt.subplot(K+1, 1, i+2)
    plt.plot(u[i], color=colors[i])
    plt.title(mode_names[i] if i < 3 else f'Mode {i+1}')
    plt.grid(True)

plt.tight_layout()
# plt.show() # Commenting out show to run in background

print("Decomposition Complete. You now have 3 separate datasets in variable 'u'.")

# --- 5. SAVING DATA ---
print("Saving IMF 3 to CSV...")
print(f"Length of dates: {len(dates)}")
print(f"Length of u[2]: {len(u[2])}")

if len(dates) != len(u[2]):
    print("Warning: Length mismatch. Truncating dates to match IMF length.")
    dates = dates[:len(u[2])]

imf3_df = pd.DataFrame({'Date': dates, 'IMF_3': u[2]})
# Save to the parent directory as expected by Phase-B script
imf3_df.to_csv('../IMF_3.csv', index=False)
print("Saved IMF 3 to ../IMF_3.csv")

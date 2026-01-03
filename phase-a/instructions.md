Here is the Python implementation for **Step A: Signal Decomposition**.

I have written this script to be "plug-and-play." It currently fetches live INR/USD data using `yfinance` so you can test it immediately. Later, you can swap the `yfinance` line with your own CSV loader.

### **Prerequisites**

You will need to install the VMD library (it's not in standard Anaconda).

```bash
pip install vmdpy yfinance matplotlib

```

### **The Code (Step A)**

```python
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from vmdpy import VMD

# --- 1. GET DATA (Using yfinance as placeholder for your dataset) ---
print("Fetching INR/USD data...")
# 'INR=X' is the ticker for USD/INR exchange rate
data = yf.download('INR=X', period='2y', interval='1d') 
prices = data['Close'].values.flatten()

# Clean data (remove NaNs if any)
prices = prices[~np.isnan(prices)]

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
plt.show()

print("Decomposition Complete. You now have 3 separate datasets in variable 'u'.")

```

---

### **How to Read the Output (The "Expert" Check)**

Run this script and look at the graphs generated.

1. **IMF 1 (Trend - Blue):** This should look like a very smooth curve. It represents the "Fundamental Value" of INR based on inflation and interest rates.
* *Expert Note:* If this is too jagged, increase `alpha`.


2. **IMF 2 (Cycle - Green):** This should look like a wave (oscillating). This captures the market's "breathing."
3. **IMF 3 (Noise - Red):**  This should look like random static or a heartbeat monitor.
* **CRITICAL:** This Red line is the *only* thing your GDELT/News data can predict. Do not try to predict the Blue line with news headlines.



### **Next Step**

Once you have this running and can see the "Red Line" (IMF 3), we are ready for **Step B**.

We will need to prepare your GDELT data to align with this Red Line.
**Shall we move to the GDELT alignment strategy?**
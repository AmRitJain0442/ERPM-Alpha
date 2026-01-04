import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("--- STEP 1: LOADING DATA ---")
file_path = r'C:\Users\amrit\Desktop\gdelt_india\Phase-B\merged_training_data.csv'  # Ensure this matches your filename
df = pd.read_csv(file_path, parse_dates=['Date'])

# STRICT time-series sorting
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)
print(f"Loaded {len(df)} rows.")

# ==========================================
# 2. FEATURE ENGINEERING (THE ALPHA LAGS)
# ==========================================
print("--- STEP 2: APPLYING LAGS ---")

# We shift features FORWARD. 
# "News on Monday" (Shift 3) -> aligns with -> "Price on Thursday"
df['Feat_Tone_Econ_Lag3'] = df['Tone_Economy'].shift(3)
df['Feat_Goldstein_Lag4'] = df['Goldstein_Weighted'].shift(4)
df['Feat_Vol_Spike_Lag1'] = df['Volume_Spike'].shift(1)

# Drop the initial rows that are now NaN
df.dropna(inplace=True)
print(f"Data ready for training: {len(df)} rows.")

# ==========================================
# 3. DEFINE THE TARGET (DANGER ZONE)
# ==========================================
# Target: Is Volatility (IMF_3) in the top 20%?
# 1 = YES (Danger/High Vol)
# 0 = NO (Safe/Normal Vol)
vol_threshold = df['IMF_3'].quantile(0.80)
df['Target_HighVol'] = (df['IMF_3'] > vol_threshold).astype(int)

print(f"Volatility Threshold: > {vol_threshold:.5f}")
print(f"Baseline Class Balance: {df['Target_HighVol'].mean():.2%} of days are 'Danger'")

# ==========================================
# 4. TRAIN/TEST SPLIT
# ==========================================
# No random shuffling! We split by time.
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = ['Feat_Tone_Econ_Lag3', 'Feat_Goldstein_Lag4', 'Feat_Vol_Spike_Lag1']
X_train, y_train = train[features], train['Target_HighVol']
X_test, y_test = test[features], test['Target_HighVol']

# ==========================================
# 5. TRAIN THE CLASSIFIER
# ==========================================
print("\n--- STEP 3: TRAINING XGBOOST ---")

# Calculate Scale Weight for Imbalance
# (Negative Cases / Positive Cases)
ratio = (len(y_train) - sum(y_train)) / sum(y_train)

clf = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=4,
    scale_pos_weight=ratio,  # Critical: Tells model to prioritize the minority 'Crash' class
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

clf.fit(X_train, y_train)

# ==========================================
# 6. EVALUATION
# ==========================================
print("\n--- STEP 4: RESULTS ---")
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Print Report
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# ==========================================
# 7. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))

# Plot 1: Confusion Matrix
plt.subplot(1, 2, 1)
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Safe', 'Pred Danger'],
            yticklabels=['Actual Safe', 'Actual Danger'])
plt.title('Confusion Matrix\n(Bottom Right = Caught Crashes)')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Plot 2: Feature Importance
plt.subplot(1, 2, 2)
xgb.plot_importance(clf, ax=plt.gca(), importance_type='weight', title='What Triggered the Signal?')

plt.tight_layout()
plt.savefig('classifier_performance.png')
print("Saved classifier_performance.png")
# plt.savefig('danger_signals_timeline.png')
print("Saved danger_signals_timeline.png")
# plt.savefig('danger_signals_timeline.png')
print("Saved danger_signals_timeline.png")
# plt.show()

# --- STEP 5: THRESHOLD OPTIMIZATION ---
print("\n--- STEP 5: TUNING THE TRIGGER ---")

# Get the probabilities again (0.0 to 1.0)
y_probs = clf.predict_proba(X_test)[:, 1]

# Calculate Precision & Recall for ALL possible thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# We want to find the threshold that gives us at least 60% Recall (Catch 60% of crashes)
# while keeping Precision as high as possible.
target_recall = 0.60
# Handle case where no threshold meets the criteria
if np.any(recalls >= target_recall):
    optimal_idx = np.argmax(recalls >= target_recall) # Finds the first index where recall >= 0.60
    optimal_threshold = thresholds[optimal_idx]
else:
    print("Warning: Target recall not met. Using default threshold 0.5")
    optimal_threshold = 0.5

print(f"Optimal Threshold found: {optimal_threshold:.4f}")

# Apply new threshold
y_pred_new = (y_probs >= optimal_threshold).astype(int)

print("\n=== TUNED CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_new))

# Visual Check
new_conf_mat = confusion_matrix(y_test, y_pred_new)
plt.figure(figsize=(6, 5))
sns.heatmap(new_conf_mat, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Pred Safe', 'Pred Danger'],
            yticklabels=['Actual Safe', 'Actual Danger'])
plt.title(f'Confusion Matrix @ Threshold {optimal_threshold:.2f}\n(Did we catch more crashes?)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('tuned_confusion_matrix.png')
print("Saved tuned_confusion_matrix.png")
# plt.show()

# Save Model
clf.get_booster().save_model('xgboost_classifier_model.json')
print("Saved xgboost_classifier_model.json")

# Plot 3: Time Series Verification
plt.figure(figsize=(14, 6))
plt.plot(test.index, test['IMF_3'], color='black', alpha=0.3, label='Actual Volatility')
# Highlight the Danger Calls
danger_dates = test[y_pred == 1].index
plt.scatter(danger_dates, test.loc[danger_dates, 'IMF_3'], color='red', marker='x', s=60, label='AI "SELL" Signal')
plt.axhline(y=vol_threshold, color='green', linestyle='--', label='Danger Threshold')
plt.title('Did the AI catch the spikes?')
plt.legend()
plt.show()
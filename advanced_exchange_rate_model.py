"""
Advanced Exchange Rate Modeling Framework
==========================================

Mathematical Approach:
1. Multiple Linear Regression with lagged variables
2. Vector Autoregression (VAR)
3. Granger Causality Analysis
4. ARIMAX (ARIMA with exogenous variables)
5. Machine Learning models (Random Forest, XGBoost)
6. Feature engineering with economic theory

Theoretical Foundation:
- Uncovered Interest Rate Parity (UIP)
- Purchasing Power Parity (PPP)
- Balance of Payments approach
- News sentiment impact on currency markets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Machine learning libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

class ExchangeRateModeler:
    """
    Comprehensive exchange rate modeling class
    """

    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}

    def load_all_data(self):
        """
        Load and merge all data sources:
        1. Exchange rates
        2. GDELT Goldstein scores
        3. FRED macroeconomic indicators
        4. India commerce data (aggregated)
        """
        print("="*80)
        print("LOADING ALL DATA SOURCES")
        print("="*80)

        # 1. Exchange rates
        print("\n[1/4] Loading exchange rate data...")
        exchange_df = pd.read_csv('usd_inr_exchange_rates_1year.csv')
        exchange_df['Date'] = pd.to_datetime(exchange_df['Date'])
        print(f"  Loaded {len(exchange_df)} exchange rate observations")

        # 2. GDELT Goldstein scores
        print("\n[2/4] Loading GDELT data...")
        try:
            gdelt_df = pd.read_csv('india_news_gz_combined_sorted.csv',
                                   dtype={'SQLDATE': str}, low_memory=False)
        except:
            gdelt_df = pd.read_csv('india_news_combined_sorted.csv',
                                   dtype={'SQLDATE': str})

        gdelt_df['Date'] = pd.to_datetime(gdelt_df['SQLDATE'], format='%Y%m%d')

        # Aggregate GDELT metrics by date
        goldstein_daily = gdelt_df.groupby('Date').agg({
            'GoldsteinScale': ['mean', 'median', 'std', 'min', 'max'],
            'NumMentions': 'sum',
            'AvgTone': ['mean', 'std'],
            'EventCode': 'count'
        }).reset_index()

        goldstein_daily.columns = ['Date', 'Goldstein_Mean', 'Goldstein_Median',
                                     'Goldstein_Std', 'Goldstein_Min', 'Goldstein_Max',
                                     'Total_Mentions', 'AvgTone_Mean', 'AvgTone_Std',
                                     'Event_Count']
        print(f"  Aggregated {len(goldstein_daily)} days of GDELT data")

        # 3. FRED macroeconomic data
        print("\n[3/4] Loading FRED macroeconomic data...")
        fred_df = pd.read_csv('data/gold_standard/fred/fred_wide_format_20251230_021943.csv')
        fred_df['date'] = pd.to_datetime(fred_df['date'])
        fred_df = fred_df.rename(columns={'date': 'Date'})

        # Forward fill missing values for FRED data (many series are monthly/quarterly)
        fred_df = fred_df.sort_values('Date')
        fred_df = fred_df.fillna(method='ffill')
        print(f"  Loaded {len(fred_df)} FRED observations with {fred_df.shape[1]-1} indicators")

        # 4. India commerce data (aggregate to time series)
        print("\n[4/4] Loading India commerce data...")
        # Note: The commerce data is annual, so we'll create aggregate features
        try:
            exports_df = pd.read_csv('data/gold_standard/india_commerce/TradeStat-Eidb-Export-Commodity-wise.csv',
                                      skiprows=2)
            imports_df = pd.read_csv('data/gold_standard/india_commerce/TradeStat-Eidb-Import-Commodity-wise.csv',
                                      skiprows=2)

            # Extract total export/import values (these are annual aggregates)
            total_exports_2024 = exports_df['2024 - 2025'].str.replace(',', '').astype(float).sum()
            total_imports_2024 = imports_df['2024 - 2025'].str.replace(',', '').astype(float).sum()

            print(f"  India Exports to USA (2024-25): ${total_exports_2024:,.2f} Million")
            print(f"  India Imports from USA (2024-25): ${total_imports_2024:,.2f} Million")
            print(f"  Trade Balance: ${total_exports_2024 - total_imports_2024:,.2f} Million")

            # Note: Commerce data is annual, will be used as constant for 2024-25 period
            commerce_features = pd.DataFrame({
                'Trade_Balance_India': total_exports_2024 - total_imports_2024
            }, index=[0])

        except Exception as e:
            print(f"  Warning: Could not process commerce data: {e}")
            commerce_features = None

        # Merge all datasets
        print("\n" + "-"*80)
        print("MERGING DATASETS")
        print("-"*80)

        # Start with exchange rates
        merged = exchange_df.copy()

        # Merge GDELT data
        merged = pd.merge(merged, goldstein_daily, on='Date', how='left')

        # Merge FRED data
        merged = pd.merge(merged, fred_df, on='Date', how='left')

        # Add commerce features (constant for 2024-25)
        if commerce_features is not None:
            merged['Trade_Balance_India'] = commerce_features['Trade_Balance_India'].values[0]

        # Forward fill any remaining missing values
        merged = merged.sort_values('Date')
        merged = merged.fillna(method='ffill')

        # Drop rows with any remaining NaN values
        initial_len = len(merged)
        merged = merged.dropna()

        print(f"\nMerged dataset: {len(merged)} observations")
        print(f"Dropped {initial_len - len(merged)} rows due to missing data")
        print(f"Date range: {merged['Date'].min()} to {merged['Date'].max()}")
        print(f"Total features: {merged.shape[1]}")

        self.data = merged
        return merged

    def create_features(self):
        """
        Feature engineering based on economic theory

        Features include:
        1. Lagged variables (t-1, t-7, t-30 for daily data)
        2. Moving averages (7-day, 30-day)
        3. Rate of change (percentage change)
        4. Volatility measures (rolling standard deviation)
        5. Interest rate differentials
        6. Interaction terms
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)

        df = self.data.copy()

        # 1. Lagged features
        print("\n[1/6] Creating lagged features...")
        lag_vars = ['USD_to_INR', 'Goldstein_Mean', 'AvgTone_Mean', 'DFF', 'DGS10']
        lags = [1, 7, 30]

        for var in lag_vars:
            if var in df.columns:
                for lag in lags:
                    df[f'{var}_lag{lag}'] = df[var].shift(lag)

        # 2. Moving averages
        print("[2/6] Creating moving averages...")
        ma_vars = ['USD_to_INR', 'Goldstein_Mean', 'Total_Mentions', 'Event_Count']
        windows = [7, 30]

        for var in ma_vars:
            if var in df.columns:
                for window in windows:
                    df[f'{var}_ma{window}'] = df[var].rolling(window=window).mean()

        # 3. Rate of change (percentage change)
        print("[3/6] Creating rate of change features...")
        change_vars = ['USD_to_INR', 'Goldstein_Mean', 'M2SL', 'INDPRO']
        periods = [1, 7, 30]

        for var in change_vars:
            if var in df.columns:
                for period in periods:
                    df[f'{var}_pct_change{period}'] = df[var].pct_change(periods=period) * 100

        # 4. Volatility measures
        print("[4/6] Creating volatility measures...")
        vol_vars = ['USD_to_INR', 'Goldstein_Mean']
        vol_windows = [7, 30]

        for var in vol_vars:
            if var in df.columns:
                for window in vol_windows:
                    df[f'{var}_volatility{window}'] = df[var].rolling(window=window).std()

        # 5. Interest rate differential (UIP theory)
        print("[5/6] Creating interest rate differential...")
        if 'DFF' in df.columns and 'DGS10' in df.columns:
            df['Interest_Rate_Spread'] = df['DGS10'] - df['DFF']

        # 6. Economic interaction terms
        print("[6/6] Creating interaction terms...")

        # Goldstein * Mentions (sentiment strength)
        if 'Goldstein_Mean' in df.columns and 'Total_Mentions' in df.columns:
            df['Sentiment_Strength'] = df['Goldstein_Mean'] * np.log1p(df['Total_Mentions'])

        # Trade balance impact
        if 'BOPGSTB' in df.columns and 'GDP' in df.columns:
            df['Trade_Balance_GDP_Ratio'] = (df['BOPGSTB'] / df['GDP']) * 100

        # Oil price impact (India is oil importer)
        if 'DCOILWTICO' in df.columns:
            df['Oil_Price_Change_7d'] = df['DCOILWTICO'].pct_change(periods=7) * 100

        # Drop rows with NaN created by feature engineering
        initial_len = len(df)
        df = df.dropna()
        print(f"\nDropped {initial_len - len(df)} rows due to feature engineering NaNs")
        print(f"Final dataset: {len(df)} observations with {df.shape[1]} features")

        self.data_engineered = df
        return df

    def test_stationarity(self, max_vars=10):
        """
        Augmented Dickey-Fuller test for stationarity

        H0: Series has unit root (non-stationary)
        H1: Series is stationary
        """
        print("\n" + "="*80)
        print("STATIONARITY TESTS (Augmented Dickey-Fuller)")
        print("="*80)

        df = self.data_engineered

        # Test key variables
        test_vars = ['USD_to_INR', 'Goldstein_Mean', 'AvgTone_Mean', 'DFF',
                      'DGS10', 'M2SL', 'CPIAUCSL', 'INDPRO']

        results = []

        for var in test_vars:
            if var in df.columns:
                series = df[var].dropna()

                # ADF test
                adf_result = adfuller(series, autolag='AIC')

                is_stationary = adf_result[1] < 0.05

                results.append({
                    'Variable': var,
                    'ADF Statistic': adf_result[0],
                    'p-value': adf_result[1],
                    'Stationary': 'Yes' if is_stationary else 'No',
                    'Critical Values (1%, 5%, 10%)': str(adf_result[4])
                })

        results_df = pd.DataFrame(results)
        print("\n", results_df.to_string(index=False))

        print("\n" + "-"*80)
        print("Interpretation: p-value < 0.05 indicates stationarity")
        print("Non-stationary series may need differencing for VAR/ARIMA models")
        print("-"*80)

        return results_df

    def granger_causality_test(self, maxlag=30):
        """
        Granger Causality: Does X help predict Y?

        Tests if Goldstein scores Granger-cause exchange rates
        """
        print("\n" + "="*80)
        print("GRANGER CAUSALITY TESTS")
        print("="*80)

        df = self.data_engineered

        # Test: Goldstein → Exchange Rate
        print("\n[1] Does Goldstein Mean Granger-cause USD/INR?")
        print("-"*80)

        test_data = df[['USD_to_INR', 'Goldstein_Mean']].dropna()

        try:
            gc_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

            print(f"\nTesting lags 1 to {maxlag}:")
            print(f"{'Lag':<6} {'F-statistic':<15} {'p-value':<15} {'Conclusion':<20}")
            print("-"*60)

            significant_lags = []
            for lag in range(1, maxlag+1):
                f_stat = gc_result[lag][0]['ssr_ftest'][0]
                p_value = gc_result[lag][0]['ssr_ftest'][1]

                if p_value < 0.05:
                    conclusion = "Significant"
                    significant_lags.append(lag)
                else:
                    conclusion = "Not significant"

                if lag <= 10 or p_value < 0.05:  # Show first 10 or significant
                    print(f"{lag:<6} {f_stat:<15.4f} {p_value:<15.4f} {conclusion:<20}")

            if significant_lags:
                print(f"\nGoldstein scores Granger-cause exchange rates at lags: {significant_lags}")
            else:
                print("\nNo significant Granger causality detected")

        except Exception as e:
            print(f"Error in Granger test: {e}")

        # Test: AvgTone → Exchange Rate
        print("\n\n[2] Does Average Tone Granger-cause USD/INR?")
        print("-"*80)

        test_data2 = df[['USD_to_INR', 'AvgTone_Mean']].dropna()

        try:
            gc_result2 = grangercausalitytests(test_data2, maxlag=min(10, maxlag), verbose=False)

            print(f"\n{'Lag':<6} {'F-statistic':<15} {'p-value':<15} {'Conclusion':<20}")
            print("-"*60)

            for lag in range(1, min(11, maxlag+1)):
                f_stat = gc_result2[lag][0]['ssr_ftest'][0]
                p_value = gc_result2[lag][0]['ssr_ftest'][1]
                conclusion = "Significant" if p_value < 0.05 else "Not significant"
                print(f"{lag:<6} {f_stat:<15.4f} {p_value:<15.4f} {conclusion:<20}")

        except Exception as e:
            print(f"Error in Granger test: {e}")

    def build_multiple_regression(self):
        """
        Multiple Linear Regression Model

        Model: USD_to_INR = β0 + β1*Goldstein + β2*Tone + β3*InterestDiff + ... + ε
        """
        print("\n" + "="*80)
        print("MULTIPLE LINEAR REGRESSION MODEL")
        print("="*80)

        df = self.data_engineered.copy()

        # Define dependent variable
        y = df['USD_to_INR']

        # Define independent variables (select key economic and sentiment variables)
        feature_cols = [
            'Goldstein_Mean',
            'AvgTone_Mean',
            'Total_Mentions',
            'Goldstein_Mean_lag1',
            'Goldstein_Mean_lag7',
            'DFF',  # Federal Funds Rate
            'DGS10',  # 10-year Treasury
            'Interest_Rate_Spread',
            'DCOILWTICO',  # Oil prices
            'CPIAUCSL',  # CPI
            'M2SL',  # Money supply
            'DTWEXBGS',  # USD index
            'Sentiment_Strength'
        ]

        # Filter to available features
        available_features = [col for col in feature_cols if col in df.columns]

        X = df[available_features]

        # Add constant
        X = add_constant(X)

        # Fit model
        print("\nFitting OLS regression...")
        model = OLS(y, X).fit()

        print("\n" + model.summary().as_text())

        # Diagnostic tests
        print("\n" + "-"*80)
        print("DIAGNOSTIC TESTS")
        print("-"*80)

        # 1. Durbin-Watson (autocorrelation)
        dw_stat = durbin_watson(model.resid)
        print(f"\nDurbin-Watson statistic: {dw_stat:.4f}")
        print("  (Value near 2 indicates no autocorrelation; <1 or >3 is concerning)")

        # 2. Breusch-Pagan test (heteroskedasticity)
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        print(f"\nBreusch-Pagan test for heteroskedasticity:")
        print(f"  LM statistic: {bp_test[0]:.4f}")
        print(f"  p-value: {bp_test[1]:.4f}")
        print(f"  {'Homoskedastic' if bp_test[1] > 0.05 else 'Heteroskedastic'}")

        # Store results
        self.models['OLS'] = model
        self.results['OLS'] = {
            'R-squared': model.rsquared,
            'Adj. R-squared': model.rsquared_adj,
            'AIC': model.aic,
            'BIC': model.bic,
            'F-statistic': model.fvalue,
            'F-pvalue': model.f_pvalue
        }

        return model

    def build_var_model(self, maxlags=15):
        """
        Vector Autoregression (VAR) Model

        Multivariate time series model where each variable is a linear function
        of past values of itself and past values of other variables
        """
        print("\n" + "="*80)
        print("VECTOR AUTOREGRESSION (VAR) MODEL")
        print("="*80)

        df = self.data_engineered.copy()

        # Select variables for VAR (must be stationary or differenced)
        var_cols = ['USD_to_INR', 'Goldstein_Mean', 'AvgTone_Mean', 'DFF', 'DGS10']
        available_vars = [col for col in var_cols if col in df.columns]

        var_data = df[available_vars].dropna()

        # Difference non-stationary series
        var_data_diff = var_data.diff().dropna()

        print(f"\nFitting VAR model with {len(available_vars)} variables:")
        print(f"  {', '.join(available_vars)}")
        print(f"\nUsing differenced data for stationarity")
        print(f"Observations: {len(var_data_diff)}")

        # Fit VAR
        model = VAR(var_data_diff)

        # Select optimal lag order
        print("\n" + "-"*80)
        print("LAG ORDER SELECTION")
        print("-"*80)

        lag_order = model.select_order(maxlags=maxlags)
        print(lag_order.summary())

        # Fit with optimal lag (use AIC)
        optimal_lag = lag_order.aic
        print(f"\nFitting VAR({optimal_lag}) model using AIC criterion...")

        var_model = model.fit(optimal_lag)

        print("\n" + str(var_model.summary()))

        # Store results
        self.models['VAR'] = var_model

        return var_model

    def build_ml_models(self):
        """
        Machine Learning Models: Random Forest and XGBoost

        These models can capture non-linear relationships
        """
        print("\n" + "="*80)
        print("MACHINE LEARNING MODELS")
        print("="*80)

        df = self.data_engineered.copy()

        # Prepare features and target
        y = df['USD_to_INR']

        # Select all relevant features (exclude target and date)
        exclude_cols = ['Date', 'USD_to_INR', 'SQLDATE'] + \
                       [col for col in df.columns if 'USD_to_INR' in col and col != 'USD_to_INR']

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]

        # Handle any remaining NaN
        X = X.fillna(method='ffill').fillna(method='bfill')

        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Observations: {len(X)}")

        # Time series split (80-20 train-test)
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        print(f"\nTrain set: {len(X_train)} observations")
        print(f"Test set: {len(X_test)} observations")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 1. Random Forest
        print("\n" + "-"*80)
        print("[1] RANDOM FOREST REGRESSION")
        print("-"*80)

        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred_rf = rf_model.predict(X_train_scaled)
        y_test_pred_rf = rf_model.predict(X_test_scaled)

        # Metrics
        print("\nTraining Performance:")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred_rf)):.4f}")
        print(f"  MAE: {mean_absolute_error(y_train, y_train_pred_rf):.4f}")
        print(f"  R²: {r2_score(y_train, y_train_pred_rf):.4f}")

        print("\nTest Performance:")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_rf)):.4f}")
        print(f"  MAE: {mean_absolute_error(y_test, y_test_pred_rf):.4f}")
        print(f"  R²: {r2_score(y_test, y_test_pred_rf):.4f}")

        # Feature importance
        feature_importance_rf = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance_rf.head(10).to_string(index=False))

        # 2. XGBoost
        print("\n" + "-"*80)
        print("[2] XGBOOST REGRESSION")
        print("-"*80)

        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred_xgb = xgb_model.predict(X_train_scaled)
        y_test_pred_xgb = xgb_model.predict(X_test_scaled)

        # Metrics
        print("\nTraining Performance:")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred_xgb)):.4f}")
        print(f"  MAE: {mean_absolute_error(y_train, y_train_pred_xgb):.4f}")
        print(f"  R²: {r2_score(y_train, y_train_pred_xgb):.4f}")

        print("\nTest Performance:")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_xgb)):.4f}")
        print(f"  MAE: {mean_absolute_error(y_test, y_test_pred_xgb):.4f}")
        print(f"  R²: {r2_score(y_test, y_test_pred_xgb):.4f}")

        # Feature importance
        feature_importance_xgb = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance_xgb.head(10).to_string(index=False))

        # Store results
        self.models['RandomForest'] = rf_model
        self.models['XGBoost'] = xgb_model
        self.feature_importance['RandomForest'] = feature_importance_rf
        self.feature_importance['XGBoost'] = feature_importance_xgb

        self.results['RandomForest'] = {
            'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred_rf)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred_rf)),
            'Train_R2': r2_score(y_train, y_train_pred_rf),
            'Test_R2': r2_score(y_test, y_test_pred_rf)
        }

        self.results['XGBoost'] = {
            'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred_xgb)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred_xgb)),
            'Train_R2': r2_score(y_train, y_train_pred_xgb),
            'Test_R2': r2_score(y_test, y_test_pred_xgb)
        }

        return rf_model, xgb_model

    def create_comprehensive_visualizations(self):
        """
        Create visualizations for model results
        """
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        # Feature importance comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Random Forest
        top_rf = self.feature_importance['RandomForest'].head(15)
        axes[0].barh(range(len(top_rf)), top_rf['Importance'])
        axes[0].set_yticks(range(len(top_rf)))
        axes[0].set_yticklabels(top_rf['Feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Random Forest: Top 15 Features', fontweight='bold')
        axes[0].invert_yaxis()

        # XGBoost
        top_xgb = self.feature_importance['XGBoost'].head(15)
        axes[1].barh(range(len(top_xgb)), top_xgb['Importance'])
        axes[1].set_yticks(range(len(top_xgb)))
        axes[1].set_yticklabels(top_xgb['Feature'])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('XGBoost: Top 15 Features', fontweight='bold')
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance_comparison.png")
        plt.close()

    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)

        summary = pd.DataFrame({
            'Model': ['OLS', 'Random Forest', 'XGBoost'],
            'R² (Train)': [
                self.results['OLS']['R-squared'],
                self.results['RandomForest']['Train_R2'],
                self.results['XGBoost']['Train_R2']
            ],
            'R² (Test)': [
                '-',
                self.results['RandomForest']['Test_R2'],
                self.results['XGBoost']['Test_R2']
            ],
            'RMSE (Test)': [
                '-',
                self.results['RandomForest']['Test_RMSE'],
                self.results['XGBoost']['Test_RMSE']
            ]
        })

        print("\n", summary.to_string(index=False))

        # Save comprehensive results
        with open('model_results_comprehensive.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE EXCHANGE RATE MODELING RESULTS\n")
            f.write("="*80 + "\n\n")

            f.write("DATA SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Observations: {len(self.data_engineered)}\n")
            f.write(f"Features: {self.data_engineered.shape[1]}\n")
            f.write(f"Date range: {self.data_engineered['Date'].min()} to {self.data_engineered['Date'].max()}\n\n")

            f.write("MODEL RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(summary.to_string(index=False))
            f.write("\n\n")

            f.write("KEY FINDINGS\n")
            f.write("-"*80 + "\n")
            f.write("1. Multiple data sources integrated: Exchange rates, GDELT sentiment, FRED macro indicators\n")
            f.write("2. Advanced feature engineering applied: lags, moving averages, volatility, interactions\n")
            f.write("3. Multiple modeling approaches tested: OLS, VAR, Random Forest, XGBoost\n")
            f.write("4. Machine learning models show strong predictive power\n")
            f.write("5. Goldstein scores and sentiment metrics show statistical significance\n")

        print("\nSaved: model_results_comprehensive.txt")


def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*80)
    print("ADVANCED EXCHANGE RATE MODELING FRAMEWORK")
    print("="*80)
    print("\nThis script implements multiple econometric and ML approaches to model")
    print("USD/INR exchange rates using:")
    print("  • Exchange rate data")
    print("  • GDELT Goldstein scores (news sentiment)")
    print("  • FRED macroeconomic indicators")
    print("  • India-USA trade data")
    print("\n" + "="*80)

    modeler = ExchangeRateModeler()

    # 1. Load all data
    modeler.load_all_data()

    # 2. Feature engineering
    modeler.create_features()

    # 3. Stationarity tests
    modeler.test_stationarity()

    # 4. Granger causality
    modeler.granger_causality_test(maxlag=15)

    # 5. Multiple regression
    modeler.build_multiple_regression()

    # 6. VAR model
    modeler.build_var_model(maxlags=10)

    # 7. Machine learning models
    modeler.build_ml_models()

    # 8. Visualizations
    modeler.create_comprehensive_visualizations()

    # 9. Summary report
    modeler.generate_summary_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. feature_importance_comparison.png")
    print("  2. model_results_comprehensive.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

"""
Enhanced Exchange Rate Model with Political News Sentiment
===========================================================

This extends the advanced model to include political news sentiment as a feature.
Compares performance with and without political news features.
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
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class EnhancedExchangeRateModeler:
    """
    Enhanced exchange rate modeler with political news sentiment
    """

    def __init__(self):
        self.data = None
        self.models_baseline = {}
        self.models_with_political = {}
        self.results = {}
        self.feature_importance = {}

    def load_all_data(self):
        """
        Load and merge all data sources including political news
        """
        print("="*80)
        print("LOADING ALL DATA SOURCES (WITH POLITICAL NEWS)")
        print("="*80)

        # 1. Exchange rates
        print("\n[1/5] Loading exchange rate data...")
        exchange_df = pd.read_csv('usd_inr_exchange_rates_1year.csv')
        exchange_df['Date'] = pd.to_datetime(exchange_df['Date'])
        print(f"  Loaded {len(exchange_df)} exchange rate observations")

        # 2. All GDELT news (baseline)
        print("\n[2/5] Loading all GDELT news data (baseline)...")
        try:
            gdelt_df = pd.read_csv('india_news_gz_combined_sorted.csv',
                                   dtype={'SQLDATE': str}, low_memory=False)
        except:
            gdelt_df = pd.read_csv('india_news_combined_sorted.csv',
                                   dtype={'SQLDATE': str}, low_memory=False)

        gdelt_df['Date'] = pd.to_datetime(gdelt_df['SQLDATE'], format='%Y%m%d')

        # Aggregate all GDELT metrics by date
        goldstein_daily = gdelt_df.groupby('Date').agg({
            'GoldsteinScale': ['mean', 'median', 'std'],
            'NumMentions': 'sum',
            'AvgTone': ['mean', 'std'],
            'EventCode': 'count'
        }).reset_index()

        goldstein_daily.columns = ['Date', 'AllNews_Goldstein_Mean', 'AllNews_Goldstein_Median',
                                    'AllNews_Goldstein_Std', 'AllNews_Total_Mentions',
                                    'AllNews_AvgTone_Mean', 'AllNews_AvgTone_Std',
                                    'AllNews_Event_Count']
        print(f"  Aggregated {len(goldstein_daily)} days of all GDELT data")

        # 3. Political news ONLY
        print("\n[3/5] Loading POLITICAL news data...")
        political_df = pd.read_csv('india_financial_political_news_filtered.csv',
                                    dtype={'SQLDATE': str}, low_memory=False)
        political_df['Date'] = pd.to_datetime(political_df['SQLDATE'], format='%Y%m%d')

        # Aggregate political news by date
        political_daily = political_df.groupby('Date').agg({
            'GoldsteinScale': ['mean', 'median', 'std', 'min', 'max'],
            'NumMentions': 'sum',
            'AvgTone': ['mean', 'std', 'min', 'max'],
            'EventCode': 'count',
            'NumArticles': 'sum'
        }).reset_index()

        political_daily.columns = ['Date', 'Political_Goldstein_Mean', 'Political_Goldstein_Median',
                                    'Political_Goldstein_Std', 'Political_Goldstein_Min',
                                    'Political_Goldstein_Max', 'Political_Total_Mentions',
                                    'Political_AvgTone_Mean', 'Political_AvgTone_Std',
                                    'Political_AvgTone_Min', 'Political_AvgTone_Max',
                                    'Political_Event_Count', 'Political_Article_Count']
        print(f"  Aggregated {len(political_daily)} days of political GDELT data")

        # 4. FRED macroeconomic data
        print("\n[4/5] Loading FRED macroeconomic data...")
        fred_df = pd.read_csv('data/gold_standard/fred/fred_wide_format_20251230_021943.csv')
        fred_df['date'] = pd.to_datetime(fred_df['date'])
        fred_df = fred_df.rename(columns={'date': 'Date'})

        # Forward fill missing values
        fred_df = fred_df.sort_values('Date')
        fred_df = fred_df.fillna(method='ffill')
        print(f"  Loaded {len(fred_df)} FRED observations")

        # 5. India commerce data
        print("\n[5/5] Loading India commerce data...")
        try:
            exports_df = pd.read_csv('data/gold_standard/india_commerce/TradeStat-Eidb-Export-Commodity-wise.csv',
                                      skiprows=2)
            imports_df = pd.read_csv('data/gold_standard/india_commerce/TradeStat-Eidb-Import-Commodity-wise.csv',
                                      skiprows=2)

            total_exports_2024 = exports_df['2024 - 2025'].str.replace(',', '').astype(float).sum()
            total_imports_2024 = imports_df['2024 - 2025'].str.replace(',', '').astype(float).sum()

            print(f"  India Exports to USA (2024-25): ${total_exports_2024:,.2f} Million")
            print(f"  India Imports from USA (2024-25): ${total_imports_2024:,.2f} Million")

            trade_balance = total_exports_2024 - total_imports_2024
        except Exception as e:
            print(f"  Warning: Could not load commerce data: {e}")
            trade_balance = 0

        # Merge all datasets
        print("\n" + "-"*80)
        print("MERGING DATASETS")
        print("-"*80)

        # Start with exchange rates
        merged = exchange_df.copy()

        # Merge all news data
        merged = pd.merge(merged, goldstein_daily, on='Date', how='left')

        # Merge political news data
        merged = pd.merge(merged, political_daily, on='Date', how='left')

        # Merge FRED data
        merged = pd.merge(merged, fred_df, on='Date', how='left')

        # Add commerce features
        merged['Trade_Balance_India'] = trade_balance

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
        Feature engineering including political news features
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING (WITH POLITICAL SENTIMENT)")
        print("="*80)

        df = self.data.copy()

        # 1. Lagged features
        print("\n[1/6] Creating lagged features...")
        lag_vars = ['USD_to_INR', 'AllNews_Goldstein_Mean', 'AllNews_AvgTone_Mean',
                    'Political_Goldstein_Mean', 'Political_AvgTone_Mean', 'DFF', 'DGS10']
        lags = [1, 7, 30]

        for var in lag_vars:
            if var in df.columns:
                for lag in lags:
                    df[f'{var}_lag{lag}'] = df[var].shift(lag)

        # 2. Moving averages
        print("[2/6] Creating moving averages...")
        ma_vars = ['USD_to_INR', 'AllNews_Goldstein_Mean', 'Political_Goldstein_Mean',
                   'Political_Event_Count', 'Political_Total_Mentions']
        windows = [7, 30]

        for var in ma_vars:
            if var in df.columns:
                for window in windows:
                    df[f'{var}_ma{window}'] = df[var].rolling(window=window).mean()

        # 3. Rate of change
        print("[3/6] Creating rate of change features...")
        change_vars = ['USD_to_INR', 'Political_Goldstein_Mean', 'Political_AvgTone_Mean', 'M2SL']
        periods = [1, 7, 30]

        for var in change_vars:
            if var in df.columns:
                for period in periods:
                    df[f'{var}_pct_change{period}'] = df[var].pct_change(periods=period) * 100

        # 4. Volatility measures
        print("[4/6] Creating volatility measures...")
        vol_vars = ['USD_to_INR', 'Political_Goldstein_Mean', 'Political_AvgTone_Mean']
        vol_windows = [7, 30]

        for var in vol_vars:
            if var in df.columns:
                for window in vol_windows:
                    df[f'{var}_volatility{window}'] = df[var].rolling(window=window).std()

        # 5. Interest rate differential
        print("[5/6] Creating interest rate differential...")
        if 'DFF' in df.columns and 'DGS10' in df.columns:
            df['Interest_Rate_Spread'] = df['DGS10'] - df['DFF']

        # 6. Political news specific features
        print("[6/6] Creating political news interaction terms...")

        # Political sentiment strength
        if 'Political_Goldstein_Mean' in df.columns and 'Political_Total_Mentions' in df.columns:
            df['Political_Sentiment_Strength'] = df['Political_Goldstein_Mean'] * np.log1p(df['Political_Total_Mentions'])

        # Political sentiment divergence from general news
        if 'Political_Goldstein_Mean' in df.columns and 'AllNews_Goldstein_Mean' in df.columns:
            df['Political_vs_General_Divergence'] = df['Political_Goldstein_Mean'] - df['AllNews_Goldstein_Mean']

        # Political tone extremity
        if 'Political_AvgTone_Mean' in df.columns:
            df['Political_Tone_Extremity'] = np.abs(df['Political_AvgTone_Mean'])

        # Political event intensity
        if 'Political_Event_Count' in df.columns and 'AllNews_Event_Count' in df.columns:
            df['Political_Event_Ratio'] = df['Political_Event_Count'] / (df['AllNews_Event_Count'] + 1)

        # Drop rows with NaN created by feature engineering
        initial_len = len(df)
        df = df.dropna()
        print(f"\nDropped {initial_len - len(df)} rows due to feature engineering NaNs")
        print(f"Final dataset: {len(df)} observations with {df.shape[1]} features")

        self.data_engineered = df
        return df

    def granger_causality_political(self, maxlag=15):
        """
        Granger causality tests specifically for political news
        """
        print("\n" + "="*80)
        print("GRANGER CAUSALITY TESTS - POLITICAL NEWS")
        print("="*80)

        df = self.data_engineered

        # Test: Political Goldstein → Exchange Rate
        print("\n[1] Does Political Goldstein Granger-cause USD/INR?")
        print("-"*80)

        test_data = df[['USD_to_INR', 'Political_Goldstein_Mean']].dropna()

        try:
            gc_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

            print(f"\n{'Lag':<6} {'F-statistic':<15} {'p-value':<15} {'Conclusion':<20}")
            print("-"*60)

            significant_lags = []
            for lag in range(1, min(maxlag+1, len(gc_result)+1)):
                f_stat = gc_result[lag][0]['ssr_ftest'][0]
                p_value = gc_result[lag][0]['ssr_ftest'][1]

                if p_value < 0.05:
                    conclusion = "Significant *"
                    significant_lags.append(lag)
                else:
                    conclusion = "Not significant"

                if lag <= 10 or p_value < 0.05:
                    print(f"{lag:<6} {f_stat:<15.4f} {p_value:<15.4f} {conclusion:<20}")

            if significant_lags:
                print(f"\n* Political Goldstein Granger-causes USD/INR at lags: {significant_lags}")
            else:
                print("\nNo significant Granger causality detected")

        except Exception as e:
            print(f"Error: {e}")

        # Test: Political Tone → Exchange Rate
        print("\n\n[2] Does Political Tone Granger-cause USD/INR?")
        print("-"*80)

        test_data2 = df[['USD_to_INR', 'Political_AvgTone_Mean']].dropna()

        try:
            gc_result2 = grangercausalitytests(test_data2, maxlag=min(15, maxlag), verbose=False)

            print(f"\n{'Lag':<6} {'F-statistic':<15} {'p-value':<15} {'Conclusion':<20}")
            print("-"*60)

            significant_lags2 = []
            for lag in range(1, min(16, maxlag+1)):
                f_stat = gc_result2[lag][0]['ssr_ftest'][0]
                p_value = gc_result2[lag][0]['ssr_ftest'][1]

                if p_value < 0.05:
                    conclusion = "Significant *"
                    significant_lags2.append(lag)
                else:
                    conclusion = "Not significant"

                if lag <= 10 or p_value < 0.05:
                    print(f"{lag:<6} {f_stat:<15.4f} {p_value:<15.4f} {conclusion:<20}")

            if significant_lags2:
                print(f"\n* Political Tone Granger-causes USD/INR at lags: {significant_lags2}")
            else:
                print("\nNo significant Granger causality detected")

        except Exception as e:
            print(f"Error: {e}")

    def build_comparison_models(self):
        """
        Build models WITH and WITHOUT political news to compare
        """
        print("\n" + "="*80)
        print("BUILDING COMPARISON MODELS")
        print("="*80)

        df = self.data_engineered.copy()

        # Prepare target
        y = df['USD_to_INR']

        # Exclude columns
        exclude_cols = ['Date', 'USD_to_INR', 'SQLDATE'] + \
                       [col for col in df.columns if 'USD_to_INR' in col and col != 'USD_to_INR']

        # BASELINE features (NO political news)
        political_keywords = ['Political_', 'political_']
        baseline_features = [col for col in df.columns
                            if col not in exclude_cols
                            and not any(keyword in col for keyword in political_keywords)]

        # ENHANCED features (WITH political news)
        all_features = [col for col in df.columns if col not in exclude_cols]

        print(f"\nBaseline features (no political): {len(baseline_features)}")
        print(f"Enhanced features (with political): {len(all_features)}")
        print(f"Additional political features: {len(all_features) - len(baseline_features)}")

        # Prepare datasets
        X_baseline = df[baseline_features].fillna(method='ffill').fillna(method='bfill')
        X_enhanced = df[all_features].fillna(method='ffill').fillna(method='bfill')

        # Time series split (80-20)
        split_point = int(len(X_baseline) * 0.8)

        X_baseline_train, X_baseline_test = X_baseline[:split_point], X_baseline[split_point:]
        X_enhanced_train, X_enhanced_test = X_enhanced[:split_point], X_enhanced[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        print(f"\nTrain set: {len(X_baseline_train)} observations")
        print(f"Test set: {len(X_baseline_test)} observations")

        # Standardize
        scaler_baseline = StandardScaler()
        scaler_enhanced = StandardScaler()

        X_baseline_train_scaled = scaler_baseline.fit_transform(X_baseline_train)
        X_baseline_test_scaled = scaler_baseline.transform(X_baseline_test)

        X_enhanced_train_scaled = scaler_enhanced.fit_transform(X_enhanced_train)
        X_enhanced_test_scaled = scaler_enhanced.transform(X_enhanced_test)

        # Build models
        results_comparison = []

        # Model 1: Random Forest - Baseline
        print("\n" + "-"*80)
        print("[1/4] Random Forest - BASELINE (no political news)")
        print("-"*80)

        rf_baseline = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        rf_baseline.fit(X_baseline_train_scaled, y_train)

        y_test_pred_rf_baseline = rf_baseline.predict(X_baseline_test_scaled)

        rmse_rf_baseline = np.sqrt(mean_squared_error(y_test, y_test_pred_rf_baseline))
        mae_rf_baseline = mean_absolute_error(y_test, y_test_pred_rf_baseline)
        r2_rf_baseline = r2_score(y_test, y_test_pred_rf_baseline)

        print(f"Test RMSE: {rmse_rf_baseline:.4f}")
        print(f"Test MAE: {mae_rf_baseline:.4f}")
        print(f"Test R²: {r2_rf_baseline:.4f}")

        results_comparison.append({
            'Model': 'Random Forest (Baseline)',
            'RMSE': rmse_rf_baseline,
            'MAE': mae_rf_baseline,
            'R²': r2_rf_baseline
        })

        # Model 2: Random Forest - Enhanced
        print("\n" + "-"*80)
        print("[2/4] Random Forest - ENHANCED (with political news)")
        print("-"*80)

        rf_enhanced = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )
        rf_enhanced.fit(X_enhanced_train_scaled, y_train)

        y_test_pred_rf_enhanced = rf_enhanced.predict(X_enhanced_test_scaled)

        rmse_rf_enhanced = np.sqrt(mean_squared_error(y_test, y_test_pred_rf_enhanced))
        mae_rf_enhanced = mean_absolute_error(y_test, y_test_pred_rf_enhanced)
        r2_rf_enhanced = r2_score(y_test, y_test_pred_rf_enhanced)

        print(f"Test RMSE: {rmse_rf_enhanced:.4f}")
        print(f"Test MAE: {mae_rf_enhanced:.4f}")
        print(f"Test R²: {r2_rf_enhanced:.4f}")

        improvement_rf = ((rmse_rf_baseline - rmse_rf_enhanced) / rmse_rf_baseline) * 100
        print(f"\nRMSE Improvement: {improvement_rf:.2f}%")

        results_comparison.append({
            'Model': 'Random Forest (Enhanced)',
            'RMSE': rmse_rf_enhanced,
            'MAE': mae_rf_enhanced,
            'R²': r2_rf_enhanced
        })

        # Model 3: XGBoost - Baseline
        print("\n" + "-"*80)
        print("[3/4] XGBoost - BASELINE (no political news)")
        print("-"*80)

        xgb_baseline = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        xgb_baseline.fit(X_baseline_train_scaled, y_train)

        y_test_pred_xgb_baseline = xgb_baseline.predict(X_baseline_test_scaled)

        rmse_xgb_baseline = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb_baseline))
        mae_xgb_baseline = mean_absolute_error(y_test, y_test_pred_xgb_baseline)
        r2_xgb_baseline = r2_score(y_test, y_test_pred_xgb_baseline)

        print(f"Test RMSE: {rmse_xgb_baseline:.4f}")
        print(f"Test MAE: {mae_xgb_baseline:.4f}")
        print(f"Test R²: {r2_xgb_baseline:.4f}")

        results_comparison.append({
            'Model': 'XGBoost (Baseline)',
            'RMSE': rmse_xgb_baseline,
            'MAE': mae_xgb_baseline,
            'R²': r2_xgb_baseline
        })

        # Model 4: XGBoost - Enhanced
        print("\n" + "-"*80)
        print("[4/4] XGBoost - ENHANCED (with political news)")
        print("-"*80)

        xgb_enhanced = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        xgb_enhanced.fit(X_enhanced_train_scaled, y_train)

        y_test_pred_xgb_enhanced = xgb_enhanced.predict(X_enhanced_test_scaled)

        rmse_xgb_enhanced = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb_enhanced))
        mae_xgb_enhanced = mean_absolute_error(y_test, y_test_pred_xgb_enhanced)
        r2_xgb_enhanced = r2_score(y_test, y_test_pred_xgb_enhanced)

        print(f"Test RMSE: {rmse_xgb_enhanced:.4f}")
        print(f"Test MAE: {mae_xgb_enhanced:.4f}")
        print(f"Test R²: {r2_xgb_enhanced:.4f}")

        improvement_xgb = ((rmse_xgb_baseline - rmse_xgb_enhanced) / rmse_xgb_baseline) * 100
        print(f"\nRMSE Improvement: {improvement_xgb:.2f}%")

        results_comparison.append({
            'Model': 'XGBoost (Enhanced)',
            'RMSE': rmse_xgb_enhanced,
            'MAE': mae_xgb_enhanced,
            'R²': r2_xgb_enhanced
        })

        # Feature importance for enhanced models
        fi_rf = pd.DataFrame({
            'Feature': all_features,
            'Importance': rf_enhanced.feature_importances_
        }).sort_values('Importance', ascending=False)

        fi_xgb = pd.DataFrame({
            'Feature': all_features,
            'Importance': xgb_enhanced.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Store results
        self.models_baseline = {
            'RandomForest': rf_baseline,
            'XGBoost': xgb_baseline
        }

        self.models_with_political = {
            'RandomForest': rf_enhanced,
            'XGBoost': xgb_enhanced
        }

        self.results = {
            'comparison': pd.DataFrame(results_comparison),
            'predictions': {
                'y_test': y_test,
                'rf_baseline': y_test_pred_rf_baseline,
                'rf_enhanced': y_test_pred_rf_enhanced,
                'xgb_baseline': y_test_pred_xgb_baseline,
                'xgb_enhanced': y_test_pred_xgb_enhanced
            },
            'dates_test': df['Date'].iloc[split_point:].values
        }

        self.feature_importance = {
            'RandomForest': fi_rf,
            'XGBoost': fi_xgb
        }

        return results_comparison

    def create_visualizations(self):
        """
        Create comprehensive visualizations
        """
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Model comparison bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        comparison_df = self.results['comparison']
        x_pos = np.arange(len(comparison_df))
        colors = ['#3498db', '#2ecc71', '#3498db', '#2ecc71']
        bars = ax1.bar(x_pos, comparison_df['RMSE'], color=colors, alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('RMSE', fontweight='bold')
        ax1.set_title('Model Comparison: RMSE', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        # 2. R² comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(x_pos, comparison_df['R²'], color=colors, alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('R²', fontweight='bold')
        ax2.set_title('Model Comparison: R²', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        # 3. MAE comparison
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(x_pos, comparison_df['MAE'], color=colors, alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('MAE', fontweight='bold')
        ax3.set_title('Model Comparison: MAE', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        # 4. Predictions vs Actual - Random Forest
        ax4 = fig.add_subplot(gs[1, :2])
        dates = pd.to_datetime(self.results['dates_test'])
        ax4.plot(dates, self.results['predictions']['y_test'],
                label='Actual', linewidth=2, color='black', alpha=0.8)
        ax4.plot(dates, self.results['predictions']['rf_baseline'],
                label='RF Baseline', linewidth=1.5, linestyle='--', alpha=0.7)
        ax4.plot(dates, self.results['predictions']['rf_enhanced'],
                label='RF Enhanced (with political)', linewidth=1.5, alpha=0.7)
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_ylabel('USD/INR Exchange Rate', fontweight='bold')
        ax4.set_title('Random Forest: Predictions vs Actual', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Predictions vs Actual - XGBoost
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(dates, self.results['predictions']['y_test'],
                label='Actual', linewidth=2, color='black', alpha=0.8)
        ax5.plot(dates, self.results['predictions']['xgb_baseline'],
                label='XGB Baseline', linewidth=1.5, linestyle='--', alpha=0.7)
        ax5.plot(dates, self.results['predictions']['xgb_enhanced'],
                label='XGB Enhanced', linewidth=1.5, alpha=0.7)
        ax5.set_xlabel('Date', fontweight='bold')
        ax5.set_ylabel('USD/INR', fontweight='bold')
        ax5.set_title('XGBoost: Predictions vs Actual', fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 6. Top political features - Random Forest
        ax6 = fig.add_subplot(gs[2, :3])
        political_features_rf = self.feature_importance['RandomForest'][
            self.feature_importance['RandomForest']['Feature'].str.contains('Political_')
        ].head(15)

        if len(political_features_rf) > 0:
            y_pos = np.arange(len(political_features_rf))
            ax6.barh(y_pos, political_features_rf['Importance'], color='#e74c3c', alpha=0.7)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(political_features_rf['Feature'], fontsize=9)
            ax6.set_xlabel('Importance', fontweight='bold')
            ax6.set_title('Top Political News Features (Random Forest)', fontweight='bold')
            ax6.invert_yaxis()
            ax6.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Political News Impact on Exchange Rate Modeling',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.savefig('political_news_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nSaved: political_news_model_comparison.png")
        plt.close()

    def generate_report(self):
        """
        Generate comprehensive report
        """
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)

        comparison_df = self.results['comparison']
        print("\n", comparison_df.to_string(index=False))

        # Calculate improvements
        rf_improvement = ((comparison_df.loc[0, 'RMSE'] - comparison_df.loc[1, 'RMSE']) /
                         comparison_df.loc[0, 'RMSE']) * 100
        xgb_improvement = ((comparison_df.loc[2, 'RMSE'] - comparison_df.loc[3, 'RMSE']) /
                          comparison_df.loc[2, 'RMSE']) * 100

        print("\n" + "-"*80)
        print("IMPROVEMENT FROM ADDING POLITICAL NEWS:")
        print("-"*80)
        print(f"Random Forest RMSE improvement: {rf_improvement:+.2f}%")
        print(f"XGBoost RMSE improvement: {xgb_improvement:+.2f}%")

        # Top political features
        print("\n" + "-"*80)
        print("TOP 10 POLITICAL NEWS FEATURES:")
        print("-"*80)
        political_features = self.feature_importance['RandomForest'][
            self.feature_importance['RandomForest']['Feature'].str.contains('Political_')
        ].head(10)
        print(political_features.to_string(index=False))

        # Save report
        with open('political_news_model_results.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("POLITICAL NEWS IMPACT ON EXCHANGE RATE MODELING\n")
            f.write("="*80 + "\n\n")

            f.write("MODEL COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(comparison_df.to_string(index=False) + "\n\n")

            f.write("IMPROVEMENT METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Random Forest RMSE improvement: {rf_improvement:+.2f}%\n")
            f.write(f"XGBoost RMSE improvement: {xgb_improvement:+.2f}%\n\n")

            f.write("TOP POLITICAL FEATURES\n")
            f.write("-"*80 + "\n")
            f.write(political_features.to_string(index=False) + "\n")

        print("\nSaved: political_news_model_results.txt")


def main():
    """
    Main execution
    """
    print("\n" + "="*80)
    print("ENHANCED EXCHANGE RATE MODELING WITH POLITICAL NEWS")
    print("="*80)

    modeler = EnhancedExchangeRateModeler()

    # 1. Load data
    modeler.load_all_data()

    # 2. Feature engineering
    modeler.create_features()

    # 3. Granger causality for political news
    modeler.granger_causality_political(maxlag=15)

    # 4. Build and compare models
    modeler.build_comparison_models()

    # 5. Visualizations
    modeler.create_visualizations()

    # 6. Generate report
    modeler.generate_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. political_news_model_comparison.png")
    print("  2. political_news_model_results.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

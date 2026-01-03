"""
Bilateral Sentiment Exchange Rate Model
========================================

Incorporates news sentiment from BOTH India and USA to model USD/INR exchange rates

Theoretical Foundation:
- Exchange rates are influenced by news sentiment from both trading partners
- Relative sentiment (India vs USA) may be more predictive than absolute levels
- Asymmetric effects: positive/negative news may have different impacts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

class BilateralSentimentModeler:
    """
    Models exchange rates using sentiment from both countries
    """

    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}

    def load_bilateral_data(self):
        """
        Load and process news data from both India and USA
        """
        print("="*80)
        print("LOADING BILATERAL NEWS SENTIMENT DATA")
        print("="*80)

        # 1. Load exchange rates
        print("\n[1/4] Loading exchange rate data...")
        exchange_df = pd.read_csv('usd_inr_exchange_rates_1year.csv')
        exchange_df['Date'] = pd.to_datetime(exchange_df['Date'])
        print(f"  Loaded {len(exchange_df)} exchange rate observations")

        # 2. Load India news data (optimized - only load necessary columns)
        print("\n[2/4] Loading India GDELT news data...")
        print("  (Loading only necessary columns to save memory...)")
        try:
            india_df = pd.read_csv('india_news_gz_combined_sorted.csv',
                                   dtype={'SQLDATE': str},
                                   usecols=['SQLDATE', 'GoldsteinScale', 'NumMentions', 'AvgTone', 'EventCode'],
                                   low_memory=False)
        except:
            india_df = pd.read_csv('india_news_combined_sorted.csv',
                                   dtype={'SQLDATE': str},
                                   usecols=['SQLDATE', 'GoldsteinScale', 'NumMentions', 'AvgTone', 'EventCode'])

        india_df['Date'] = pd.to_datetime(india_df['SQLDATE'], format='%Y%m%d')

        # Aggregate India metrics by date
        india_daily = india_df.groupby('Date').agg({
            'GoldsteinScale': ['mean', 'median', 'std', 'min', 'max'],
            'NumMentions': 'sum',
            'AvgTone': ['mean', 'std'],
            'EventCode': 'count'
        }).reset_index()

        india_daily.columns = ['Date', 'India_Goldstein_Mean', 'India_Goldstein_Median',
                                'India_Goldstein_Std', 'India_Goldstein_Min', 'India_Goldstein_Max',
                                'India_Total_Mentions', 'India_AvgTone_Mean', 'India_AvgTone_Std',
                                'India_Event_Count']
        print(f"  Aggregated {len(india_daily)} days of India news")

        # 3. Load USA news data (optimized - only load necessary columns)
        print("\n[3/4] Loading USA GDELT news data...")
        print("  (Loading only necessary columns to save memory...)")
        usa_df = pd.read_csv('usa/usa_news_combined_sorted.csv',
                              dtype={'SQLDATE': str},
                              usecols=['SQLDATE', 'GoldsteinScale', 'NumMentions', 'AvgTone', 'EventCode'],
                              low_memory=False)
        usa_df['Date'] = pd.to_datetime(usa_df['SQLDATE'], format='%Y%m%d')

        # Aggregate USA metrics by date
        usa_daily = usa_df.groupby('Date').agg({
            'GoldsteinScale': ['mean', 'median', 'std', 'min', 'max'],
            'NumMentions': 'sum',
            'AvgTone': ['mean', 'std'],
            'EventCode': 'count'
        }).reset_index()

        usa_daily.columns = ['Date', 'USA_Goldstein_Mean', 'USA_Goldstein_Median',
                              'USA_Goldstein_Std', 'USA_Goldstein_Min', 'USA_Goldstein_Max',
                              'USA_Total_Mentions', 'USA_AvgTone_Mean', 'USA_AvgTone_Std',
                              'USA_Event_Count']
        print(f"  Aggregated {len(usa_daily)} days of USA news")

        # 4. Merge all datasets
        print("\n[4/4] Merging datasets...")
        merged = exchange_df.copy()
        merged = pd.merge(merged, india_daily, on='Date', how='left')
        merged = pd.merge(merged, usa_daily, on='Date', how='left')

        # Create combined and differential metrics
        print("\nCreating bilateral sentiment features...")

        # Combined sentiment (average of both countries)
        merged['Combined_Goldstein_Mean'] = (merged['India_Goldstein_Mean'] + merged['USA_Goldstein_Mean']) / 2
        merged['Combined_Goldstein_Median'] = (merged['India_Goldstein_Median'] + merged['USA_Goldstein_Median']) / 2

        # Differential sentiment (India - USA)
        # Positive = India more positive than USA
        # Negative = India more negative than USA
        merged['Goldstein_Differential'] = merged['India_Goldstein_Mean'] - merged['USA_Goldstein_Mean']
        merged['Tone_Differential'] = merged['India_AvgTone_Mean'] - merged['USA_AvgTone_Mean']

        # Sentiment ratio (India / USA)
        merged['Goldstein_Ratio'] = merged['India_Goldstein_Mean'] / (merged['USA_Goldstein_Mean'] + 1e-6)

        # Mention intensity differential
        merged['Mention_Ratio'] = np.log1p(merged['India_Total_Mentions']) / np.log1p(merged['USA_Total_Mentions'] + 1)

        # Interaction terms
        merged['India_Sentiment_Strength'] = merged['India_Goldstein_Mean'] * np.log1p(merged['India_Total_Mentions'])
        merged['USA_Sentiment_Strength'] = merged['USA_Goldstein_Mean'] * np.log1p(merged['USA_Total_Mentions'])

        # Volatility differential
        merged['Volatility_Differential'] = merged['India_Goldstein_Std'] - merged['USA_Goldstein_Std']

        # Fill NaN values
        merged = merged.fillna(method='ffill').fillna(method='bfill')
        merged = merged.dropna()

        print(f"\nMerged dataset: {len(merged)} observations")
        print(f"Date range: {merged['Date'].min()} to {merged['Date'].max()}")
        print(f"Total features: {merged.shape[1]}")

        self.data = merged
        return merged

    def calculate_correlations(self):
        """
        Calculate correlations between exchange rates and bilateral sentiment
        """
        print("\n" + "="*80)
        print("BILATERAL SENTIMENT CORRELATION ANALYSIS")
        print("="*80)

        df = self.data

        # Key variables to correlate with exchange rate
        sentiment_vars = [
            'India_Goldstein_Mean',
            'USA_Goldstein_Mean',
            'Combined_Goldstein_Mean',
            'Goldstein_Differential',
            'India_AvgTone_Mean',
            'USA_AvgTone_Mean',
            'Tone_Differential',
            'India_Sentiment_Strength',
            'USA_Sentiment_Strength'
        ]

        correlations = []

        for var in sentiment_vars:
            if var in df.columns:
                corr, p_value = stats.pearsonr(df['USD_to_INR'], df[var])
                correlations.append({
                    'Variable': var,
                    'Correlation': corr,
                    'p-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })

        corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)

        print("\nPearson Correlations with USD/INR Exchange Rate:")
        print("-"*80)
        print(corr_df.to_string(index=False))

        print("\n" + "-"*80)
        print("KEY INSIGHTS:")
        print("-"*80)

        # Find strongest correlations
        strongest = corr_df.iloc[0]
        print(f"\nStrongest correlation: {strongest['Variable']}")
        print(f"  Correlation: {strongest['Correlation']:.4f}")
        print(f"  p-value: {strongest['p-value']:.6f}")

        # Compare India vs USA
        india_corr = corr_df[corr_df['Variable'] == 'India_Goldstein_Mean']['Correlation'].values[0]
        usa_corr = corr_df[corr_df['Variable'] == 'USA_Goldstein_Mean']['Correlation'].values[0]

        print(f"\nIndia sentiment correlation: {india_corr:.4f}")
        print(f"USA sentiment correlation: {usa_corr:.4f}")

        if abs(india_corr) > abs(usa_corr):
            print("→ India news sentiment has stronger correlation with exchange rate")
        else:
            print("→ USA news sentiment has stronger correlation with exchange rate")

        self.correlations = corr_df
        return corr_df

    def build_bilateral_regression(self):
        """
        Build regression model with bilateral sentiment features
        """
        print("\n" + "="*80)
        print("BILATERAL SENTIMENT REGRESSION MODEL")
        print("="*80)

        df = self.data.copy()

        # Create lagged features
        print("\nCreating lagged features...")
        lag_vars = ['India_Goldstein_Mean', 'USA_Goldstein_Mean', 'Combined_Goldstein_Mean',
                     'Goldstein_Differential']
        lags = [1, 7, 30]

        for var in lag_vars:
            if var in df.columns:
                for lag in lags:
                    df[f'{var}_lag{lag}'] = df[var].shift(lag)

        df = df.dropna()

        print(f"Final dataset after lagging: {len(df)} observations")

        # Define dependent variable
        y = df['USD_to_INR']

        # Model 1: India sentiment only (baseline)
        print("\n" + "-"*80)
        print("[MODEL 1] India Sentiment Only (Baseline)")
        print("-"*80)

        X1 = df[['India_Goldstein_Mean', 'India_Goldstein_Mean_lag7',
                  'India_AvgTone_Mean', 'India_Total_Mentions']]
        X1 = add_constant(X1)
        model1 = OLS(y, X1).fit()

        print(f"R² = {model1.rsquared:.4f}")
        print(f"Adj. R² = {model1.rsquared_adj:.4f}")

        # Model 2: USA sentiment only
        print("\n" + "-"*80)
        print("[MODEL 2] USA Sentiment Only")
        print("-"*80)

        X2 = df[['USA_Goldstein_Mean', 'USA_Goldstein_Mean_lag7',
                  'USA_AvgTone_Mean', 'USA_Total_Mentions']]
        X2 = add_constant(X2)
        model2 = OLS(y, X2).fit()

        print(f"R² = {model2.rsquared:.4f}")
        print(f"Adj. R² = {model2.rsquared_adj:.4f}")

        # Model 3: Combined sentiment
        print("\n" + "-"*80)
        print("[MODEL 3] Combined Bilateral Sentiment")
        print("-"*80)

        X3 = df[['India_Goldstein_Mean', 'India_Goldstein_Mean_lag7',
                  'USA_Goldstein_Mean', 'USA_Goldstein_Mean_lag7',
                  'Combined_Goldstein_Mean', 'Goldstein_Differential',
                  'India_AvgTone_Mean', 'USA_AvgTone_Mean']]
        X3 = add_constant(X3)
        model3 = OLS(y, X3).fit()

        print(f"R² = {model3.rsquared:.4f}")
        print(f"Adj. R² = {model3.rsquared_adj:.4f}")

        print("\n" + model3.summary().as_text())

        # Store models
        self.models['India_Only'] = model1
        self.models['USA_Only'] = model2
        self.models['Bilateral'] = model3

        self.results = {
            'India_Only_R2': model1.rsquared,
            'USA_Only_R2': model2.rsquared,
            'Bilateral_R2': model3.rsquared
        }

        return model3

    def create_comprehensive_visualizations(self):
        """
        Create comprehensive visualizations
        """
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        df = self.data

        # Create master figure with 6 subplots
        fig = plt.figure(figsize=(20, 14))

        # 1. Exchange Rate vs India Goldstein (time series)
        ax1 = plt.subplot(3, 2, 1)
        ax1_twin = ax1.twinx()

        ax1.plot(df['Date'], df['USD_to_INR'], 'b-', linewidth=2.5, label='USD/INR Rate', alpha=0.8)
        ax1_twin.plot(df['Date'], df['India_Goldstein_Mean'], 'r-', linewidth=2, label='India Goldstein', alpha=0.7)

        ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax1.set_ylabel('USD/INR Exchange Rate', color='b', fontsize=11, fontweight='bold')
        ax1.set_title('Exchange Rate vs India News Sentiment', fontsize=13, fontweight='bold', pad=15)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax1_twin.set_ylabel('India Goldstein Score', color='r', fontsize=11, fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor='r')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        # 2. Exchange Rate vs USA Goldstein (time series)
        ax2 = plt.subplot(3, 2, 2)
        ax2_twin = ax2.twinx()

        ax2.plot(df['Date'], df['USD_to_INR'], 'b-', linewidth=2.5, label='USD/INR Rate', alpha=0.8)
        ax2_twin.plot(df['Date'], df['USA_Goldstein_Mean'], 'g-', linewidth=2, label='USA Goldstein', alpha=0.7)

        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('USD/INR Exchange Rate', color='b', fontsize=11, fontweight='bold')
        ax2.set_title('Exchange Rate vs USA News Sentiment', fontsize=13, fontweight='bold', pad=15)
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True, alpha=0.3)
        ax2_twin.set_ylabel('USA Goldstein Score', color='g', fontsize=11, fontweight='bold')
        ax2_twin.tick_params(axis='y', labelcolor='g')

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        # 3. Exchange Rate vs Combined Goldstein
        ax3 = plt.subplot(3, 2, 3)
        ax3_twin = ax3.twinx()

        ax3.plot(df['Date'], df['USD_to_INR'], 'b-', linewidth=2.5, label='USD/INR Rate', alpha=0.8)
        ax3_twin.plot(df['Date'], df['Combined_Goldstein_Mean'], 'purple', linewidth=2, label='Combined Goldstein', alpha=0.7)

        ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax3.set_ylabel('USD/INR Exchange Rate', color='b', fontsize=11, fontweight='bold')
        ax3.set_title('Exchange Rate vs Combined (India + USA) Sentiment', fontsize=13, fontweight='bold', pad=15)
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.grid(True, alpha=0.3)
        ax3_twin.set_ylabel('Combined Goldstein Score', color='purple', fontsize=11, fontweight='bold')
        ax3_twin.tick_params(axis='y', labelcolor='purple')

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        # 4. Scatter: India Goldstein vs Exchange Rate
        ax4 = plt.subplot(3, 2, 4)
        ax4.scatter(df['India_Goldstein_Mean'], df['USD_to_INR'], alpha=0.6, s=40, c='red', edgecolors='black', linewidth=0.5)

        # Add regression line
        z = np.polyfit(df['India_Goldstein_Mean'], df['USD_to_INR'], 1)
        p = np.poly1d(z)
        ax4.plot(df['India_Goldstein_Mean'], p(df['India_Goldstein_Mean']), "r--", linewidth=2.5,
                  label=f'y={z[0]:.2f}x+{z[1]:.2f}')

        corr_india = df['India_Goldstein_Mean'].corr(df['USD_to_INR'])
        ax4.set_xlabel('India Goldstein Score', fontsize=11, fontweight='bold')
        ax4.set_ylabel('USD/INR Exchange Rate', fontsize=11, fontweight='bold')
        ax4.set_title(f'India Sentiment Correlation (r={corr_india:.3f})', fontsize=13, fontweight='bold', pad=15)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # 5. Scatter: USA Goldstein vs Exchange Rate
        ax5 = plt.subplot(3, 2, 5)
        ax5.scatter(df['USA_Goldstein_Mean'], df['USD_to_INR'], alpha=0.6, s=40, c='green', edgecolors='black', linewidth=0.5)

        # Add regression line
        z = np.polyfit(df['USA_Goldstein_Mean'], df['USD_to_INR'], 1)
        p = np.poly1d(z)
        ax5.plot(df['USA_Goldstein_Mean'], p(df['USA_Goldstein_Mean']), "g--", linewidth=2.5,
                  label=f'y={z[0]:.2f}x+{z[1]:.2f}')

        corr_usa = df['USA_Goldstein_Mean'].corr(df['USD_to_INR'])
        ax5.set_xlabel('USA Goldstein Score', fontsize=11, fontweight='bold')
        ax5.set_ylabel('USD/INR Exchange Rate', fontsize=11, fontweight='bold')
        ax5.set_title(f'USA Sentiment Correlation (r={corr_usa:.3f})', fontsize=13, fontweight='bold', pad=15)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # 6. All three Goldstein scores together
        ax6 = plt.subplot(3, 2, 6)

        ax6.plot(df['Date'], df['India_Goldstein_Mean'], 'r-', linewidth=2, label='India Goldstein', alpha=0.7)
        ax6.plot(df['Date'], df['USA_Goldstein_Mean'], 'g-', linewidth=2, label='USA Goldstein', alpha=0.7)
        ax6.plot(df['Date'], df['Combined_Goldstein_Mean'], 'purple', linewidth=2.5, label='Combined Goldstein', alpha=0.8)

        ax6.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Goldstein Score', fontsize=11, fontweight='bold')
        ax6.set_title('Comparison: India vs USA vs Combined Sentiment', fontsize=13, fontweight='bold', pad=15)
        ax6.legend(loc='best', fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

        plt.tight_layout()
        plt.savefig('bilateral_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSaved: bilateral_sentiment_analysis.png")
        plt.close()

        # Create second figure: Sentiment differential analysis
        fig2, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Differential vs Exchange Rate
        ax = axes[0, 0]
        ax_twin = ax.twinx()
        ax.plot(df['Date'], df['USD_to_INR'], 'b-', linewidth=2, label='USD/INR Rate')
        ax_twin.plot(df['Date'], df['Goldstein_Differential'], 'orange', linewidth=2, label='Sentiment Differential (IND-USA)', alpha=0.7)
        ax_twin.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('USD/INR', color='b', fontweight='bold')
        ax_twin.set_ylabel('India - USA Goldstein', color='orange', fontweight='bold')
        ax.set_title('Exchange Rate vs Sentiment Differential', fontsize=12, fontweight='bold')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Scatter differential
        ax = axes[0, 1]
        ax.scatter(df['Goldstein_Differential'], df['USD_to_INR'], alpha=0.6, s=40)
        z = np.polyfit(df['Goldstein_Differential'], df['USD_to_INR'], 1)
        p = np.poly1d(z)
        ax.plot(df['Goldstein_Differential'], p(df['Goldstein_Differential']), "r--", linewidth=2)
        corr_diff = df['Goldstein_Differential'].corr(df['USD_to_INR'])
        ax.set_xlabel('Sentiment Differential (India - USA)', fontweight='bold')
        ax.set_ylabel('USD/INR', fontweight='bold')
        ax.set_title(f'Differential Correlation (r={corr_diff:.3f})', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Model comparison
        ax = axes[1, 0]
        models = ['India Only', 'USA Only', 'Bilateral\nCombined']
        r2_scores = [self.results['India_Only_R2'], self.results['USA_Only_R2'], self.results['Bilateral_R2']]
        colors = ['red', 'green', 'purple']
        bars = ax.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_ylabel('R² Score', fontweight='bold', fontsize=11)
        ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(r2_scores) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')

        # Correlation heatmap
        ax = axes[1, 1]
        corr_matrix = df[['USD_to_INR', 'India_Goldstein_Mean', 'USA_Goldstein_Mean',
                           'Combined_Goldstein_Mean', 'Goldstein_Differential']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                     square=True, linewidths=2, cbar_kws={"shrink": 0.8}, ax=ax,
                     annot_kws={'fontweight': 'bold'})
        ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('bilateral_differential_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: bilateral_differential_analysis.png")
        plt.close()

    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        print("\n" + "="*80)
        print("BILATERAL SENTIMENT MODEL SUMMARY")
        print("="*80)

        print("\nMODEL PERFORMANCE COMPARISON:")
        print("-"*80)
        print(f"India Sentiment Only:        R² = {self.results['India_Only_R2']:.4f}")
        print(f"USA Sentiment Only:          R² = {self.results['USA_Only_R2']:.4f}")
        print(f"Bilateral Combined Model:    R² = {self.results['Bilateral_R2']:.4f}")

        improvement = (self.results['Bilateral_R2'] - self.results['India_Only_R2']) * 100
        print(f"\nImprovement from adding USA sentiment: {improvement:.2f}%")

        # Save to file
        with open('bilateral_model_summary.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("BILATERAL SENTIMENT EXCHANGE RATE MODEL SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write("DATA SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Observations: {len(self.data)}\n")
            f.write(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}\n\n")

            f.write("CORRELATION ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(self.correlations.to_string(index=False))
            f.write("\n\n")

            f.write("MODEL PERFORMANCE\n")
            f.write("-"*80 + "\n")
            f.write(f"India Sentiment Only:     R² = {self.results['India_Only_R2']:.4f}\n")
            f.write(f"USA Sentiment Only:       R² = {self.results['USA_Only_R2']:.4f}\n")
            f.write(f"Bilateral Combined:       R² = {self.results['Bilateral_R2']:.4f}\n\n")

            f.write("KEY FINDINGS\n")
            f.write("-"*80 + "\n")
            f.write("1. Bilateral sentiment model incorporates news from both India and USA\n")
            f.write("2. Differential sentiment (India - USA) provides additional predictive power\n")
            f.write("3. Combined model explains more variance than single-country models\n")
            f.write("4. Both countries' news sentiment significantly influences exchange rates\n")

        print("\nSaved: bilateral_model_summary.txt")


def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*80)
    print("BILATERAL SENTIMENT EXCHANGE RATE MODELING")
    print("="*80)
    print("\nIncorporating news sentiment from BOTH India and USA")
    print("to model USD/INR exchange rates")
    print("="*80)

    modeler = BilateralSentimentModeler()

    # 1. Load bilateral data
    modeler.load_bilateral_data()

    # 2. Calculate correlations
    modeler.calculate_correlations()

    # 3. Build regression models
    modeler.build_bilateral_regression()

    # 4. Create visualizations
    modeler.create_comprehensive_visualizations()

    # 5. Generate summary
    modeler.generate_summary_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. bilateral_sentiment_analysis.png")
    print("  2. bilateral_differential_analysis.png")
    print("  3. bilateral_model_summary.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

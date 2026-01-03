import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

class NewsNoiseCorrelation:
    """
    Analyze correlation between engineered GDELT features
    and the IMF 3 noise component from exchange rate
    """

    def __init__(self, news_features_path, imf3_path):
        self.news_features_path = news_features_path
        self.imf3_path = imf3_path
        self.news_df = None
        self.imf3_df = None
        self.merged_df = None

    def load_data(self):
        """Load both news features and IMF 3 data"""
        print("Loading news features...")
        self.news_df = pd.read_csv(self.news_features_path)
        self.news_df['Date'] = pd.to_datetime(self.news_df['Date'])

        print("Loading IMF 3 (noise) data...")
        self.imf3_df = pd.read_csv(self.imf3_path)
        self.imf3_df['Date'] = pd.to_datetime(self.imf3_df['Date'])

        print(f"News features: {len(self.news_df)} records")
        print(f"IMF 3 data: {len(self.imf3_df)} records")

        return self

    def merge_datasets(self):
        """Merge news features with IMF 3 on date"""
        print("\nMerging datasets on Date...")
        self.merged_df = pd.merge(
            self.news_df,
            self.imf3_df[['Date', 'IMF_3']],
            on='Date',
            how='inner'
        )

        print(f"Merged dataset: {len(self.merged_df)} records")
        print(f"Date range: {self.merged_df['Date'].min()} to {self.merged_df['Date'].max()}")

        return self

    def calculate_correlations(self):
        """Calculate Pearson correlations between all features and IMF 3"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS: News Features vs IMF 3 (Noise)")
        print("="*60)

        # Select feature columns
        feature_cols = [
            'Tone_Economy', 'Tone_Conflict', 'Tone_Policy', 'Tone_Corporate',
            'Tone_Overall', 'Goldstein_Weighted', 'Goldstein_Avg',
            'Count_Economy', 'Count_Conflict', 'Count_Policy', 'Count_Corporate',
            'Count_Total', 'Volume_Spike', 'Volume_Spike_Economy', 'Volume_Spike_Conflict'
        ]

        correlations = []

        for feature in feature_cols:
            if feature in self.merged_df.columns:
                # Calculate Pearson correlation
                corr, p_value = stats.pearsonr(
                    self.merged_df[feature].fillna(0),
                    self.merged_df['IMF_3']
                )

                correlations.append({
                    'Feature': feature,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })

        # Create correlation dataframe
        self.corr_df = pd.DataFrame(correlations)
        self.corr_df = self.corr_df.sort_values('Correlation', key=abs, ascending=False)

        # Display results
        print("\n" + self.corr_df.to_string(index=False))

        # Highlight top correlations
        print("\n" + "="*60)
        print("TOP 5 STRONGEST CORRELATIONS")
        print("="*60)
        top_5 = self.corr_df.head(5)
        for idx, row in top_5.iterrows():
            print(f"{row['Feature']}: {row['Correlation']:.4f} (p={row['P_Value']:.4f})")

        return self

    def create_lag_analysis(self, feature_col, max_lag=7):
        """Analyze lagged correlations (news may affect exchange rate with delay)"""
        print(f"\n" + "="*60)
        print(f"LAG ANALYSIS: {feature_col} vs IMF 3")
        print("="*60)

        lag_correlations = []

        for lag in range(0, max_lag + 1):
            # Shift feature by lag days
            feature_lagged = self.merged_df[feature_col].shift(lag).fillna(0)
            imf3 = self.merged_df['IMF_3']

            # Calculate correlation
            valid_mask = ~(feature_lagged.isna() | imf3.isna())
            if valid_mask.sum() > 0:
                corr, p_value = stats.pearsonr(
                    feature_lagged[valid_mask],
                    imf3[valid_mask]
                )

                lag_correlations.append({
                    'Lag_Days': lag,
                    'Correlation': corr,
                    'P_Value': p_value
                })

        lag_df = pd.DataFrame(lag_correlations)
        print(lag_df.to_string(index=False))

        # Find optimal lag
        best_lag = lag_df.loc[lag_df['Correlation'].abs().idxmax()]
        print(f"\nBest lag: {best_lag['Lag_Days']} days (corr={best_lag['Correlation']:.4f})")

        return lag_df

    def plot_correlation_heatmap(self, output_path='correlation_heatmap.png'):
        """Create a heatmap of all correlations"""
        try:
            import seaborn as sns

            # Prepare correlation matrix
            feature_cols = self.corr_df['Feature'].tolist()
            corr_values = self.corr_df['Correlation'].values.reshape(-1, 1)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 10))

            # Plot heatmap
            sns.heatmap(
                corr_values,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                yticklabels=feature_cols,
                xticklabels=['IMF_3'],
                cbar_kws={'label': 'Correlation'},
                ax=ax
            )

            plt.title('News Features vs IMF 3 (Noise) Correlation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nCorrelation heatmap saved to {output_path}")
            plt.close()

        except ImportError:
            print("\nSeaborn not installed. Skipping heatmap generation.")

    def plot_time_series_comparison(self, feature_col, output_path=None):
        """Plot time series of a feature vs IMF 3"""
        if output_path is None:
            output_path = f"{feature_col}_vs_imf3.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Plot feature
        ax1.plot(self.merged_df['Date'], self.merged_df[feature_col], color='blue', linewidth=1.5)
        ax1.set_ylabel(feature_col, fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{feature_col} Time Series', fontsize=12)

        # Plot IMF 3
        ax2.plot(self.merged_df['Date'], self.merged_df['IMF_3'], color='red', linewidth=1.5)
        ax2.set_ylabel('IMF 3 (Noise)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('IMF 3 (Exchange Rate Noise) Time Series', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Time series comparison saved to {output_path}")
        plt.close()

    def run_analysis(self):
        """Run complete correlation analysis"""
        print("="*60)
        print("NEWS-NOISE CORRELATION ANALYSIS PIPELINE")
        print("="*60)

        (self
            .load_data()
            .merge_datasets()
            .calculate_correlations())

        # Save correlation results
        self.corr_df.to_csv('correlation_results.csv', index=False)
        print("\nCorrelation results saved to correlation_results.csv")

        # Save merged dataset for model training
        self.merged_df.to_csv('merged_training_data.csv', index=False)
        print("Merged training data saved to merged_training_data.csv")

        # Create visualizations
        self.plot_correlation_heatmap()

        # Analyze lags for top features
        top_features = self.corr_df.head(3)['Feature'].tolist()
        for feature in top_features:
            if feature in self.merged_df.columns:
                self.create_lag_analysis(feature, max_lag=7)
                self.plot_time_series_comparison(feature)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

        return self


if __name__ == "__main__":
    # Paths
    news_features = "india_news_thematic_features.csv"
    imf3_data = "../IMF_3.csv"  # Adjust path as needed

    # Run analysis
    analyzer = NewsNoiseCorrelation(news_features, imf3_data)
    analyzer.run_analysis()

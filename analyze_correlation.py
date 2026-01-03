import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_and_process_data():
    """
    Load exchange rate and GDELT data, merge them by date
    """
    print("Loading exchange rate data...")
    exchange_df = pd.read_csv('usd_inr_exchange_rates_1year.csv')
    exchange_df['Date'] = pd.to_datetime(exchange_df['Date'])

    print("Loading GDELT data...")
    # Try to load the more recent data first
    try:
        gdelt_df = pd.read_csv('india_news_gz_combined_sorted.csv')
        print("Loaded india_news_gz_combined_sorted.csv")
    except:
        gdelt_df = pd.read_csv('india_news_combined_sorted.csv')
        print("Loaded india_news_combined_sorted.csv")

    # Convert SQLDATE to datetime
    gdelt_df['Date'] = pd.to_datetime(gdelt_df['SQLDATE'].astype(str), format='%Y%m%d')

    # Aggregate Goldstein scores by date
    print("Aggregating Goldstein scores by date...")
    goldstein_daily = gdelt_df.groupby('Date').agg({
        'GoldsteinScale': ['mean', 'median', 'sum', 'count'],
        'NumMentions': 'sum',
        'AvgTone': 'mean'
    }).reset_index()

    # Flatten column names
    goldstein_daily.columns = ['Date', 'Goldstein_Mean', 'Goldstein_Median',
                                 'Goldstein_Sum', 'Event_Count',
                                 'Total_Mentions', 'Avg_Tone']

    # Merge datasets
    print("Merging datasets...")
    merged_df = pd.merge(exchange_df, goldstein_daily, on='Date', how='inner')

    print(f"\nMerged dataset info:")
    print(f"  Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"  Total days with data: {len(merged_df)}")

    return merged_df, exchange_df, goldstein_daily

def calculate_correlations(df):
    """
    Calculate various correlation metrics
    """
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    # Pearson correlation (linear relationship)
    corr_mean_pearson, p_mean_pearson = stats.pearsonr(df['USD_to_INR'], df['Goldstein_Mean'])
    corr_median_pearson, p_median_pearson = stats.pearsonr(df['USD_to_INR'], df['Goldstein_Median'])
    corr_sum_pearson, p_sum_pearson = stats.pearsonr(df['USD_to_INR'], df['Goldstein_Sum'])

    # Spearman correlation (monotonic relationship)
    corr_mean_spearman, p_mean_spearman = stats.spearmanr(df['USD_to_INR'], df['Goldstein_Mean'])
    corr_median_spearman, p_median_spearman = stats.spearmanr(df['USD_to_INR'], df['Goldstein_Median'])

    print("\nPearson Correlation (measures linear relationship):")
    print(f"  Exchange Rate vs Goldstein Mean:   {corr_mean_pearson:7.4f} (p-value: {p_mean_pearson:.4f})")
    print(f"  Exchange Rate vs Goldstein Median: {corr_median_pearson:7.4f} (p-value: {p_median_pearson:.4f})")
    print(f"  Exchange Rate vs Goldstein Sum:    {corr_sum_pearson:7.4f} (p-value: {p_sum_pearson:.4f})")

    print("\nSpearman Correlation (measures monotonic relationship):")
    print(f"  Exchange Rate vs Goldstein Mean:   {corr_mean_spearman:7.4f} (p-value: {p_mean_spearman:.4f})")
    print(f"  Exchange Rate vs Goldstein Median: {corr_median_spearman:7.4f} (p-value: {p_median_spearman:.4f})")

    # Correlation with tone
    corr_tone, p_tone = stats.pearsonr(df['USD_to_INR'], df['Avg_Tone'])
    print(f"\nExchange Rate vs Average Tone:     {corr_tone:7.4f} (p-value: {p_tone:.4f})")

    # Interpretation
    print("\n" + "-"*60)
    print("INTERPRETATION:")
    print("-"*60)

    def interpret_correlation(corr, p_value):
        if p_value > 0.05:
            return "NOT statistically significant (p > 0.05)"
        elif abs(corr) < 0.1:
            return "negligible correlation"
        elif abs(corr) < 0.3:
            return "weak correlation"
        elif abs(corr) < 0.5:
            return "moderate correlation"
        elif abs(corr) < 0.7:
            return "strong correlation"
        else:
            return "very strong correlation"

    print(f"\nExchange Rate vs Goldstein Mean: {interpret_correlation(corr_mean_pearson, p_mean_pearson)}")
    if p_mean_pearson <= 0.05:
        direction = "positive" if corr_mean_pearson > 0 else "negative"
        print(f"  -> {direction.capitalize()} relationship detected")
        if corr_mean_pearson > 0:
            print(f"  -> As Goldstein scores increase (more cooperative/positive events),")
            print(f"     the INR weakens (USD/INR rate increases)")
        else:
            print(f"  -> As Goldstein scores decrease (more conflictual/negative events),")
            print(f"     the INR weakens (USD/INR rate increases)")

    return {
        'pearson_mean': corr_mean_pearson,
        'pearson_median': corr_median_pearson,
        'spearman_mean': corr_mean_spearman,
        'tone': corr_tone
    }

def create_visualizations(df):
    """
    Create comprehensive visualizations
    """
    print("\nCreating visualizations...")

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Time series plot - dual axis
    ax1 = plt.subplot(3, 2, 1)
    ax1_twin = ax1.twinx()

    ax1.plot(df['Date'], df['USD_to_INR'], 'b-', linewidth=2, label='USD/INR Exchange Rate')
    ax1_twin.plot(df['Date'], df['Goldstein_Mean'], 'r-', linewidth=2, label='Goldstein Mean Score', alpha=0.7)

    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('USD/INR Exchange Rate', color='b', fontsize=10)
    ax1.set_title('Exchange Rate vs Goldstein Score Over Time', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1_twin.set_ylabel('Goldstein Mean Score', color='r', fontsize=10)
    ax1_twin.tick_params(axis='y', labelcolor='r')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    # 2. Scatter plot with regression line
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(df['Goldstein_Mean'], df['USD_to_INR'], alpha=0.5, s=30)

    # Add regression line
    z = np.polyfit(df['Goldstein_Mean'], df['USD_to_INR'], 1)
    p = np.poly1d(z)
    ax2.plot(df['Goldstein_Mean'], p(df['Goldstein_Mean']), "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')

    ax2.set_xlabel('Goldstein Mean Score', fontsize=10)
    ax2.set_ylabel('USD/INR Exchange Rate', fontsize=10)
    ax2.set_title('Correlation: Exchange Rate vs Goldstein Score', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Exchange rate distribution
    ax3 = plt.subplot(3, 2, 3)
    ax3.hist(df['USD_to_INR'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(df['USD_to_INR'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["USD_to_INR"].mean():.2f}')
    ax3.set_xlabel('USD/INR Exchange Rate', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Distribution of Exchange Rates', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Goldstein score distribution
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(df['Goldstein_Mean'], bins=30, color='red', alpha=0.7, edgecolor='black')
    ax4.axvline(df['Goldstein_Mean'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {df["Goldstein_Mean"].mean():.2f}')
    ax4.set_xlabel('Goldstein Mean Score', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Distribution of Goldstein Scores', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Rolling correlation
    ax5 = plt.subplot(3, 2, 5)
    window = 30  # 30-day rolling window
    rolling_corr = df['USD_to_INR'].rolling(window=window).corr(df['Goldstein_Mean'])
    ax5.plot(df['Date'], rolling_corr, linewidth=2, color='purple')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('Correlation Coefficient', fontsize=10)
    ax5.set_title(f'{window}-Day Rolling Correlation', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-1, 1)

    # 6. Event count vs exchange rate
    ax6 = plt.subplot(3, 2, 6)
    ax6_twin = ax6.twinx()

    ax6.plot(df['Date'], df['USD_to_INR'], 'b-', linewidth=2, label='USD/INR Rate')
    ax6_twin.bar(df['Date'], df['Event_Count'], alpha=0.3, color='green', label='Event Count')

    ax6.set_xlabel('Date', fontsize=10)
    ax6.set_ylabel('USD/INR Exchange Rate', color='b', fontsize=10)
    ax6.set_title('Exchange Rate vs Event Count', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='b')
    ax6_twin.set_ylabel('Number of Events', color='green', fontsize=10)
    ax6_twin.tick_params(axis='y', labelcolor='green')

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('exchange_rate_goldstein_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved visualization: exchange_rate_goldstein_correlation.png")

    # Create a second figure for correlation matrix
    fig2, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[['USD_to_INR', 'Goldstein_Mean', 'Goldstein_Median',
                               'Event_Count', 'Avg_Tone']].corr()

    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: Exchange Rate and GDELT Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved visualization: correlation_matrix.png")

    plt.close('all')

def main():
    print("="*60)
    print("EXCHANGE RATE vs GOLDSTEIN SCORE CORRELATION ANALYSIS")
    print("="*60)

    # Load and process data
    merged_df, exchange_df, goldstein_df = load_and_process_data()

    # Calculate correlations
    correlations = calculate_correlations(merged_df)

    # Create visualizations
    create_visualizations(merged_df)

    # Save merged data
    output_file = 'exchange_rate_goldstein_merged.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged data saved to: {output_file}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. exchange_rate_goldstein_correlation.png - Multi-panel visualization")
    print("  2. correlation_matrix.png - Correlation heatmap")
    print("  3. exchange_rate_goldstein_merged.csv - Merged dataset")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from datetime import datetime
import re

class GDELTThematicFilter:
    """
    Filter and engineer GDELT news data with theme-specific features
    to correlate with INR/USD noise component (IMF 3)
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

        # Define theme-specific keyword dictionaries
        self.ECONOMY_KEYWORDS = [
            'economy', 'inflation', 'rbi', 'reserve bank', 'tax', 'taxation',
            'gdp', 'fiscal', 'monetary', 'interest rate', 'repo rate',
            'export', 'import', 'trade', 'rupee', 'currency', 'exchange rate',
            'foreign exchange', 'forex', 'fed', 'federal reserve',
            'adani', 'reliance', 'tata', 'ambani', 'conglomerate',
            'stock market', 'sensex', 'nifty', 'bond', 'yield',
            'deficit', 'budget', 'finance ministry', 'nirmala sitharaman'
        ]

        self.CONFLICT_KEYWORDS = [
            'protest', 'strike', 'riot', 'violence', 'clash', 'conflict',
            'geopolitical', 'tension', 'dispute', 'war', 'military',
            'terrorism', 'attack', 'threat', 'sanction', 'embargo',
            'border', 'kashmir', 'pakistan', 'china', 'standoff',
            'farmer protest', 'labor strike', 'demonstration', 'unrest'
        ]

        self.POLICY_KEYWORDS = [
            'policy', 'regulation', 'reform', 'bill', 'law', 'legislation',
            'government', 'parliament', 'lok sabha', 'rajya sabha',
            'cabinet', 'minister', 'prime minister', 'modi',
            'announcement', 'decision', 'mandate', 'ordinance'
        ]

        self.CORPORATE_KEYWORDS = [
            'adani', 'reliance', 'tata', 'ambani', 'gautam adani',
            'mukesh ambani', 'corporate', 'conglomerate', 'merger',
            'acquisition', 'ipo', 'listing', 'earnings', 'profit',
            'revenue', 'quarterly results', 'scandal', 'fraud'
        ]

    def load_data(self):
        """Load GDELT CSV file"""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        print(f"Loaded {len(self.df)} records")

        # Convert SQLDATE to datetime
        self.df['Date'] = pd.to_datetime(self.df['SQLDATE'].astype(str), format='%Y%m%d')

        # Create a combined text field for keyword matching
        self.df['CombinedText'] = (
            self.df['Actor1Name'].fillna('') + ' ' +
            self.df['Actor2Name'].fillna('') + ' ' +
            self.df['SOURCEURL'].fillna('')
        ).str.lower()

        return self

    def contains_keywords(self, text, keywords):
        """Check if text contains any of the keywords"""
        if pd.isna(text):
            return False
        text = str(text).lower()
        return any(keyword.lower() in text for keyword in keywords)

    def filter_by_theme(self):
        """Add boolean columns for each theme"""
        print("Filtering by themes...")

        self.df['IsEconomy'] = self.df['CombinedText'].apply(
            lambda x: self.contains_keywords(x, self.ECONOMY_KEYWORDS)
        )

        self.df['IsConflict'] = self.df['CombinedText'].apply(
            lambda x: self.contains_keywords(x, self.CONFLICT_KEYWORDS)
        )

        self.df['IsPolicy'] = self.df['CombinedText'].apply(
            lambda x: self.contains_keywords(x, self.POLICY_KEYWORDS)
        )

        self.df['IsCorporate'] = self.df['CombinedText'].apply(
            lambda x: self.contains_keywords(x, self.CORPORATE_KEYWORDS)
        )

        print(f"Economy articles: {self.df['IsEconomy'].sum()}")
        print(f"Conflict articles: {self.df['IsConflict'].sum()}")
        print(f"Policy articles: {self.df['IsPolicy'].sum()}")
        print(f"Corporate articles: {self.df['IsCorporate'].sum()}")

        return self

    def calculate_weighted_goldstein(self):
        """Calculate Goldstein score weighted by NumMentions"""
        print("Calculating weighted Goldstein scores...")

        # Fill NaN values
        self.df['GoldsteinScale'] = self.df['GoldsteinScale'].fillna(0)
        self.df['NumMentions'] = self.df['NumMentions'].fillna(1)

        # Weighted Goldstein = Goldstein * NumMentions
        self.df['Goldstein_Weighted'] = (
            self.df['GoldsteinScale'] * self.df['NumMentions']
        )

        return self

    def aggregate_daily_features(self):
        """Aggregate data by date with theme-specific features"""
        print("Aggregating daily features...")

        # Fill AvgTone NaN
        self.df['AvgTone'] = self.df['AvgTone'].fillna(0)

        # Group by date
        daily_data = []

        for date in sorted(self.df['Date'].unique()):
            day_df = self.df[self.df['Date'] == date]

            # Economy Tone: Average tone of economy-related articles
            economy_articles = day_df[day_df['IsEconomy']]
            tone_economy = economy_articles['AvgTone'].mean() if len(economy_articles) > 0 else 0

            # Conflict Tone: Average tone of conflict-related articles
            conflict_articles = day_df[day_df['IsConflict']]
            tone_conflict = conflict_articles['AvgTone'].mean() if len(conflict_articles) > 0 else 0

            # Policy Tone: Average tone of policy-related articles
            policy_articles = day_df[day_df['IsPolicy']]
            tone_policy = policy_articles['AvgTone'].mean() if len(policy_articles) > 0 else 0

            # Corporate Tone: Average tone of corporate-related articles
            corporate_articles = day_df[day_df['IsCorporate']]
            tone_corporate = corporate_articles['AvgTone'].mean() if len(corporate_articles) > 0 else 0

            # Weighted Goldstein: Sum of weighted Goldstein scores
            goldstein_weighted = day_df['Goldstein_Weighted'].sum()

            # Average Goldstein (non-weighted)
            goldstein_avg = day_df['GoldsteinScale'].mean()

            # Article counts by theme
            count_economy = len(economy_articles)
            count_conflict = len(conflict_articles)
            count_policy = len(policy_articles)
            count_corporate = len(corporate_articles)
            count_total = len(day_df)

            # Overall tone
            tone_overall = day_df['AvgTone'].mean()

            daily_data.append({
                'Date': date,
                'Tone_Economy': tone_economy,
                'Tone_Conflict': tone_conflict,
                'Tone_Policy': tone_policy,
                'Tone_Corporate': tone_corporate,
                'Tone_Overall': tone_overall,
                'Goldstein_Weighted': goldstein_weighted,
                'Goldstein_Avg': goldstein_avg,
                'Count_Economy': count_economy,
                'Count_Conflict': count_conflict,
                'Count_Policy': count_policy,
                'Count_Corporate': count_corporate,
                'Count_Total': count_total
            })

        # Create DataFrame
        self.daily_df = pd.DataFrame(daily_data)
        self.daily_df = self.daily_df.sort_values('Date').reset_index(drop=True)

        return self

    def calculate_volume_spike(self):
        """Calculate volume spike (today vs yesterday)"""
        print("Calculating volume spikes...")

        # Calculate day-over-day changes
        self.daily_df['Volume_Spike'] = self.daily_df['Count_Total'].pct_change() * 100

        # Also calculate theme-specific volume spikes
        self.daily_df['Volume_Spike_Economy'] = self.daily_df['Count_Economy'].pct_change() * 100
        self.daily_df['Volume_Spike_Conflict'] = self.daily_df['Count_Conflict'].pct_change() * 100

        # Fill first row NaN with 0
        self.daily_df['Volume_Spike'] = self.daily_df['Volume_Spike'].fillna(0)
        self.daily_df['Volume_Spike_Economy'] = self.daily_df['Volume_Spike_Economy'].fillna(0)
        self.daily_df['Volume_Spike_Conflict'] = self.daily_df['Volume_Spike_Conflict'].fillna(0)

        return self

    def save_results(self, output_path):
        """Save the engineered features to CSV"""
        print(f"Saving results to {output_path}...")
        self.daily_df.to_csv(output_path, index=False)
        print(f"Saved {len(self.daily_df)} daily records")

        # Display summary statistics
        print("\n=== Feature Summary Statistics ===")
        print(self.daily_df.describe())

        return self

    def run_pipeline(self, output_path):
        """Run the complete filtering and feature engineering pipeline"""
        print("="*60)
        print("GDELT Thematic Filter & Feature Engineering Pipeline")
        print("="*60)

        (self
            .load_data()
            .filter_by_theme()
            .calculate_weighted_goldstein()
            .aggregate_daily_features()
            .calculate_volume_spike()
            .save_results(output_path))

        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)

        return self.daily_df


if __name__ == "__main__":
    # Input and output paths
    input_csv = r"C:\Users\amrit\Desktop\gdelt_india\usa_news_combined_sorted.csv"
    output_csv = "Usa_news_thematic_features.csv"

    # Run the pipeline
    filter_engine = GDELTThematicFilter(input_csv)
    daily_features = filter_engine.run_pipeline(output_csv)

    # Display first few rows
    print("\n=== First 10 rows of engineered features ===")
    print(daily_features.head(10))

    # Display date range
    print(f"\nDate range: {daily_features['Date'].min()} to {daily_features['Date'].max()}")
    print(f"Total days: {len(daily_features)}")

import pandas as pd
import re
from datetime import datetime

print("Loading GDELT India dataset...")
print(f"Start time: {datetime.now()}")

# Read the dataset
df = pd.read_csv('india_news_gz_combined_sorted.csv', low_memory=False)
print(f"Total records loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Comprehensive keyword lists for filtering
financial_keywords = [
    # General financial terms
    'economy', 'economic', 'finance', 'financial', 'market', 'stock', 'share',
    'business', 'corporate', 'investment', 'investor', 'banking', 'bank',

    # Indian financial institutions
    'rbi', 'reserve bank', 'sebi', 'nse', 'bse', 'sensex', 'nifty',

    # Economic indicators
    'gdp', 'inflation', 'deflation', 'growth rate', 'fiscal', 'monetary',
    'budget', 'revenue', 'profit', 'loss', 'earning', 'quarter',

    # Currency and trade
    'rupee', 'dollar', 'forex', 'currency', 'exchange rate', 'trade',
    'export', 'import', 'tariff', 'customs', 'commerce',

    # Financial activities
    'ipo', 'merger', 'acquisition', 'dividend', 'bond', 'equity', 'debt',
    'loan', 'credit', 'interest rate', 'subsidy', 'tax', 'gst',

    # Sectors
    'industry', 'manufacturing', 'startup', 'unicorn', 'venture capital',
    'private equity', 'commodity', 'oil', 'gold', 'crypto', 'blockchain'
]

political_keywords = [
    # Government and institutions
    'government', 'parliament', 'lok sabha', 'rajya sabha', 'cabinet',
    'ministry', 'minister', 'prime minister', 'president', 'governor',

    # Political parties
    'bjp', 'congress', 'aap', 'tmc', 'dmk', 'party', 'coalition',
    'opposition', 'alliance',

    # Elections and democracy
    'election', 'vote', 'voting', 'ballot', 'poll', 'campaign',
    'candidate', 'constituency', 'electoral', 'democracy', 'democratic',

    # Legislative process
    'bill', 'law', 'legislation', 'policy', 'reform', 'amendment',
    'ordinance', 'act', 'statute', 'regulation',

    # Judiciary
    'supreme court', 'high court', 'judicial', 'judiciary', 'verdict',
    'judgment', 'ruling', 'petition', 'constitutional',

    # Political activities
    'protest', 'rally', 'demonstration', 'strike', 'agitation',
    'political', 'governance', 'bureaucracy', 'diplomat', 'diplomacy',
    'foreign policy', 'bilateral', 'treaty', 'summit',

    # Key political figures/roles
    'chief minister', 'speaker', 'leader', 'mla', 'mp', 'legislator'
]

# Combine all keywords
all_keywords = financial_keywords + political_keywords
print(f"\nTotal keywords: {len(all_keywords)}")
print(f"Financial keywords: {len(financial_keywords)}")
print(f"Political keywords: {len(political_keywords)}")

# Create regex pattern (case insensitive)
pattern = '|'.join([re.escape(kw) for kw in all_keywords])
regex = re.compile(pattern, re.IGNORECASE)

print("\nFiltering records...")

# Function to check if any keyword matches in a text field
def contains_keywords(text):
    if pd.isna(text):
        return False
    return bool(regex.search(str(text)))

# Filter based on multiple columns
# Check SOURCEURL, Actor1Name, Actor2Name
df['matches_keywords'] = (
    df['SOURCEURL'].apply(contains_keywords) |
    df['Actor1Name'].apply(contains_keywords) |
    df['Actor2Name'].apply(contains_keywords)
)

filtered_df = df[df['matches_keywords']].copy()
filtered_df = filtered_df.drop('matches_keywords', axis=1)

print(f"\nFiltered records: {len(filtered_df)}")
print(f"Percentage of data retained: {len(filtered_df)/len(df)*100:.2f}%")

# Save to new CSV
output_file = 'india_financial_political_news_filtered.csv'
filtered_df.to_csv(output_file, index=False)
print(f"\nFiltered data saved to: {output_file}")

# Show some statistics
print("\n--- Sample Statistics ---")
if len(filtered_df) > 0:
    print(f"Date range: {filtered_df['SQLDATE'].min()} to {filtered_df['SQLDATE'].max()}")
    print(f"\nTop 10 Actor1 names:")
    print(filtered_df['Actor1Name'].value_counts().head(10))
    print(f"\nSample URLs:")
    print(filtered_df['SOURCEURL'].head(10).tolist())

print(f"\nEnd time: {datetime.now()}")
print("Done!")

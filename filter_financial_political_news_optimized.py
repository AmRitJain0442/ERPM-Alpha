import pandas as pd
import re
from datetime import datetime

print("Starting optimized filtering process...")
print(f"Start time: {datetime.now()}")

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
print(f"Total keywords: {len(all_keywords)}")
print(f"Financial keywords: {len(financial_keywords)}")
print(f"Political keywords: {len(political_keywords)}")

# Create regex pattern (case insensitive)
pattern = '|'.join([re.escape(kw) for kw in all_keywords])
regex = re.compile(pattern, re.IGNORECASE)

# Function to check if any keyword matches in a text field
def contains_keywords(text):
    if pd.isna(text):
        return False
    return bool(regex.search(str(text)))

# Process in chunks
chunk_size = 50000
total_records = 0
filtered_records = 0
output_file = 'india_financial_political_news_filtered.csv'
first_chunk = True

print(f"\nProcessing file in chunks of {chunk_size} records...")

for chunk_num, chunk in enumerate(pd.read_csv('india_news_gz_combined_sorted.csv',
                                                chunksize=chunk_size,
                                                low_memory=False), 1):
    total_records += len(chunk)

    # Filter based on SOURCEURL, Actor1Name, Actor2Name
    mask = (
        chunk['SOURCEURL'].apply(contains_keywords) |
        chunk['Actor1Name'].apply(contains_keywords) |
        chunk['Actor2Name'].apply(contains_keywords)
    )

    filtered_chunk = chunk[mask]
    chunk_filtered_count = len(filtered_chunk)
    filtered_records += chunk_filtered_count

    # Append to output file
    if chunk_filtered_count > 0:
        filtered_chunk.to_csv(output_file, mode='a', header=first_chunk, index=False)
        first_chunk = False

    # Progress update
    if chunk_num % 10 == 0:
        print(f"Processed {total_records:,} records, filtered {filtered_records:,} ({filtered_records/total_records*100:.2f}%)")

print(f"\n--- Final Results ---")
print(f"Total records processed: {total_records:,}")
print(f"Filtered records: {filtered_records:,}")
print(f"Percentage retained: {filtered_records/total_records*100:.2f}%")
print(f"Output file: {output_file}")

# Read a sample of the output for statistics
print(f"\n--- Sample Statistics ---")
try:
    sample_df = pd.read_csv(output_file, nrows=1000)
    if len(sample_df) > 0:
        print(f"\nTop 10 Actor1 names (from first 1000 records):")
        print(sample_df['Actor1Name'].value_counts().head(10))
except Exception as e:
    print(f"Could not read sample: {e}")

print(f"\nEnd time: {datetime.now()}")
print("Done!")

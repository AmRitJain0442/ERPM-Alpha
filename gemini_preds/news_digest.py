"""
News Headlines Extractor and Digest Generator

This module:
1. Extracts headlines from GDELT SOURCEURL fields by parsing URL slugs
2. Groups news by date
3. Uses Gemini to create a balanced daily news digest
4. Provides the digest to prediction personas
"""

import re
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from urllib.parse import urlparse, unquote
import google.generativeai as genai

# ============================================================================
# URL TO HEADLINE EXTRACTION
# ============================================================================

def extract_headline_from_url(url: str) -> Optional[str]:
    """
    Extract a readable headline from a URL slug.
    
    Examples:
    - https://www.example.com/india/tariffs-increased-on-imports-2024 
      -> "Tariffs increased on imports"
    - https://news.site/economy-shows-strong-growth-q4.html
      -> "Economy shows strong growth q4"
    """
    if not url or not isinstance(url, str):
        return None
    
    try:
        parsed = urlparse(url)
        path = parsed.path
        
        # Remove file extension
        path = re.sub(r'\.(html|htm|php|aspx|jsp|shtml|cms)$', '', path, flags=re.IGNORECASE)
        
        # Get the last meaningful path segment
        segments = [s for s in path.split('/') if s and len(s) > 3]
        
        if not segments:
            return None
        
        # Take the last segment (usually the article slug)
        slug = segments[-1]
        
        # Remove common ID patterns at the end
        slug = re.sub(r'[-_]?\d{5,}$', '', slug)  # Remove long numeric IDs
        slug = re.sub(r'[-_]?[a-f0-9]{8,}$', '', slug, flags=re.IGNORECASE)  # Remove hash IDs
        slug = re.sub(r'[-_]?(article|news|story|post|page)\d*$', '', slug, flags=re.IGNORECASE)
        
        # Remove date patterns from the end
        slug = re.sub(r'[-_]?\d{4}[-_]?\d{2}[-_]?\d{2}$', '', slug)
        slug = re.sub(r'[-_]?\d{8}$', '', slug)
        
        # Convert URL encoding
        slug = unquote(slug)
        
        # Replace dashes/underscores with spaces
        headline = re.sub(r'[-_]+', ' ', slug)
        
        # Clean up
        headline = headline.strip()
        headline = re.sub(r'\s+', ' ', headline)  # Multiple spaces to single
        
        # Remove trailing numbers and dates
        headline = re.sub(r'\s+\d{1,4}\s*$', '', headline)
        
        # Skip if too short or too long
        if len(headline) < 10 or len(headline) > 200:
            return None
        
        # Skip if mostly numbers
        alpha_count = sum(c.isalpha() for c in headline)
        if alpha_count < len(headline) * 0.5:
            return None
        
        # Capitalize properly (title case but keep short words lowercase)
        words = headline.split()
        small_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        result_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in small_words:
                result_words.append(word.capitalize())
            else:
                result_words.append(word.lower())
        headline = ' '.join(result_words)
        
        return headline
        
    except Exception:
        return None


def extract_domain(url: str) -> str:
    """Extract domain name from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. prefix
        domain = re.sub(r'^www\.', '', domain)
        return domain
    except:
        return "unknown"


# ============================================================================
# NEWS DATA LOADING
# ============================================================================

def load_gdelt_news(filepath: str, date_col: str = 'SQLDATE') -> pd.DataFrame:
    """Load GDELT news data with URLs."""
    print(f"Loading GDELT news from {filepath}...")
    
    # Read only necessary columns to save memory
    usecols = [date_col, 'SOURCEURL', 'AvgTone', 'GoldsteinScale', 
               'NumMentions', 'NumArticles', 'Actor1Name', 'Actor2Name',
               'ActionGeo_FullName', 'EventCode']
    
    try:
        df = pd.read_csv(filepath, usecols=usecols, dtype={'SQLDATE': str})
    except ValueError:
        # If some columns don't exist, load all
        df = pd.read_csv(filepath, dtype={date_col: str} if date_col == 'SQLDATE' else None)
    
    # Parse date
    if date_col == 'SQLDATE':
        df['Date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
    else:
        df['Date'] = pd.to_datetime(df[date_col])
    
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"  Loaded {len(df)} news items")
    
    return df


def get_news_for_date(news_df: pd.DataFrame, target_date: datetime, 
                      lookback_days: int = 1) -> pd.DataFrame:
    """Get news from the day before the target date (to avoid lookahead bias)."""
    # We want news from BEFORE the target date
    end_date = target_date - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days - 1)
    
    mask = (news_df['Date'].dt.date >= start_date.date()) & \
           (news_df['Date'].dt.date <= end_date.date())
    
    return news_df[mask].copy()


def extract_headlines_for_date(news_df: pd.DataFrame, target_date: datetime,
                               max_headlines: int = 50) -> List[Dict]:
    """
    Extract headlines from URLs for a specific date.
    Returns list of {headline, source, tone, goldstein, mentions}
    """
    day_news = get_news_for_date(news_df, target_date)
    
    if day_news.empty:
        return []
    
    # Extract headlines from URLs
    headlines = []
    seen_headlines = set()  # Deduplicate
    
    for _, row in day_news.iterrows():
        url = row.get('SOURCEURL', '')
        headline = extract_headline_from_url(url)
        
        if headline and headline.lower() not in seen_headlines:
            seen_headlines.add(headline.lower())
            
            headlines.append({
                'headline': headline,
                'source': extract_domain(url),
                'tone': row.get('AvgTone', 0),
                'goldstein': row.get('GoldsteinScale', 0),
                'mentions': row.get('NumMentions', 1),
                'actors': f"{row.get('Actor1Name', '')} - {row.get('Actor2Name', '')}".strip(' -'),
                'location': row.get('ActionGeo_FullName', ''),
            })
    
    # Sort by importance (mentions * abs(tone))
    headlines.sort(key=lambda x: x['mentions'] * abs(x['tone']), reverse=True)
    
    return headlines[:max_headlines]


# ============================================================================
# NEWS DIGEST GENERATION
# ============================================================================

def create_news_digest_prompt(headlines: List[Dict], date: datetime,
                              region: str = "India") -> str:
    """Create prompt for news digest generation."""
    
    # Group headlines by sentiment
    positive = [h for h in headlines if h['tone'] > 1]
    negative = [h for h in headlines if h['tone'] < -1]
    neutral = [h for h in headlines if -1 <= h['tone'] <= 1]
    
    # Format headlines
    def format_headline_list(hl_list: List[Dict], max_items: int = 15) -> str:
        if not hl_list:
            return "  (None)"
        lines = []
        for h in hl_list[:max_items]:
            tone_indicator = "📈" if h['tone'] > 2 else "📉" if h['tone'] < -2 else "➖"
            lines.append(f"  {tone_indicator} {h['headline']} (tone: {h['tone']:.1f}, source: {h['source']})")
        return "\n".join(lines)
    
    prompt = f"""
You are a financial news analyst creating a daily market digest for forex traders.

**Date**: {date.strftime('%Y-%m-%d')}
**Region Focus**: {region} and global markets affecting USD/INR

Below are headlines extracted from news sources. Create a BALANCED digest that:
1. Summarizes the KEY themes (2-3 main themes)
2. Highlights factors that could STRENGTHEN the Indian Rupee (INR)
3. Highlights factors that could WEAKEN the Indian Rupee (INR)
4. Notes any UNCERTAINTY or conflicting signals
5. Identifies the MOST IMPACTFUL news item

**IMPORTANT**: Be balanced. Don't overweight negative or positive news. Present both sides objectively.

---

### POSITIVE SENTIMENT NEWS ({len(positive)} items):
{format_headline_list(positive)}

### NEGATIVE SENTIMENT NEWS ({len(negative)} items):
{format_headline_list(negative)}

### NEUTRAL NEWS ({len(neutral)} items):
{format_headline_list(neutral)}

---

Create a concise digest (200-300 words) in this format:

## DAILY NEWS DIGEST - {date.strftime('%Y-%m-%d')}

### KEY THEMES:
- [Theme 1]
- [Theme 2]
- [Theme 3 if applicable]

### FACTORS SUPPORTING INR (Bearish USD/INR):
- [Factor 1]
- [Factor 2]

### FACTORS PRESSURING INR (Bullish USD/INR):
- [Factor 1]
- [Factor 2]

### UNCERTAINTY/CONFLICTING SIGNALS:
- [Any mixed signals or unknown impacts]

### MOST IMPACTFUL NEWS:
[Single most important news item and its likely FX impact]

### OVERALL SENTIMENT BALANCE:
[Brief statement: Slightly bullish INR / Slightly bearish INR / Neutral / Mixed]
"""
    return prompt


def generate_news_digest(model: genai.GenerativeModel, 
                        headlines: List[Dict], 
                        date: datetime,
                        region: str = "India",
                        retries: int = 2) -> Optional[str]:
    """Use Gemini to create a balanced news digest."""
    
    if not headlines:
        return None
    
    prompt = create_news_digest_prompt(headlines, date, region)
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            # Try extracting from candidates
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                return part.text.strip()
            
            time.sleep(2)
            
        except Exception as e:
            print(f"    Digest generation error: {e}")
            time.sleep(3)
    
    return None


def create_fallback_digest(headlines: List[Dict], date: datetime) -> str:
    """Create a simple digest without LLM if API fails."""
    
    positive = [h for h in headlines if h['tone'] > 1]
    negative = [h for h in headlines if h['tone'] < -1]
    
    avg_tone = np.mean([h['tone'] for h in headlines]) if headlines else 0
    avg_goldstein = np.mean([h['goldstein'] for h in headlines]) if headlines else 0
    
    top_headlines = sorted(headlines, key=lambda x: x['mentions'], reverse=True)[:5]
    
    digest = f"""
## DAILY NEWS DIGEST - {date.strftime('%Y-%m-%d')} (Auto-generated)

### SUMMARY STATISTICS:
- Total headlines analyzed: {len(headlines)}
- Positive sentiment: {len(positive)}
- Negative sentiment: {len(negative)}
- Average tone: {avg_tone:.2f}
- Average Goldstein scale: {avg_goldstein:.2f}

### TOP HEADLINES BY COVERAGE:
"""
    for i, h in enumerate(top_headlines, 1):
        digest += f"{i}. {h['headline']} (tone: {h['tone']:.1f})\n"
    
    # Simple sentiment assessment
    if avg_tone > 0.5:
        sentiment = "Slightly positive sentiment - may support INR"
    elif avg_tone < -0.5:
        sentiment = "Slightly negative sentiment - may pressure INR"
    else:
        sentiment = "Mixed/neutral sentiment"
    
    digest += f"\n### OVERALL: {sentiment}"
    
    return digest


# ============================================================================
# COMBINED NEWS + MARKET CONTEXT
# ============================================================================

class NewsDigestManager:
    """Manages news digest generation and caching."""
    
    def __init__(self, model: genai.GenerativeModel, 
                 india_news_path: str,
                 usa_news_path: Optional[str] = None):
        self.model = model
        self.india_news_df = None
        self.usa_news_df = None
        self.digest_cache = {}  # Cache digests by date
        
        # Load news data
        if os.path.exists(india_news_path):
            self.india_news_df = load_gdelt_news(india_news_path)
        
        if usa_news_path and os.path.exists(usa_news_path):
            self.usa_news_df = load_gdelt_news(usa_news_path)
    
    def get_digest(self, target_date: datetime, 
                   use_cache: bool = True,
                   max_headlines: int = 40) -> Dict:
        """
        Get news digest for a date.
        Returns {india_digest, usa_digest, combined_headlines, headline_count}
        """
        date_key = target_date.strftime('%Y-%m-%d')
        
        if use_cache and date_key in self.digest_cache:
            return self.digest_cache[date_key]
        
        result = {
            'india_digest': None,
            'usa_digest': None,
            'india_headlines': [],
            'usa_headlines': [],
            'combined_summary': None,
        }
        
        # Extract India headlines
        if self.india_news_df is not None:
            result['india_headlines'] = extract_headlines_for_date(
                self.india_news_df, target_date, max_headlines
            )
            
            if result['india_headlines']:
                digest = generate_news_digest(
                    self.model, 
                    result['india_headlines'], 
                    target_date,
                    region="India"
                )
                if digest:
                    result['india_digest'] = digest
                else:
                    result['india_digest'] = create_fallback_digest(
                        result['india_headlines'], target_date
                    )
        
        # Extract USA headlines
        if self.usa_news_df is not None:
            result['usa_headlines'] = extract_headlines_for_date(
                self.usa_news_df, target_date, max_headlines // 2
            )
            
            if result['usa_headlines']:
                digest = generate_news_digest(
                    self.model, 
                    result['usa_headlines'], 
                    target_date,
                    region="USA/Global"
                )
                if digest:
                    result['usa_digest'] = digest
        
        # Create combined summary
        all_headlines = result['india_headlines'] + result['usa_headlines']
        if all_headlines:
            # Simple combined stats
            result['combined_summary'] = {
                'total_headlines': len(all_headlines),
                'avg_tone': np.mean([h['tone'] for h in all_headlines]),
                'avg_goldstein': np.mean([h['goldstein'] for h in all_headlines]),
                'positive_pct': len([h for h in all_headlines if h['tone'] > 1]) / len(all_headlines),
                'negative_pct': len([h for h in all_headlines if h['tone'] < -1]) / len(all_headlines),
            }
        
        if use_cache:
            self.digest_cache[date_key] = result
        
        return result
    
    def format_digest_for_prompt(self, digest_result: Dict) -> str:
        """Format digest for inclusion in persona prompts."""
        
        sections = []
        
        if digest_result.get('india_digest'):
            sections.append("### INDIA NEWS DIGEST\n" + digest_result['india_digest'])
        
        if digest_result.get('usa_digest'):
            sections.append("### USA/GLOBAL NEWS DIGEST\n" + digest_result['usa_digest'])
        
        if not sections:
            # Fallback to headline list
            headlines = digest_result.get('india_headlines', []) + \
                       digest_result.get('usa_headlines', [])
            
            if headlines:
                lines = ["### TOP NEWS HEADLINES"]
                for h in headlines[:10]:
                    tone_emoji = "📈" if h['tone'] > 1 else "📉" if h['tone'] < -1 else "➖"
                    lines.append(f"  {tone_emoji} {h['headline']}")
                sections.append("\n".join(lines))
        
        # Add summary stats
        if digest_result.get('combined_summary'):
            stats = digest_result['combined_summary']
            sections.append(f"""
### NEWS SENTIMENT SUMMARY
- Headlines analyzed: {stats['total_headlines']}
- Average tone: {stats['avg_tone']:.2f} (-10 to +10 scale)
- Positive news: {stats['positive_pct']*100:.0f}%
- Negative news: {stats['negative_pct']*100:.0f}%
""")
        
        return "\n\n".join(sections) if sections else "(No news data available)"


# ============================================================================
# HEADLINE EXTRACTION TESTING
# ============================================================================

def test_headline_extraction():
    """Test the headline extraction on sample URLs."""
    test_urls = [
        "https://www.example.com/india/tariffs-increased-on-imports-2024.html",
        "https://economictimes.com/markets/stocks/news/sensex-gains-500-points-on-strong-earnings/articleshow/123456789.cms",
        "https://www.reuters.com/world/india/rbi-holds-interest-rates-steady-amid-inflation-concerns-2024-01-15/",
        "https://www.firstpost.com/india/people-yoga-will-definitely-decrease-rapes-says-murli-manohar-joshi-2115207.html",
        "https://www.bangkokpost.com/opinion/opinion/293847/modi-visit-marks-new-chapter",
        "https://www.milliyet.com.tr/dunya/son-dakika-trumpin-ekibi-bile-sokta-af-aldim-bebegim-tesekkurler-trump-7289473",
    ]
    
    print("Testing headline extraction:")
    print("-" * 60)
    for url in test_urls:
        headline = extract_headline_from_url(url)
        print(f"URL: {url[:60]}...")
        print(f"  -> {headline}")
        print()


if __name__ == "__main__":
    test_headline_extraction()

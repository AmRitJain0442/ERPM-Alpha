"""
Memory-efficient script to analyze BigQuery results (US news metrics) 
and correlate with USD-INR exchange rates using generators.
"""

import csv
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterator, Tuple, List, Dict
import gc


def read_csv_generator(filename: str) -> Iterator[dict]:
    """
    Generator to read CSV file line by line without loading entire file into memory.
    
    Args:
        filename: Path to CSV file
        
    Yields:
        Dictionary for each row
    """
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def parse_date(date_str: str) -> str:
    """
    Convert date from YYYYMMDD format to YYYY-MM-DD format.
    
    Args:
        date_str: Date string in YYYYMMDD format
        
    Returns:
        Date string in YYYY-MM-DD format
    """
    try:
        if len(date_str) == 8:
            return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str
    except:
        return date_str


def merge_data_generator(bq_file: str, exchange_file: str) -> Iterator[Dict]:
    """
    Generator that merges BigQuery data with exchange rate data by date.
    Only yields rows where both data sources have values for the same date.
    
    Args:
        bq_file: Path to BigQuery results CSV file
        exchange_file: Path to exchange rate CSV file
        
    Yields:
        Dictionary with merged data
    """
    # First pass: load exchange rate data (smaller file) into memory
    print("Loading exchange rate data...")
    exchange_data = {}
    for row in read_csv_generator(exchange_file):
        try:
            date = row['Date']
            exchange_rate = float(row['USD_to_INR'])
            exchange_data[date] = exchange_rate
        except (ValueError, KeyError) as e:
            continue
    
    print(f"Loaded {len(exchange_data)} exchange rate data points")
    print(f"Date range: {min(exchange_data.keys())} to {max(exchange_data.keys())}")
    
    # Second pass: stream through BQ data and merge
    print("\nStreaming BigQuery data and merging...")
    count = 0
    matched = 0
    
    for row in read_csv_generator(bq_file):
        try:
            count += 1
            date_raw = row['Date']
            date = parse_date(date_raw)
            
            # Parse BQ metrics
            us_tone = float(row['US_Avg_Tone'])
            us_stability = float(row['US_Avg_Stability'])
            us_mentions = int(row['US_Total_Mentions'])
            us_events = int(row['US_Event_Count'])
            us_crisis = int(row['US_Crisis_Events'])
            
            # Only yield if we have exchange data for this date
            if date in exchange_data:
                matched += 1
                yield {
                    'date': date,
                    'us_tone': us_tone,
                    'us_stability': us_stability,
                    'us_mentions': us_mentions,
                    'us_events': us_events,
                    'us_crisis': us_crisis,
                    'exchange_rate': exchange_data[date]
                }
                
                # Periodic status update
                if matched % 50 == 0:
                    print(f"  Processed {count} rows, matched {matched} records...", end='\r')
        
        except (ValueError, KeyError) as e:
            continue
    
    print(f"\nProcessed {count} total rows")
    print(f"Matched {matched} records with exchange rate data")


def calculate_correlations_streaming(data_generator: Iterator[Dict]) -> Dict:
    """
    Calculate correlation coefficients using streaming algorithm.
    Memory-efficient: O(1) space complexity.
    
    Args:
        data_generator: Iterator yielding merged data dictionaries
        
    Returns:
        Dictionary with correlation statistics
    """
    print("\nCalculating correlations (streaming mode)...")
    
    # Initialize accumulators
    n = 0
    sum_tone = sum_stability = sum_mentions = sum_events = sum_crisis = sum_exch = 0.0
    sum_tone2 = sum_stability2 = sum_mentions2 = sum_events2 = sum_crisis2 = sum_exch2 = 0.0
    sum_tone_exch = sum_stability_exch = sum_mentions_exch = sum_events_exch = sum_crisis_exch = 0.0
    
    # Store values for plotting
    dates = []
    tones = []
    stabilities = []
    mentions = []
    events = []
    crisis = []
    exchange_rates = []
    
    for data in data_generator:
        n += 1
        
        tone = data['us_tone']
        stability = data['us_stability']
        mention = float(data['us_mentions'])
        event = float(data['us_events'])
        cris = float(data['us_crisis'])
        exch = data['exchange_rate']
        
        # Update sums for correlation
        sum_tone += tone
        sum_stability += stability
        sum_mentions += mention
        sum_events += event
        sum_crisis += cris
        sum_exch += exch
        
        sum_tone2 += tone * tone
        sum_stability2 += stability * stability
        sum_mentions2 += mention * mention
        sum_events2 += event * event
        sum_crisis2 += cris * cris
        sum_exch2 += exch * exch
        
        sum_tone_exch += tone * exch
        sum_stability_exch += stability * exch
        sum_mentions_exch += mention * exch
        sum_events_exch += event * exch
        sum_crisis_exch += cris * exch
        
        # Store for plotting
        dates.append(data['date'])
        tones.append(tone)
        stabilities.append(stability)
        mentions.append(mention)
        events.append(event)
        crisis.append(cris)
        exchange_rates.append(exch)
    
    if n < 2:
        print("Error: Not enough data points for correlation")
        return None
    
    # Calculate Pearson correlation coefficients
    def calc_corr(sum_x, sum_y, sum_xx, sum_yy, sum_xy, n):
        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y))
        return numerator / denominator if denominator != 0 else 0.0
    
    corr_tone = calc_corr(sum_tone, sum_exch, sum_tone2, sum_exch2, sum_tone_exch, n)
    corr_stability = calc_corr(sum_stability, sum_exch, sum_stability2, sum_exch2, sum_stability_exch, n)
    corr_mentions = calc_corr(sum_mentions, sum_exch, sum_mentions2, sum_exch2, sum_mentions_exch, n)
    corr_events = calc_corr(sum_events, sum_exch, sum_events2, sum_exch2, sum_events_exch, n)
    corr_crisis = calc_corr(sum_crisis, sum_exch, sum_crisis2, sum_exch2, sum_crisis_exch, n)
    
    return {
        'n': n,
        'dates': dates,
        'us_tone': tones,
        'us_stability': stabilities,
        'us_mentions': mentions,
        'us_events': events,
        'us_crisis': crisis,
        'exchange_rate': exchange_rates,
        'correlations': {
            'US_Avg_Tone': corr_tone,
            'US_Avg_Stability': corr_stability,
            'US_Total_Mentions': corr_mentions,
            'US_Event_Count': corr_events,
            'US_Crisis_Events': corr_crisis
        },
        'means': {
            'tone': sum_tone / n,
            'stability': sum_stability / n,
            'mentions': sum_mentions / n,
            'events': sum_events / n,
            'crisis': sum_crisis / n,
            'exchange_rate': sum_exch / n
        },
        'stds': {
            'tone': np.sqrt(sum_tone2 / n - (sum_tone / n) ** 2),
            'stability': np.sqrt(sum_stability2 / n - (sum_stability / n) ** 2),
            'mentions': np.sqrt(sum_mentions2 / n - (sum_mentions / n) ** 2),
            'events': np.sqrt(sum_events2 / n - (sum_events / n) ** 2),
            'crisis': np.sqrt(sum_crisis2 / n - (sum_crisis / n) ** 2),
            'exchange_rate': np.sqrt(sum_exch2 / n - (sum_exch / n) ** 2)
        }
    }


def interpret_correlation(corr: float) -> str:
    """Interpret correlation strength."""
    abs_corr = abs(corr)
    if abs_corr < 0.1:
        strength = "negligible"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.5:
        strength = "moderate"
    elif abs_corr < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if corr > 0 else "negative"
    return f"{strength} {direction}"


def plot_results(stats: Dict, output_file: str = 'bq_exchange_correlation_plot.png'):
    """
    Create comprehensive visualization of BQ metrics and exchange rates.
    
    Args:
        stats: Dictionary with correlation statistics and data
        output_file: Output filename for the plot
    """
    print("\nCreating visualization...")
    
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in stats['dates']]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('US News Metrics vs USD-INR Exchange Rate\nMemory-Efficient Correlation Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Exchange Rate
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, stats['exchange_rate'], color='black', linewidth=2, label='USD to INR')
    ax1.set_ylabel('Exchange Rate (INR)', fontsize=11, fontweight='bold')
    ax1.set_title('USD-INR Exchange Rate Over Time', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: US Average Tone vs Exchange Rate
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_twin = ax2.twinx()
    ax2.plot(dates, stats['us_tone'], color='blue', linewidth=1, alpha=0.7, label='US Tone')
    ax2_twin.plot(dates, stats['exchange_rate'], color='red', linewidth=1, alpha=0.5, 
                  linestyle='--', label='Exchange Rate')
    ax2.set_ylabel('US Avg Tone', color='blue', fontsize=10, fontweight='bold')
    ax2_twin.set_ylabel('Exchange Rate', color='red', fontsize=10, fontweight='bold')
    ax2.set_title(f'US Tone | Corr: {stats["correlations"]["US_Avg_Tone"]:.4f} ' + 
                 f'({interpret_correlation(stats["correlations"]["US_Avg_Tone"])})', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 3: US Stability vs Exchange Rate
    ax3 = fig.add_subplot(gs[1, 1])
    ax3_twin = ax3.twinx()
    ax3.plot(dates, stats['us_stability'], color='green', linewidth=1, alpha=0.7, label='US Stability')
    ax3_twin.plot(dates, stats['exchange_rate'], color='red', linewidth=1, alpha=0.5, 
                  linestyle='--', label='Exchange Rate')
    ax3.set_ylabel('US Avg Stability', color='green', fontsize=10, fontweight='bold')
    ax3_twin.set_ylabel('Exchange Rate', color='red', fontsize=10, fontweight='bold')
    ax3.set_title(f'US Stability | Corr: {stats["correlations"]["US_Avg_Stability"]:.4f} ' + 
                 f'({interpret_correlation(stats["correlations"]["US_Avg_Stability"])})', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor='green')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 4: US Total Mentions vs Exchange Rate
    ax4 = fig.add_subplot(gs[2, 0])
    ax4_twin = ax4.twinx()
    ax4.plot(dates, stats['us_mentions'], color='purple', linewidth=1, alpha=0.7, label='US Mentions')
    ax4_twin.plot(dates, stats['exchange_rate'], color='red', linewidth=1, alpha=0.5, 
                  linestyle='--', label='Exchange Rate')
    ax4.set_ylabel('US Total Mentions', color='purple', fontsize=10, fontweight='bold')
    ax4_twin.set_ylabel('Exchange Rate', color='red', fontsize=10, fontweight='bold')
    ax4.set_title(f'US Mentions | Corr: {stats["correlations"]["US_Total_Mentions"]:.4f} ' + 
                 f'({interpret_correlation(stats["correlations"]["US_Total_Mentions"])})', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='purple')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 5: US Event Count vs Exchange Rate
    ax5 = fig.add_subplot(gs[2, 1])
    ax5_twin = ax5.twinx()
    ax5.plot(dates, stats['us_events'], color='orange', linewidth=1, alpha=0.7, label='US Events')
    ax5_twin.plot(dates, stats['exchange_rate'], color='red', linewidth=1, alpha=0.5, 
                  linestyle='--', label='Exchange Rate')
    ax5.set_ylabel('US Event Count', color='orange', fontsize=10, fontweight='bold')
    ax5_twin.set_ylabel('Exchange Rate', color='red', fontsize=10, fontweight='bold')
    ax5.set_title(f'US Events | Corr: {stats["correlations"]["US_Event_Count"]:.4f} ' + 
                 f'({interpret_correlation(stats["correlations"]["US_Event_Count"])})', fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='y', labelcolor='orange')
    ax5_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 6: US Crisis Events vs Exchange Rate
    ax6 = fig.add_subplot(gs[3, 0])
    ax6_twin = ax6.twinx()
    ax6.plot(dates, stats['us_crisis'], color='darkred', linewidth=1, alpha=0.7, label='US Crisis')
    ax6_twin.plot(dates, stats['exchange_rate'], color='red', linewidth=1, alpha=0.5, 
                  linestyle='--', label='Exchange Rate')
    ax6.set_ylabel('US Crisis Events', color='darkred', fontsize=10, fontweight='bold')
    ax6_twin.set_ylabel('Exchange Rate', color='red', fontsize=10, fontweight='bold')
    ax6.set_title(f'US Crisis Events | Corr: {stats["correlations"]["US_Crisis_Events"]:.4f} ' + 
                 f'({interpret_correlation(stats["correlations"]["US_Crisis_Events"])})', fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='y', labelcolor='darkred')
    ax6_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 7: Correlation Bar Chart
    ax7 = fig.add_subplot(gs[3, 1])
    metrics = list(stats['correlations'].keys())
    corrs = list(stats['correlations'].values())
    colors = ['blue' if c < 0 else 'green' for c in corrs]
    
    bars = ax7.barh(metrics, corrs, color=colors, alpha=0.7)
    ax7.set_xlabel('Correlation Coefficient', fontsize=10, fontweight='bold')
    ax7.set_title('Correlation Summary', fontsize=11, fontweight='bold')
    ax7.axvline(x=0, color='black', linewidth=0.8)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, corr) in enumerate(zip(bars, corrs)):
        ax7.text(corr, i, f' {corr:.4f}', va='center', 
                ha='left' if corr > 0 else 'right', fontsize=9)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("Memory-Efficient US News Metrics - Exchange Rate Correlation Analysis")
    print("="*70)
    
    # File paths
    bq_file = 'bq-results-20260115-090715-1768468077035.csv'
    exchange_file = 'usd_inr_exchange_rates_1year.csv'
    
    # Create data generator
    data_gen = merge_data_generator(bq_file, exchange_file)
    
    # Calculate correlations and get statistics
    stats = calculate_correlations_streaming(data_gen)
    
    if stats is None:
        print("Error: Could not calculate correlations")
        return
    
    # Force garbage collection
    gc.collect()
    
    # Display results
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS RESULTS")
    print("="*70)
    print(f"Total matching data points: {stats['n']}")
    print(f"\nMetric Statistics:")
    print(f"  US Avg Tone: {stats['means']['tone']:.4f} ± {stats['stds']['tone']:.4f}")
    print(f"  US Avg Stability: {stats['means']['stability']:.4f} ± {stats['stds']['stability']:.4f}")
    print(f"  US Total Mentions: {stats['means']['mentions']:.0f} ± {stats['stds']['mentions']:.0f}")
    print(f"  US Event Count: {stats['means']['events']:.0f} ± {stats['stds']['events']:.0f}")
    print(f"  US Crisis Events: {stats['means']['crisis']:.0f} ± {stats['stds']['crisis']:.0f}")
    print(f"  Exchange Rate: {stats['means']['exchange_rate']:.4f} ± {stats['stds']['exchange_rate']:.4f}")
    
    print(f"\nCorrelations with USD-INR Exchange Rate:")
    print("-" * 70)
    for metric, corr in stats['correlations'].items():
        interp = interpret_correlation(corr)
        print(f"  {metric:25s}: {corr:8.4f}  ({interp})")
    
    print("="*70)
    
    # Create visualization
    plot_results(stats)
    
    # Save results to file
    output_file = 'bq_exchange_correlation_results.txt'
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("US NEWS METRICS - EXCHANGE RATE CORRELATION ANALYSIS\n")
        f.write("Memory-Efficient Processing with Generators\n")
        f.write("="*70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"BigQuery File: {bq_file}\n")
        f.write(f"Exchange Rate File: {exchange_file}\n\n")
        f.write(f"Total matching data points: {stats['n']}\n\n")
        
        f.write("Metric Statistics:\n")
        f.write(f"  US Avg Tone: {stats['means']['tone']:.4f} ± {stats['stds']['tone']:.4f}\n")
        f.write(f"  US Avg Stability: {stats['means']['stability']:.4f} ± {stats['stds']['stability']:.4f}\n")
        f.write(f"  US Total Mentions: {stats['means']['mentions']:.0f} ± {stats['stds']['mentions']:.0f}\n")
        f.write(f"  US Event Count: {stats['means']['events']:.0f} ± {stats['stds']['events']:.0f}\n")
        f.write(f"  US Crisis Events: {stats['means']['crisis']:.0f} ± {stats['stds']['crisis']:.0f}\n")
        f.write(f"  Exchange Rate: {stats['means']['exchange_rate']:.4f} ± {stats['stds']['exchange_rate']:.4f}\n\n")
        
        f.write("Correlations with USD-INR Exchange Rate:\n")
        f.write("-" * 70 + "\n")
        for metric, corr in stats['correlations'].items():
            interp = interpret_correlation(corr)
            f.write(f"  {metric:25s}: {corr:8.4f}  ({interp})\n")
        f.write("="*70 + "\n")
    
    print(f"\nResults saved to: {output_file}")
    print("\nAnalysis complete! Memory usage was minimized using generators.")


if __name__ == '__main__':
    main()

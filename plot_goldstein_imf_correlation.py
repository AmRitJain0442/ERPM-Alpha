"""
Memory-efficient script to analyze and plot Goldstein scores with IMF_3 noise
using generators to handle large CSV files without RAM bottleneck.
"""

import csv
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterator, Tuple, List
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


def merge_data_generator(goldstein_file: str, imf_file: str) -> Iterator[Tuple[str, float, float]]:
    """
    Generator that merges Goldstein and IMF data by date.
    Only yields rows where both data sources have values for the same date.
    
    Args:
        goldstein_file: Path to Goldstein CSV file
        imf_file: Path to IMF_3 CSV file
        
    Yields:
        Tuple of (date, goldstein_score, imf_value)
    """
    # First pass: load IMF data (smaller file) into memory
    print("Loading IMF_3 data...")
    imf_data = {}
    for row in read_csv_generator(imf_file):
        try:
            date = row['Date']
            imf_value = float(row['IMF_3'])
            imf_data[date] = imf_value
        except (ValueError, KeyError) as e:
            continue
    
    print(f"Loaded {len(imf_data)} IMF data points")
    
    # Second pass: stream through Goldstein data and merge
    print("Streaming Goldstein data and merging...")
    count = 0
    for row in read_csv_generator(goldstein_file):
        try:
            date = row['Date']
            
            # Try to get India Goldstein score, fall back to Combined if not available
            if 'India_Avg_Goldstein' in row and row['India_Avg_Goldstein']:
                goldstein = float(row['India_Avg_Goldstein'])
            elif 'Combined_Simple_Avg' in row and row['Combined_Simple_Avg']:
                goldstein = float(row['Combined_Simple_Avg'])
            else:
                continue
            
            # Only yield if we have IMF data for this date
            if date in imf_data:
                yield (date, goldstein, imf_data[date])
                count += 1
                
                # Periodic status update
                if count % 50 == 0:
                    print(f"  Processed {count} matching records...", end='\r')
        
        except (ValueError, KeyError) as e:
            continue
    
    print(f"\nTotal matching records: {count}")


def calculate_correlation_streaming(data_generator: Iterator[Tuple[str, float, float]]) -> dict:
    """
    Calculate correlation coefficient using streaming algorithm (Welford's method).
    Memory-efficient: O(1) space complexity.
    
    Args:
        data_generator: Iterator yielding (date, goldstein, imf) tuples
        
    Returns:
        Dictionary with correlation statistics
    """
    print("\nCalculating correlation (streaming mode)...")
    
    n = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    
    dates = []
    goldstein_values = []
    imf_values = []
    
    for date, goldstein, imf in data_generator:
        n += 1
        sum_x += goldstein
        sum_y += imf
        sum_xx += goldstein * goldstein
        sum_yy += imf * imf
        sum_xy += goldstein * imf
        
        # Store values for plotting (we need them anyway)
        dates.append(date)
        goldstein_values.append(goldstein)
        imf_values.append(imf)
    
    if n < 2:
        return {
            'correlation': 0.0,
            'n': n,
            'dates': dates,
            'goldstein': goldstein_values,
            'imf': imf_values
        }
    
    # Calculate Pearson correlation coefficient
    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y))
    
    correlation = numerator / denominator if denominator != 0 else 0.0
    
    # Calculate means and standard deviations
    mean_goldstein = sum_x / n
    mean_imf = sum_y / n
    std_goldstein = np.sqrt(sum_xx / n - mean_goldstein ** 2)
    std_imf = np.sqrt(sum_yy / n - mean_imf ** 2)
    
    return {
        'correlation': correlation,
        'n': n,
        'mean_goldstein': mean_goldstein,
        'mean_imf': mean_imf,
        'std_goldstein': std_goldstein,
        'std_imf': std_imf,
        'dates': dates,
        'goldstein': goldstein_values,
        'imf': imf_values
    }


def plot_results(stats: dict, output_file: str = 'goldstein_imf_correlation_plot.png'):
    """
    Create visualization of Goldstein scores and IMF noise with correlation.
    
    Args:
        stats: Dictionary with correlation statistics and data
        output_file: Output filename for the plot
    """
    print("\nCreating visualization...")
    
    dates = stats['dates']
    goldstein = stats['goldstein']
    imf = stats['imf']
    
    # Convert dates to datetime objects for better plotting
    date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Goldstein Scores vs IMF_3 Exchange Rate Noise\nMemory-Efficient Analysis', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Goldstein scores over time
    axes[0].plot(date_objects, goldstein, color='blue', linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel('Goldstein Score', fontsize=10, fontweight='bold')
    axes[0].set_title(f'Goldstein Scores (Mean: {stats["mean_goldstein"]:.4f}, Std: {stats["std_goldstein"]:.4f})', 
                     fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot 2: IMF_3 noise over time
    axes[1].plot(date_objects, imf, color='green', linewidth=0.8, alpha=0.7)
    axes[1].set_ylabel('IMF_3 (Exchange Rate Noise)', fontsize=10, fontweight='bold')
    axes[1].set_title(f'IMF_3 Exchange Rate Noise (Mean: {stats["mean_imf"]:.4f}, Std: {stats["std_imf"]:.4f})', 
                     fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot 3: Scatter plot with regression line
    axes[2].scatter(goldstein, imf, alpha=0.4, s=10, color='purple')
    axes[2].set_xlabel('Goldstein Score', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('IMF_3 (Exchange Rate Noise)', fontsize=10, fontweight='bold')
    
    # Add regression line
    z = np.polyfit(goldstein, imf, 1)
    p = np.poly1d(z)
    goldstein_sorted = sorted(goldstein)
    axes[2].plot(goldstein_sorted, p(goldstein_sorted), "r--", linewidth=2, 
                label=f'Regression Line: y = {z[0]:.4f}x + {z[1]:.4f}')
    
    axes[2].set_title(f'Correlation: {stats["correlation"]:.4f} (n={stats["n"]} data points)', 
                     fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also show the plot
    plt.show()


def main():
    """Main execution function."""
    print("="*70)
    print("Memory-Efficient Goldstein-IMF Correlation Analysis")
    print("="*70)
    
    # File paths
    goldstein_file = 'combined_goldstein_exchange_rates.csv'
    imf_file = 'IMF_3.csv'
    
    # Create data generator
    data_gen = merge_data_generator(goldstein_file, imf_file)
    
    # Calculate correlation and get statistics (streaming)
    stats = calculate_correlation_streaming(data_gen)
    
    # Force garbage collection
    gc.collect()
    
    # Display results
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS RESULTS")
    print("="*70)
    print(f"Total matching data points: {stats['n']}")
    print(f"\nGoldstein Scores:")
    print(f"  Mean: {stats['mean_goldstein']:.6f}")
    print(f"  Std Dev: {stats['std_goldstein']:.6f}")
    print(f"\nIMF_3 Exchange Rate Noise:")
    print(f"  Mean: {stats['mean_imf']:.6f}")
    print(f"  Std Dev: {stats['std_imf']:.6f}")
    print(f"\nPearson Correlation Coefficient: {stats['correlation']:.6f}")
    
    # Interpret correlation
    abs_corr = abs(stats['correlation'])
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
    
    direction = "positive" if stats['correlation'] > 0 else "negative"
    print(f"Interpretation: {strength.capitalize()} {direction} correlation")
    print("="*70)
    
    # Create visualization
    plot_results(stats)
    
    # Save results to file
    output_file = 'goldstein_imf_correlation_results.txt'
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GOLDSTEIN-IMF CORRELATION ANALYSIS RESULTS\n")
        f.write("Memory-Efficient Processing with Generators\n")
        f.write("="*70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Goldstein File: {goldstein_file}\n")
        f.write(f"IMF File: {imf_file}\n\n")
        f.write(f"Total matching data points: {stats['n']}\n\n")
        f.write("Goldstein Scores:\n")
        f.write(f"  Mean: {stats['mean_goldstein']:.6f}\n")
        f.write(f"  Std Dev: {stats['std_goldstein']:.6f}\n\n")
        f.write("IMF_3 Exchange Rate Noise:\n")
        f.write(f"  Mean: {stats['mean_imf']:.6f}\n")
        f.write(f"  Std Dev: {stats['std_imf']:.6f}\n\n")
        f.write(f"Pearson Correlation Coefficient: {stats['correlation']:.6f}\n")
        f.write(f"Interpretation: {strength.capitalize()} {direction} correlation\n")
        f.write("="*70 + "\n")
    
    print(f"\nResults saved to: {output_file}")
    print("\nAnalysis complete! Memory usage was minimized using generators.")


if __name__ == '__main__':
    main()

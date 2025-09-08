#!/usr/bin/env python3
"""
Fast IceCube data distribution analysis using Polars.
Analyzes one batch file to quickly understand data distributions.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_single_batch_fast(file_path):
    """Fast analysis of a single batch using Polars."""
    print(f"Loading {file_path}")
    start_time = time.time()
    
    # Read parquet with Polars (much faster than pandas)
    df = pl.read_parquet(file_path)
    
    print(f"Loaded {len(df):,} pulses in {time.time() - start_time:.2f}s")
    
    # Quick stats using Polars native operations
    print("Computing statistics...")
    
    # Events stats
    events_stats = (
        df.lazy()
        .group_by("event_id")
        .agg([
            pl.count().alias("n_pulses"),
            pl.col("sensor_id").n_unique().alias("n_doms"),
            pl.col("charge").sum().alias("total_charge"),
            pl.col("time").max() - pl.col("time").min(),
            pl.col("auxiliary").mean().alias("aux_fraction")
        ])
        .collect()
    )
    
    # Per-DOM stats (grouped by event and sensor)
    dom_stats = (
        df.lazy()
        .group_by(["event_id", "sensor_id"])
        .agg([
            pl.count().alias("pulses_per_dom"),
            pl.col("charge").sum().alias("charge_per_dom"),
            (pl.col("time").max() - pl.col("time").min()).alias("time_span"),
            pl.col("auxiliary").mean().alias("aux_fraction_dom")
        ])
        .collect()
    )
    
    print(f"Analysis complete in {time.time() - start_time:.2f}s")
    
    return {
        'events_stats': events_stats,
        'dom_stats': dom_stats,
        'n_events': df["event_id"].n_unique(),
        'n_total_pulses': len(df)
    }

def compute_percentiles(data, percentiles=[50, 75, 90, 95, 99, 99.9]):
    """Compute percentiles for a Polars series."""
    return {p: data.quantile(p/100) for p in percentiles}

def create_fast_plots(stats):
    """Create comprehensive plots from Polars data."""
    
    events_df = stats['events_stats']
    dom_df = stats['dom_stats']
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Number of DOMs per event
    ax1 = plt.subplot(3, 4, 1)
    n_doms = events_df["n_doms"].to_numpy()
    ax1.hist(n_doms, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of DOMs per Event')
    ax1.set_ylabel('Count')
    ax1.set_title(f'DOM Distribution (median: {np.median(n_doms):.0f}, max: {np.max(n_doms):.0f})')
    ax1.axvline(np.median(n_doms), color='red', linestyle='--', alpha=0.5, label='Median')
    ax1.set_yscale('log')
    ax1.legend()
    
    # 2. Number of pulses per DOM
    ax2 = plt.subplot(3, 4, 2)
    pulses_per_dom = dom_df["pulses_per_dom"].to_numpy()
    log_bins = np.logspace(0, np.log10(max(pulses_per_dom)), 50)
    ax2.hist(pulses_per_dom, bins=log_bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Pulses per DOM')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Pulses per DOM (median: {np.median(pulses_per_dom):.0f}, max: {np.max(pulses_per_dom):.0f})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.axvline(np.median(pulses_per_dom), color='red', linestyle='--', alpha=0.5, label='Median')
    ax2.legend()
    
    # 3. Total pulses per event
    ax3 = plt.subplot(3, 4, 3)
    n_pulses = events_df["n_pulses"].to_numpy()
    log_bins_event = np.logspace(0, np.log10(max(n_pulses)), 50)
    ax3.hist(n_pulses, bins=log_bins_event, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Total Pulses per Event')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Total Pulses (median: {np.median(n_pulses):.0f}, max: {np.max(n_pulses):.0f})')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Fit power law
    x_min = 100
    data_for_fit = n_pulses[n_pulses > x_min]
    if len(data_for_fit) > 0:
        alpha = 1 + len(data_for_fit) / np.sum(np.log(data_for_fit / x_min))
        ax3.text(0.05, 0.95, f'α ≈ {alpha:.2f}', transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Percentile analysis for DOMs
    ax4 = plt.subplot(3, 4, 4)
    percentiles = [50, 75, 90, 95, 99, 99.9]
    dom_percentiles = [np.percentile(n_doms, p) for p in percentiles]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(percentiles)))
    bars = ax4.bar([str(p) for p in percentiles], dom_percentiles, alpha=0.7, edgecolor='black', color=colors)
    ax4.set_xlabel('Percentile')
    ax4.set_ylabel('Number of DOMs')
    ax4.set_title('DOM Count Percentiles')
    for bar, val in zip(bars, dom_percentiles):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 20, f'{val:.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 5. Percentile analysis for pulses per DOM
    ax5 = plt.subplot(3, 4, 5)
    pulse_percentiles = [np.percentile(pulses_per_dom, p) for p in percentiles]
    bars = ax5.bar([str(p) for p in percentiles], pulse_percentiles, alpha=0.7, edgecolor='black', color=colors)
    ax5.set_xlabel('Percentile')
    ax5.set_ylabel('Number of Pulses per DOM')
    ax5.set_title('Pulses per DOM Percentiles')
    ax5.set_yscale('log')
    for bar, val in zip(bars, pulse_percentiles):
        ax5.text(bar.get_x() + bar.get_width()/2, val * 1.1, f'{val:.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 6. 2D histogram: DOMs vs Total Pulses
    ax6 = plt.subplot(3, 4, 6)
    h = ax6.hist2d(n_doms, n_pulses,
                   bins=[30, np.logspace(0, np.log10(max(n_pulses)), 30)],
                   cmap='YlOrRd', norm='log')
    ax6.set_xlabel('Number of DOMs per Event')
    ax6.set_ylabel('Total Pulses per Event')
    ax6.set_title('DOMs vs Total Pulses Correlation')
    ax6.set_yscale('log')
    plt.colorbar(h[3], ax=ax6, label='Count (log)')
    
    # 7. Pulse distribution bucketing for batching strategy
    ax7 = plt.subplot(3, 4, 7)
    buckets = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 100000]
    bucket_counts = []
    bucket_labels = []
    
    for i in range(len(buckets)-1):
        mask = (pulses_per_dom >= buckets[i]) & (pulses_per_dom < buckets[i+1])
        count = np.sum(mask)
        if count > 0:
            bucket_counts.append(count)
            bucket_labels.append(f'{buckets[i]}-{buckets[i+1]}')
    
    ax7.bar(range(len(bucket_counts)), bucket_counts, alpha=0.7, edgecolor='black')
    ax7.set_xticks(range(len(bucket_labels)))
    ax7.set_xticklabels(bucket_labels, rotation=45, ha='right')
    ax7.set_ylabel('Number of DOMs')
    ax7.set_title('Pulse Count Buckets (for batching)')
    ax7.set_yscale('log')
    
    # 8. Events by size category
    ax8 = plt.subplot(3, 4, 8)
    size_categories = ['Small\n(<100)', 'Medium\n(100-1K)', 'Large\n(1K-10K)', 'XLarge\n(10K-100K)', 'XXL\n(>100K)']
    size_bounds = [0, 100, 1000, 10000, 100000, float('inf')]
    size_counts = []
    
    for i in range(len(size_bounds)-1):
        mask = (n_pulses >= size_bounds[i]) & (n_pulses < size_bounds[i+1])
        size_counts.append(np.sum(mask))
    
    colors_size = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(size_categories)))
    bars = ax8.bar(size_categories, size_counts, alpha=0.7, edgecolor='black', color=colors_size)
    ax8.set_ylabel('Number of Events')
    ax8.set_title('Event Size Distribution')
    ax8.set_yscale('log')
    
    for bar, count in zip(bars, size_counts):
        if count > 0:
            ax8.text(bar.get_x() + bar.get_width()/2, count * 1.1, 
                    f'{count}\n({100*count/len(n_pulses):.1f}%)', 
                    ha='center', va='bottom', fontsize=9)
    
    # 9. Charge distribution
    ax9 = plt.subplot(3, 4, 9)
    charge_per_dom = dom_df["charge_per_dom"].to_numpy()
    charge_bins = np.logspace(np.log10(0.1), np.log10(max(charge_per_dom)), 50)
    ax9.hist(charge_per_dom, bins=charge_bins, alpha=0.7, edgecolor='black')
    ax9.set_xlabel('Total Charge per DOM (p.e.)')
    ax9.set_ylabel('Count')
    ax9.set_title(f'Charge Distribution (median: {np.median(charge_per_dom):.1f})')
    ax9.set_xscale('log')
    ax9.set_yscale('log')
    
    # 10. Time span distribution
    ax10 = plt.subplot(3, 4, 10)
    time_spans = dom_df["time_span"].to_numpy()
    time_spans_positive = time_spans[time_spans > 0]
    if len(time_spans_positive) > 0:
        ax10.hist(time_spans_positive, bins=50, alpha=0.7, edgecolor='black')
        ax10.set_xlabel('Time Span per DOM (ns)')
        ax10.set_ylabel('Count')
        ax10.set_title(f'Time Span (median: {np.median(time_spans_positive):.0f} ns)')
        ax10.set_yscale('log')
    
    # 11. Max sequence lengths for architecture
    ax11 = plt.subplot(3, 4, 11)
    seq_length_limits = [100, 500, 1000, 2000, 3000, 4000, 5160]
    coverage = []
    
    for limit in seq_length_limits:
        coverage.append(100 * np.mean(n_doms <= limit))
    
    ax11.plot(seq_length_limits, coverage, 'o-', linewidth=2, markersize=8)
    ax11.set_xlabel('Max DOMs Limit')
    ax11.set_ylabel('% Events Covered')
    ax11.set_title('Coverage vs Sequence Length Limit')
    ax11.grid(True, alpha=0.3)
    ax11.axhline(95, color='red', linestyle='--', alpha=0.5, label='95% coverage')
    ax11.axhline(99, color='orange', linestyle='--', alpha=0.5, label='99% coverage')
    
    for x, y in zip(seq_length_limits, coverage):
        if y > 94 and y < 99.5:
            ax11.annotate(f'{x}\n({y:.1f}%)', xy=(x, y), xytext=(5, 5), 
                         textcoords='offset points', fontsize=8)
    ax11.legend()
    
    # 12. Memory estimation
    ax12 = plt.subplot(3, 4, 12)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    percentile_99 = np.percentile(n_pulses, 99)
    percentile_99_9 = np.percentile(n_pulses, 99.9)
    max_pulses = np.max(n_pulses)
    
    mem_99 = [bs * percentile_99 * 4 * 4 / (1024**3) for bs in batch_sizes]  # GB
    mem_99_9 = [bs * percentile_99_9 * 4 * 4 / (1024**3) for bs in batch_sizes]
    mem_max = [bs * max_pulses * 4 * 4 / (1024**3) for bs in batch_sizes]
    
    ax12.plot(batch_sizes, mem_99, 'o-', label='99th percentile', linewidth=2)
    ax12.plot(batch_sizes, mem_99_9, 's-', label='99.9th percentile', linewidth=2)
    ax12.plot(batch_sizes, mem_max, '^-', label='Maximum', linewidth=2)
    ax12.set_xlabel('Batch Size')
    ax12.set_ylabel('Memory (GB)')
    ax12.set_title('Memory Requirements (raw data only)')
    ax12.grid(True, alpha=0.3)
    ax12.legend()
    ax12.set_yscale('log')
    
    plt.suptitle(f'IceCube Data Analysis - {stats["n_events"]:,} events, {stats["n_total_pulses"]:,} pulses', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig

def print_architecture_recommendations(stats):
    """Print specific architecture recommendations based on the data."""
    
    events_df = stats['events_stats']
    dom_df = stats['dom_stats']
    
    n_doms = events_df["n_doms"].to_numpy()
    n_pulses = events_df["n_pulses"].to_numpy()
    pulses_per_dom = dom_df["pulses_per_dom"].to_numpy()
    
    print("\n" + "="*70)
    print("ARCHITECTURE RECOMMENDATIONS BASED ON DATA ANALYSIS")
    print("="*70)
    
    # DOM-level recommendations
    print("\n1. DOM-LEVEL TRANSFORMER (T1) RECOMMENDATIONS:")
    print("-" * 50)
    
    pulse_buckets = [
        (1, 10, "Short sequences - direct attention"),
        (10, 100, "Medium sequences - standard transformer"),
        (100, 1000, "Long sequences - consider chunking"),
        (1000, 10000, "Very long - need hierarchical/windowed attention"),
        (10000, 100000, "Extreme - must use chunking/pooling")
    ]
    
    for low, high, desc in pulse_buckets:
        mask = (pulses_per_dom >= low) & (pulses_per_dom < high)
        pct = 100 * np.mean(mask)
        if pct > 0.01:
            print(f"   {low:6d}-{high:6d} pulses: {pct:6.2f}% of DOMs - {desc}")
    
    p99_pulses = np.percentile(pulses_per_dom, 99)
    print(f"\n   Suggested max sequence length for T1: {int(p99_pulses)} (covers 99% of DOMs)")
    print(f"   For the remaining 1%, use chunking with window size: {min(512, int(p99_pulses))}")
    
    # Event-level recommendations
    print("\n2. EVENT-LEVEL TRANSFORMER (T2) RECOMMENDATIONS:")
    print("-" * 50)
    
    p95_doms = np.percentile(n_doms, 95)
    p99_doms = np.percentile(n_doms, 99)
    p99_9_doms = np.percentile(n_doms, 99.9)
    
    print(f"   95th percentile: {p95_doms:.0f} DOMs")
    print(f"   99th percentile: {p99_doms:.0f} DOMs")
    print(f"   99.9th percentile: {p99_9_doms:.0f} DOMs")
    print(f"   Maximum: {np.max(n_doms):.0f} DOMs")
    
    if p99_doms < 2000:
        print(f"\n   ✓ Set max sequence length to {int(p99_doms)} for T2")
        print("   ✓ This is manageable without special techniques")
    else:
        print(f"\n   ⚠ Consider hierarchical approach - {p99_doms:.0f} is large for attention")
    
    # Batching strategy
    print("\n3. BATCHING STRATEGY:")
    print("-" * 50)
    
    # Check data distribution characteristics
    heavy_tail_pct = 100 * np.mean(pulses_per_dom > 1000)
    very_long_pct = 100 * np.mean(pulses_per_dom > 100)
    cv = np.std(pulses_per_dom) / np.mean(pulses_per_dom)
    
    # Determine best strategy based on data
    if very_long_pct < 1 and np.percentile(n_doms, 99) < 500:
        print("   Recommended approach: OPTION 2 (Grouped Batch Processing)")
        print("   Reasons:")
        print(f"   - Only {very_long_pct:.2f}% of DOMs have >100 pulses")
        print(f"   - 99th percentile of DOMs per event is only {np.percentile(n_doms, 99):.0f}")
        print("   - Simple grouping by pulse count will work well")
    else:
        print("   Recommended approach: OPTION 3 (Continuous/Flattened Batching)")
        print("   Reasons:")
        if heavy_tail_pct > 0.1:
            print(f"   - {heavy_tail_pct:.2f}% of DOMs have >1000 pulses (heavy tail)")
        if very_long_pct > 1:
            print(f"   - {very_long_pct:.2f}% of DOMs have >100 pulses")
    
    print(f"   - Coefficient of variation: {cv:.1f} (high variance in lengths)")
    
    # Memory constraints
    print("\n4. MEMORY MANAGEMENT:")
    print("-" * 50)
    
    batch_32_mem_99 = 32 * np.percentile(n_pulses, 99) * 4 * 4 / (1024**3)
    batch_32_mem_max = 32 * np.max(n_pulses) * 4 * 4 / (1024**3)
    
    print(f"   Batch size 32:")
    print(f"   - 99th percentile: {batch_32_mem_99:.2f} GB (manageable)")
    print(f"   - Worst case: {batch_32_mem_max:.2f} GB (may OOM on smaller GPUs)")
    print(f"\n   Recommendation: Use gradient accumulation for large events")
    print(f"   or implement dynamic batch sizing based on total pulses")
    
    # Special handling
    print("\n5. SPECIAL HANDLING NEEDED:")
    print("-" * 50)
    
    xxl_events = np.mean(n_pulses > 100000) * 100
    xl_events = np.mean(n_pulses > 10000) * 100
    if xxl_events > 0:
        print(f"   ⚠ {xxl_events:.3f}% of events have >100K pulses")
        print("   → Implement special path for these mega-events")
        print("   → Consider: Reservoir sampling, importance sampling, or splitting")
    elif xl_events > 0.1:
        print(f"   ⚠ {xl_events:.2f}% of events have >10K pulses")
        print("   → May need special handling for these large events")
    else:
        print("   ✓ No extreme outliers requiring special handling")
    
    # Add surprising findings
    print("\n6. KEY INSIGHTS FROM DATA:")
    print("-" * 50)
    
    short_dom_pct = 100 * np.mean(pulses_per_dom < 10)
    print(f"   📊 {short_dom_pct:.1f}% of DOMs have <10 pulses (very sparse!)")
    
    if np.percentile(n_doms, 99) < 300:
        print(f"   📊 99% of events use <{np.percentile(n_doms, 99):.0f} DOMs (out of 5160 total)")
        print("   → Most events only activate ~5% of the detector!")
    
    median_pulses_per_dom = np.median(pulses_per_dom)
    if median_pulses_per_dom < 5:
        print(f"   📊 Median pulses per DOM: {median_pulses_per_dom:.0f}")
        print("   → Most DOMs capture very few photons")
    
    print("\n" + "="*70)

def main():
    # Configuration
    data_dir = "/groups/pheno/inar/icecube_kaggle/train"
    
    # Start with just one file for quick analysis
    batch_file = Path(data_dir) / "batch_1.parquet"
    
    if not batch_file.exists():
        # Try to find any batch file
        import glob
        files = glob.glob(str(Path(data_dir) / "batch_*.parquet"))
        if files:
            batch_file = files[0]
        else:
            raise FileNotFoundError(f"No batch files found in {data_dir}")
    
    print(f"Analyzing: {batch_file}")
    
    # Run fast analysis
    stats = analyze_single_batch_fast(batch_file)
    
    # Create visualizations
    fig = create_fast_plots(stats)
    
    # Print recommendations
    print_architecture_recommendations(stats)
    
    # Save outputs
    n_files = stats.get('n_files', 1)
    output_path = f"icecube_data_analysis_{n_files}batches.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    
    plt.show()
    
    # Quick additional stats
    print("\n" + "="*70)
    print("QUICK STATS SUMMARY")
    print("="*70)
    events_df = stats['events_stats']
    dom_df = stats['dom_stats']
    
    print(f"Total events analyzed: {len(events_df):,}")
    print(f"Total pulses: {stats['n_total_pulses']:,}")
    print(f"Total DOM readings: {len(dom_df):,}")
    print(f"Average DOMs per event: {events_df['n_doms'].mean():.1f}")
    print(f"Average pulses per DOM: {dom_df['pulses_per_dom'].mean():.1f}")
    print(f"Max pulses in single DOM: {dom_df['pulses_per_dom'].max():,}")
    print(f"Max pulses in single event: {events_df['n_pulses'].max():,}")

if __name__ == "__main__":
    main()
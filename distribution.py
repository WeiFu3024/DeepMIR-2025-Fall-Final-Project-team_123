import os
import glob
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import json

def get_audio_duration(filepath):
    """
    Get duration of an audio file in seconds.
    
    Parameters
    ----------
    filepath : str
        Path to audio file
    
    Returns
    -------
    duration : float
        Duration in seconds, or None if error
    """
    try:
        info = sf.info(filepath)
        return info.duration
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def categorize_duration(duration):
    """
    Categorize duration into bins.
    
    Parameters
    ----------
    duration : float
        Duration in seconds
    
    Returns
    -------
    category : str
        Duration category
    """
    if duration < 5:
        return "0-5s"
    elif duration < 10:
        return "5-10s"
    elif duration < 15:
        return "10-15s"
    elif duration < 20:
        return "15-20s"
    elif duration < 25:
        return "20-25s"
    elif duration < 30:
        return "25-30s"
    else:
        return "30s+"

def analyze_audio_distribution(data_dir, save_dir='distribution_results'):
    """
    Analyze duration distribution of audio files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing .wav files
    save_dir : str
        Directory to save results and plots
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all wav files
    wav_files = glob.glob(os.path.join(data_dir, '*.wav'))
    wav_files += glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True)
    
    print(f"Found {len(wav_files)} WAV files in {data_dir}")
    
    # Storage for results
    durations = []
    categories = defaultdict(list)  # category -> list of (filename, duration)
    
    # Define category order
    category_order = ["0-5s", "5-10s", "10-15s", "15-20s", "20-25s", "25-30s", "30s+"]
    
    # Process each file
    for filepath in tqdm(wav_files, desc="Analyzing audio files"):
        duration = get_audio_duration(filepath)
        
        if duration is not None:
            durations.append(duration)
            category = categorize_duration(duration)
            filename = os.path.basename(filepath)
            categories[category].append((filename, duration))
    
    # Calculate statistics
    durations = np.array(durations)
    stats = {
        'total_files': len(durations),
        'mean_duration': float(np.mean(durations)),
        'median_duration': float(np.median(durations)),
        'std_duration': float(np.std(durations)),
        'min_duration': float(np.min(durations)),
        'max_duration': float(np.max(durations)),
        'categories': {}
    }
    
    # Count per category
    for cat in category_order:
        count = len(categories[cat])
        percentage = (count / len(durations) * 100) if len(durations) > 0 else 0
        stats['categories'][cat] = {
            'count': count,
            'percentage': round(percentage, 2)
        }
    
    # Print statistics
    print("\n" + "="*60)
    print("AUDIO DURATION DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"\nTotal files analyzed: {stats['total_files']}")
    print(f"Mean duration: {stats['mean_duration']:.2f} seconds")
    print(f"Median duration: {stats['median_duration']:.2f} seconds")
    print(f"Std deviation: {stats['std_duration']:.2f} seconds")
    print(f"Min duration: {stats['min_duration']:.2f} seconds")
    print(f"Max duration: {stats['max_duration']:.2f} seconds")
    print("\nDistribution by category:")
    print("-" * 60)
    
    for cat in category_order:
        count = stats['categories'][cat]['count']
        pct = stats['categories'][cat]['percentage']
        print(f"{cat:>10}: {count:>6} files ({pct:>6.2f}%)")
    
    # Save statistics to JSON
    stats_path = os.path.join(save_dir, 'duration_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")
    
    # Save file lists per category
    for cat in category_order:
        cat_path = os.path.join(save_dir, f'files_{cat.replace("-", "_").replace("+", "plus")}.txt')
        with open(cat_path, 'w') as f:
            for filename, duration in sorted(categories[cat]):
                f.write(f"{filename}\t{duration:.2f}s\n")
    print(f"File lists saved to: {save_dir}/files_*.txt")
    
    # Create histogram plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Bar chart by category
    counts = [stats['categories'][cat]['count'] for cat in category_order]
    percentages = [stats['categories'][cat]['percentage'] for cat in category_order]
    
    bars = ax1.bar(category_order, counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Duration Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Files', fontsize=12, fontweight='bold')
    ax1.set_title('Audio Duration Distribution by Category', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Continuous histogram
    ax2.hist(durations, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(stats['mean_duration'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {stats["mean_duration"]:.2f}s')
    ax2.axvline(stats['median_duration'], color='green', linestyle='--', 
                linewidth=2, label=f'Median: {stats["median_duration"]:.2f}s')
    ax2.set_xlabel('Duration (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Files', fontsize=12, fontweight='bold')
    ax2.set_title('Continuous Duration Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'duration_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {plot_path}")
    
    # Show plot
    plt.show()
    
    # Create detailed statistics plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Pie chart
    axes[0].pie(counts, labels=category_order, autopct='%1.1f%%', 
                startangle=90, colors=plt.cm.Set3(range(len(category_order))))
    axes[0].set_title('Distribution Percentage', fontsize=12, fontweight='bold')
    
    # Box plot
    axes[1].boxplot(durations, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Duration Statistics', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Cumulative distribution
    sorted_durations = np.sort(durations)
    cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
    axes[2].plot(sorted_durations, cumulative, linewidth=2, color='purple')
    axes[2].set_xlabel('Duration (seconds)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save detailed plot
    detail_plot_path = os.path.join(save_dir, 'duration_detailed_analysis.png')
    plt.savefig(detail_plot_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved to: {detail_plot_path}")
    
    # plt.show()
    
    return stats, categories

if __name__ == "__main__":
    # Analyze the bgm_cut folder
    data_dir = 'RapBank/data/bgm_cut'
    save_dir = 'distribution_results'
    
    stats, categories = analyze_audio_distribution(data_dir, save_dir)
    
    print("\n" + "="*60)
    print("Analysis complete! Results saved to:", save_dir)
    print("="*60)
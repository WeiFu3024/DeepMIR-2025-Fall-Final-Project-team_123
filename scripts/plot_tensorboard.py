#!/usr/bin/env python3
"""
Script to read TensorBoard event files, plot them, and save detailed logs.

Usage:
    python plot_tensorboard.py --log_dir results/experiment_distilhubert/models
    python plot_tensorboard.py --log_dir results/experiment_distilhubert/models --output_dir plots
"""

import os
import argparse
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Error: tensorboard package not found. Please install it:")
    print("pip install tensorboard")
    exit(1)


def read_tensorboard_events(log_dir):
    """
    Read all TensorBoard event files in a directory.
    
    Parameters
    ----------
    log_dir : str
        Directory containing event files
    
    Returns
    -------
    metrics : dict
        Dictionary of metric names to lists of (step, value) tuples
    """
    metrics = defaultdict(list)
    
    # Find all event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No event files found in {log_dir}")
        return metrics
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Read each event file
    for event_file in event_files:
        print(f"Reading: {event_file}")
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Get all scalar tags
        tags = ea.Tags()['scalars']
        
        for tag in tags:
            events = ea.Scalars(tag)
            for event in events:
                metrics[tag].append((event.step, event.value))
    
    # Sort by step and remove duplicates
    for tag in metrics:
        metrics[tag] = sorted(set(metrics[tag]), key=lambda x: x[0])
    
    return metrics


def save_to_csv(metrics, output_dir):
    """
    Save all metrics to CSV files.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to lists of (step, value) tuples
    output_dir : str
        Directory to save CSV files
    """
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Save each metric to a separate CSV
    for tag in metrics:
        steps = [v[0] for v in metrics[tag]]
        values = [v[1] for v in metrics[tag]]
        
        # Create safe filename from tag
        safe_tag = tag.replace('/', '_').replace('\\', '_')
        csv_path = os.path.join(csv_dir, f'{safe_tag}.csv')
        
        with open(csv_path, 'w') as f:
            f.write('step,value\n')
            for step, val in zip(steps, values):
                f.write(f'{step},{val}\n')
        
        print(f"Saved: {csv_path}")
    
    # Save all metrics to a single wide CSV
    all_csv_path = os.path.join(output_dir, 'all_metrics.csv')
    
    # Get all unique steps
    all_steps = set()
    for tag in metrics:
        for step, _ in metrics[tag]:
            all_steps.add(step)
    all_steps = sorted(all_steps)
    
    # Create header
    header = ['step']
    for tag in sorted(metrics.keys()):
        safe_tag = tag.replace('/', '_').replace('\\', '_')
        header.append(safe_tag)
    
    with open(all_csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        
        for step in all_steps:
            row = [str(step)]
            for tag in sorted(metrics.keys()):
                # Find value for this step
                val = None
                for s, v in metrics[tag]:
                    if s == step:
                        val = v
                        break
                row.append(str(val) if val is not None else '')
            f.write(','.join(row) + '\n')
    
    print(f"Saved combined: {all_csv_path}")


def plot_metrics(metrics, output_dir):
    """
    Create plots for all metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to lists of (step, value) tuples
    output_dir : str
        Directory to save plots
    """
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Group metrics by prefix (train, val, test)
    grouped_metrics = defaultdict(list)
    for tag in metrics:
        prefix = tag.split('/')[0] if '/' in tag else 'other'
        grouped_metrics[prefix].append(tag)
    
    # Plot each metric individually
    for tag in sorted(metrics.keys()):
        steps = [v[0] for v in metrics[tag]]
        values = [v[1] for v in metrics[tag]]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(tag, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create safe filename
        safe_tag = tag.replace('/', '_').replace('\\', '_')
        plot_path = os.path.join(plot_dir, f'{safe_tag}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")
    
    # Create grouped plots (all metrics with same prefix in one plot)
    for prefix, tags in grouped_metrics.items():
        if len(tags) <= 1:
            continue
        
        plt.figure(figsize=(12, 8))
        for tag in tags:
            steps = [v[0] for v in metrics[tag]]
            values = [v[1] for v in metrics[tag]]
            label = tag.split('/')[-1] if '/' in tag else tag
            plt.plot(steps, values, marker='o', linestyle='-', linewidth=2, markersize=3, label=label)
        
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(f'{prefix.capitalize()} Metrics', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(plot_dir, f'{prefix}_all_metrics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved grouped plot: {plot_path}")


def save_detailed_summary(metrics, output_dir):
    """
    Save a detailed text summary of all metrics.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to lists of (step, value) tuples
    output_dir : str
        Directory to save summary
    """
    summary_path = os.path.join(output_dir, 'detailed_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("DETAILED TENSORBOARD SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        for tag in sorted(metrics.keys()):
            values = metrics[tag]
            if not values:
                continue
            
            steps = [v[0] for v in values]
            vals = [v[1] for v in values]
            
            f.write(f"\n{tag}\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Total entries: {len(values)}\n")
            f.write(f"  Steps: {min(steps)} to {max(steps)}\n")
            f.write(f"  First value: {vals[0]:.8f} (step {steps[0]})\n")
            f.write(f"  Last value: {vals[-1]:.8f} (step {steps[-1]})\n")
            f.write(f"  Min value: {min(vals):.8f} (step {steps[vals.index(min(vals))]})\n")
            f.write(f"  Max value: {max(vals):.8f} (step {steps[vals.index(max(vals))]})\n")
            f.write(f"  Mean value: {np.mean(vals):.8f}\n")
            f.write(f"  Std dev: {np.std(vals):.8f}\n")
            f.write(f"  Median: {np.median(vals):.8f}\n")
            
            # Calculate trend
            if len(vals) >= 2:
                trend = vals[-1] - vals[0]
                trend_pct = (trend / abs(vals[0])) * 100 if vals[0] != 0 else 0
                f.write(f"  Trend (first to last): {trend:+.8f} ({trend_pct:+.2f}%)\n")
            
            # Show all values
            f.write(f"\n  All values:\n")
            f.write(f"  {'Step':>8s}  {'Value':>15s}\n")
            f.write(f"  {'-'*8}  {'-'*15}\n")
            for step, val in values:
                f.write(f"  {step:8d}  {val:15.8f}\n")
            
            f.write("\n")
        
        f.write("=" * 100 + "\n")
    
    print(f"Saved detailed summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot and save TensorBoard event files')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing TensorBoard event files')
    parser.add_argument('--output_dir', type=str, default='tensorboard_analysis',
                       help='Output directory for plots and logs (default: tensorboard_analysis)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"Error: Directory not found: {args.log_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading event files from: {args.log_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Read metrics
    metrics = read_tensorboard_events(args.log_dir)
    
    if not metrics:
        print("No metrics found in event files")
        return
    
    print(f"\nFound {len(metrics)} metric(s)")
    print()
    
    # Save to CSV
    print("Saving to CSV...")
    save_to_csv(metrics, args.output_dir)
    print()
    
    # Create plots
    print("Creating plots...")
    plot_metrics(metrics, args.output_dir)
    print()
    
    # Save detailed summary
    print("Saving detailed summary...")
    save_detailed_summary(metrics, args.output_dir)
    print()
    
    print("=" * 80)
    print("COMPLETE!")
    print(f"All outputs saved to: {args.output_dir}")
    print("  - plots/: Individual and grouped metric plots")
    print("  - csv/: Individual metric CSV files")
    print("  - all_metrics.csv: Combined metrics in one file")
    print("  - detailed_summary.txt: Detailed text summary")
    print("=" * 80)


if __name__ == '__main__':
    main()
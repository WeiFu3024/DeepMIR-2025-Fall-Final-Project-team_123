#!/usr/bin/env python3
"""
Script to read TensorBoard event files and output a summary.

Usage:
    python summarize_tensorboard.py --log_dir results/experiment_distilhubert/models
    python summarize_tensorboard.py --log_dir results/experiment_distilhubert/models --output summary.txt
"""

import os
import argparse
from collections import defaultdict
import numpy as np

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
    
    # Sort by step
    for tag in metrics:
        metrics[tag] = sorted(metrics[tag], key=lambda x: x[0])
    
    return metrics


def format_summary(metrics):
    """
    Format metrics into a readable summary.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to lists of (step, value) tuples
    
    Returns
    -------
    summary : str
        Formatted summary string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("TENSORBOARD SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    for tag in sorted(metrics.keys()):
        values = metrics[tag]
        if not values:
            continue
        
        steps = [v[0] for v in values]
        vals = [v[1] for v in values]
        
        lines.append(f"\n{tag}")
        lines.append("-" * 80)
        lines.append(f"  Total entries: {len(values)}")
        lines.append(f"  Steps: {min(steps)} to {max(steps)}")
        lines.append(f"  First value: {vals[0]:.6f} (step {steps[0]})")
        lines.append(f"  Last value: {vals[-1]:.6f} (step {steps[-1]})")
        lines.append(f"  Min value: {min(vals):.6f} (step {steps[vals.index(min(vals))]})")
        lines.append(f"  Max value: {max(vals):.6f} (step {steps[vals.index(max(vals))]})")
        lines.append(f"  Mean value: {np.mean(vals):.6f}")
        lines.append(f"  Std dev: {np.std(vals):.6f}")
        
        # Show recent trend (last 5 values)
        if len(vals) >= 5:
            recent = vals[-5:]
            lines.append(f"  Last 5 values: {', '.join([f'{v:.6f}' for v in recent])}")
        
        # Show full progression if not too many values
        if len(values) <= 20:
            lines.append(f"  Full progression:")
            for step, val in values:
                lines.append(f"    Step {step:6d}: {val:.6f}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarize TensorBoard event files')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing TensorBoard event files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (if not specified, prints to stdout)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"Error: Directory not found: {args.log_dir}")
        return
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if args.output else None

    print(f"Reading event files from: {args.log_dir}")
    metrics = read_tensorboard_events(args.log_dir)
    
    if not metrics:
        print("No metrics found in event files")
        return
    
    summary = format_summary(metrics)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(summary)
        print(f"\nSummary saved to: {args.output}")
    else:
        print("\n" + summary)


if __name__ == '__main__':
    main()
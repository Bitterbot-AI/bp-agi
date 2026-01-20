#!/usr/bin/env python3
"""
ARC Submission Visualizer

Loads submission.json and creates side-by-side comparisons:
- Attempt 1 (Clean/Deterministic)
- Attempt 2 (Noisy/Stochastic Resonance)

This allows visual verification of whether the noise injection
generates useful variations.
"""

import json
import os
import sys
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Text-only output.")

# ARC Color Palette (official colors)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Magenta/Pink
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Cyan/Light Blue
    '#870C25',  # 9: Maroon/Dark Red
]

def load_submission(path: str) -> dict:
    """Load submission.json"""
    with open(path) as f:
        return json.load(f)

def visualize_task(task_id: str, predictions: list, output_dir: str):
    """Create visualization for a single task"""
    if not HAS_MATPLOTLIB:
        return

    cmap = ListedColormap(ARC_COLORS)
    num_tests = len(predictions)

    fig, axs = plt.subplots(num_tests, 2, figsize=(8, 4 * num_tests))

    # Handle single test case
    if num_tests == 1:
        axs = [axs]

    for i, pred in enumerate(predictions):
        # Attempt 1 (Clean)
        att1 = np.array(pred['attempt_1'])
        axs[i][0].imshow(att1, cmap=cmap, vmin=0, vmax=9)
        axs[i][0].set_title(f"Task {task_id} - Test {i}\nAttempt 1 (Clean)")
        axs[i][0].axis('off')

        # Attempt 2 (Noisy)
        att2 = np.array(pred['attempt_2'])
        axs[i][1].imshow(att2, cmap=cmap, vmin=0, vmax=9)
        axs[i][1].set_title(f"Attempt 2 (Noisy)")
        axs[i][1].axis('off')

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"verify_{task_id}.png")
    plt.savefig(outpath, dpi=100)
    plt.close()
    print(f"  Saved: {outpath}")

def compare_attempts(predictions: list) -> dict:
    """Compare attempt 1 and attempt 2 for differences"""
    stats = {
        'identical': 0,
        'different': 0,
        'total': 0
    }

    for pred in predictions:
        att1 = pred['attempt_1']
        att2 = pred['attempt_2']
        stats['total'] += 1

        if att1 == att2:
            stats['identical'] += 1
        else:
            stats['different'] += 1

    return stats

def main():
    parser = argparse.ArgumentParser(description='Verify ARC submission')
    parser.add_argument('--submission', default='submission.json',
                        help='Path to submission.json')
    parser.add_argument('--output', default='verification',
                        help='Output directory for visualizations')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of tasks to visualize (0=all)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization, only show statistics')
    args = parser.parse_args()

    print("=" * 60)
    print("ARC Submission Verification Tool")
    print("=" * 60)
    print()

    # Load submission
    if not os.path.exists(args.submission):
        print(f"Error: {args.submission} not found")
        return 1

    print(f"Loading: {args.submission}")
    submission = load_submission(args.submission)

    num_tasks = len(submission)
    total_tests = sum(len(preds) for preds in submission.values())
    print(f"Tasks: {num_tasks}")
    print(f"Total test cases: {total_tests}")
    print()

    # Analyze attempt differences
    all_stats = {'identical': 0, 'different': 0, 'total': 0}
    for task_id, predictions in submission.items():
        stats = compare_attempts(predictions)
        all_stats['identical'] += stats['identical']
        all_stats['different'] += stats['different']
        all_stats['total'] += stats['total']

    print("=" * 60)
    print("ATTEMPT COMPARISON")
    print("=" * 60)
    print(f"Identical (Att1 == Att2): {all_stats['identical']}")
    print(f"Different (Att1 != Att2): {all_stats['different']}")
    print(f"Total test cases:         {all_stats['total']}")
    print()

    if all_stats['different'] == 0:
        print("WARNING: All attempts are identical!")
        print("The noise injection may not be working, or the system")
        print("is deterministically returning the same output.")
    else:
        pct = 100.0 * all_stats['different'] / all_stats['total']
        print(f"Stochastic variation: {pct:.1f}% of tests show differences")
    print()

    # Generate visualizations
    if not args.no_viz and HAS_MATPLOTLIB:
        print("=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        count = 0
        for task_id, predictions in submission.items():
            if args.limit > 0 and count >= args.limit:
                break

            visualize_task(task_id, predictions, args.output)
            count += 1

        print()
        print(f"Visualizations saved to: {args.output}/")

    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

    return 0

if __name__ == '__main__':
    sys.exit(main())

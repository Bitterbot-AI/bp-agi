#!/usr/bin/env python3
"""
ARC Benchmark Visualizer - Full PDF Report

Generates a single PDF showing all ARC tasks with:
- Input grid
- Expected output grid
- Predicted output grid (from C++ benchmark)
- Accuracy % and PASS/FAIL status

Usage:
    python visualize_arc_benchmark.py --results honeybee_results.json --output report.pdf
    python visualize_arc_benchmark.py --bin arc_eval.bin --output all_tasks.pdf  (just show puzzles)
"""

import json
import struct
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Error: matplotlib required. Install with: pip install matplotlib")
    exit(1)

# ARC Official Color Palette
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Cyan
    '#870C25',  # 9: Maroon
]
ARC_CMAP = ListedColormap(ARC_COLORS)

# Grayscale voltage to ARC color mapping (inverse of convert_arc.py)
VOLTAGE_TO_COLOR = {
    0: 0, 28: 1, 56: 2, 84: 3, 112: 4,
    140: 5, 168: 6, 196: 7, 224: 8, 252: 9
}

def voltage_to_arc(voltage):
    """Convert grayscale voltage back to ARC color index."""
    # Find closest mapped value
    if voltage in VOLTAGE_TO_COLOR:
        return VOLTAGE_TO_COLOR[voltage]
    # Find nearest
    closest = min(VOLTAGE_TO_COLOR.keys(), key=lambda x: abs(x - voltage))
    return VOLTAGE_TO_COLOR[closest]

def load_binary_tasks(bin_path):
    """Load tasks from binary format."""
    tasks = []
    with open(bin_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'BARC':
            raise ValueError(f"Invalid magic: {magic}")

        num_tasks = struct.unpack('I', f.read(4))[0]

        for _ in range(num_tasks):
            task_id = f.read(8).rstrip(b'\x00').decode('utf-8')
            num_train, num_test = struct.unpack('II', f.read(8))

            train_pairs = []
            for _ in range(num_train):
                inp = np.frombuffer(f.read(4096), dtype=np.uint8).reshape(64, 64)
                out = np.frombuffer(f.read(4096), dtype=np.uint8).reshape(64, 64)
                train_pairs.append({'input': inp, 'output': out})

            test_pairs = []
            for _ in range(num_test):
                inp = np.frombuffer(f.read(4096), dtype=np.uint8).reshape(64, 64)
                out = np.frombuffer(f.read(4096), dtype=np.uint8).reshape(64, 64)
                test_pairs.append({'input': inp, 'output': out})

            tasks.append({
                'id': task_id,
                'train': train_pairs,
                'test': test_pairs
            })

    return tasks

def extract_content_bounds(grid):
    """Find the bounding box of non-black content."""
    # Convert voltages to colors
    colored = np.vectorize(voltage_to_arc)(grid)

    # Find non-black pixels
    rows = np.any(colored != 0, axis=1)
    cols = np.any(colored != 0, axis=0)

    if not rows.any() or not cols.any():
        return 0, 64, 0, 64  # Full grid if empty

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add small padding
    pad = 1
    return max(0, y_min-pad), min(64, y_max+pad+1), max(0, x_min-pad), min(64, x_max+pad+1)

def plot_grid(ax, grid, title, show_gridlines=True):
    """Plot a single ARC grid."""
    # Convert voltages to ARC colors
    colored = np.vectorize(voltage_to_arc)(grid)

    # Extract content bounds
    y1, y2, x1, x2 = extract_content_bounds(grid)
    cropped = colored[y1:y2, x1:x2]

    ax.imshow(cropped, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_gridlines and cropped.shape[0] < 32 and cropped.shape[1] < 32:
        # Add gridlines for small grids
        ax.set_xticks(np.arange(-0.5, cropped.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cropped.shape[0], 1), minor=True)
        ax.grid(which='minor', color='#444444', linewidth=0.5)
        ax.tick_params(which='minor', size=0)

def create_task_page(pdf, task, result=None, task_num=None, total_tasks=None):
    """Create a single page for one task."""
    task_id = task['id']
    num_train = len(task['train'])
    num_test = len(task['test'])

    # Determine grid layout
    # Top rows: training pairs (input -> output)
    # Bottom row: test (input -> expected -> predicted)

    max_cols = max(num_train, num_test) * 2 + 1
    num_rows = 2 if num_test > 0 else 1

    fig = plt.figure(figsize=(12, 4 * num_rows))

    # Title with result info
    title = f"Task {task_id}"
    if task_num and total_tasks:
        title = f"[{task_num}/{total_tasks}] {title}"
    if result:
        acc = result.get('accuracy', 0) * 100
        passed = result.get('passed', False)
        status = "PASS" if passed else "FAIL"
        color = 'green' if passed else 'red'
        title += f" | {acc:.1f}% [{status}]"

    fig.suptitle(title, fontsize=14, fontweight='bold',
                 color='green' if (result and result.get('passed')) else 'black')

    # Training examples
    for i, pair in enumerate(task['train']):
        # Input
        ax_in = fig.add_subplot(num_rows, max_cols, i*2 + 1)
        plot_grid(ax_in, pair['input'], f"Train {i+1} In")

        # Arrow
        if i*2 + 2 <= max_cols:
            ax_arrow = fig.add_subplot(num_rows, max_cols, i*2 + 2)
            ax_arrow.text(0.5, 0.5, '→', fontsize=20, ha='center', va='center')
            ax_arrow.axis('off')

        # Output (skip if would overflow)
        if i*2 + 3 <= max_cols:
            ax_out = fig.add_subplot(num_rows, max_cols, i*2 + 3)
            plot_grid(ax_out, pair['output'], f"Train {i+1} Out")

    # Test examples (second row)
    if num_test > 0 and num_rows > 1:
        test_pair = task['test'][0]  # First test case

        # Test input
        ax_test_in = fig.add_subplot(num_rows, max_cols, max_cols + 1)
        plot_grid(ax_test_in, test_pair['input'], "Test Input")

        # Arrow
        ax_arrow = fig.add_subplot(num_rows, max_cols, max_cols + 2)
        ax_arrow.text(0.5, 0.5, '→', fontsize=20, ha='center', va='center')
        ax_arrow.axis('off')

        # Expected output
        ax_expected = fig.add_subplot(num_rows, max_cols, max_cols + 3)
        plot_grid(ax_expected, test_pair['output'], "Expected")

        # Predicted output (if we have results)
        if result and 'predicted' in result:
            ax_arrow2 = fig.add_subplot(num_rows, max_cols, max_cols + 4)
            ax_arrow2.text(0.5, 0.5, 'vs', fontsize=14, ha='center', va='center')
            ax_arrow2.axis('off')

            ax_pred = fig.add_subplot(num_rows, max_cols, max_cols + 5)
            pred_grid = np.array(result['predicted']).reshape(64, 64).astype(np.uint8)
            plot_grid(ax_pred, pred_grid, "Predicted")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_summary_page(pdf, tasks, results=None):
    """Create a summary page with statistics."""
    fig = plt.figure(figsize=(12, 8))

    fig.suptitle("ARC Benchmark Summary", fontsize=16, fontweight='bold')

    # Stats text
    ax = fig.add_subplot(111)
    ax.axis('off')

    text_lines = [
        f"Total Tasks: {len(tasks)}",
        "",
    ]

    if results:
        passed = sum(1 for r in results.values() if r.get('passed', False))
        total = len(results)
        avg_acc = np.mean([r.get('accuracy', 0) for r in results.values()]) * 100

        text_lines.extend([
            f"Tasks Evaluated: {total}",
            f"Passed: {passed} ({100*passed/total:.1f}%)",
            f"Failed: {total - passed}",
            f"Average Accuracy: {avg_acc:.1f}%",
            "",
            "Near Misses (95-99.9%):",
        ])

        # Find near misses
        near_misses = [(tid, r['accuracy']*100) for tid, r in results.items()
                       if 95 <= r.get('accuracy', 0)*100 < 100]
        near_misses.sort(key=lambda x: -x[1])

        for tid, acc in near_misses[:10]:
            text_lines.append(f"  {tid}: {acc:.1f}%")
    else:
        text_lines.append("No results loaded - showing puzzles only")

    ax.text(0.1, 0.9, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='ARC Benchmark PDF Visualizer')
    parser.add_argument('--bin', type=str, default='arc_eval.bin',
                        help='Binary task file (arc_eval.bin or arc_training.bin)')
    parser.add_argument('--results', type=str, default=None,
                        help='JSON results file from benchmark run')
    parser.add_argument('--output', type=str, default='arc_benchmark_report.pdf',
                        help='Output PDF filename')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of tasks (0=all)')
    args = parser.parse_args()

    print("=" * 60)
    print("ARC Benchmark PDF Visualizer")
    print("=" * 60)

    # Load tasks
    bin_path = Path(args.bin)
    if not bin_path.exists():
        # Try relative to script
        bin_path = Path(__file__).parent / args.bin

    if not bin_path.exists():
        print(f"Error: {args.bin} not found")
        return 1

    print(f"Loading tasks from: {bin_path}")
    tasks = load_binary_tasks(bin_path)
    print(f"Loaded {len(tasks)} tasks")

    # Load results if provided
    results = None
    if args.results and Path(args.results).exists():
        print(f"Loading results from: {args.results}")
        with open(args.results) as f:
            results = json.load(f)
        print(f"Loaded results for {len(results)} tasks")

    # Apply limit
    if args.limit > 0:
        tasks = tasks[:args.limit]
        print(f"Limited to {len(tasks)} tasks")

    # Generate PDF
    print(f"\nGenerating PDF: {args.output}")
    print("This may take a moment...")

    with PdfPages(args.output) as pdf:
        # Summary page first
        create_summary_page(pdf, tasks, results)

        # Individual task pages
        for i, task in enumerate(tasks):
            task_result = results.get(task['id']) if results else None
            create_task_page(pdf, task, task_result, i+1, len(tasks))

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(tasks)} tasks")

    print(f"\nDone! Saved to: {args.output}")
    print("=" * 60)

    return 0

if __name__ == '__main__':
    exit(main())

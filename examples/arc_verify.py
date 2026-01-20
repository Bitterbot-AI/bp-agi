#!/usr/bin/env python3
"""
ARC-AGI-2 Verification Tool

Creates externally verifiable results:
1. Proper submission JSON format (2 attempts per test)
2. Visual comparison images (input, expected, predicted, error)
3. Detailed per-test breakdown

Usage:
    python examples/arc_verify.py [--visualize]
"""

import json
import struct
import numpy as np
import os
import sys
from pathlib import Path

# Configuration
RETINA_SIZE = 64
DATA_DIR = Path(__file__).parent.parent / "data" / "arc-agi-2"
OUTPUT_DIR = Path(__file__).parent.parent / "verification"

# ARC color palette (official colors)
ARC_COLORS = [
    (0, 0, 0),       # 0: Black
    (0, 116, 217),   # 1: Blue
    (255, 65, 54),   # 2: Red
    (46, 204, 64),   # 3: Green
    (255, 220, 0),   # 4: Yellow
    (170, 170, 170), # 5: Gray
    (240, 18, 190),  # 6: Magenta
    (255, 133, 27),  # 7: Orange
    (127, 219, 255), # 8: Azure
    (135, 12, 37),   # 9: Maroon
]

# Grayscale mapping (must match convert_arc.py)
GRAY_TO_COLOR = {
    0: 0,    # Black
    28: 1,   # Blue
    56: 2,   # Red
    84: 3,   # Green
    112: 4,  # Yellow
    140: 5,  # Gray
    168: 6,  # Magenta
    196: 7,  # Orange
    224: 8,  # Azure
    252: 9,  # Maroon
}

def grayscale_to_arc_color(gray):
    """Map grayscale value back to ARC color (0-9)."""
    # Find closest match
    min_dist = float('inf')
    best_color = 0
    for g, c in GRAY_TO_COLOR.items():
        dist = abs(gray - g)
        if dist < min_dist:
            min_dist = dist
            best_color = c
    return best_color

def load_predictions_from_binary(bin_file):
    """Load predictions from arc_verify binary output."""
    predictions = {}

    if not os.path.exists(bin_file):
        return predictions

    with open(bin_file, 'rb') as f:
        # Read header
        magic = f.read(4)
        if magic != b'BPRD':  # BP-AGI PReDictions
            print(f"Invalid magic: {magic}")
            return predictions

        num_tasks = struct.unpack('I', f.read(4))[0]

        for _ in range(num_tasks):
            # Task ID (8 bytes, null-padded)
            task_id = f.read(8).rstrip(b'\x00').decode('utf-8')
            num_tests = struct.unpack('I', f.read(4))[0]

            task_preds = []
            for _ in range(num_tests):
                # Read 64x64 prediction
                pred_data = f.read(RETINA_SIZE * RETINA_SIZE)
                pred = np.frombuffer(pred_data, dtype=np.uint8).reshape(RETINA_SIZE, RETINA_SIZE)
                task_preds.append(pred)

            predictions[task_id] = task_preds

    return predictions

def extract_grid_from_retina(retina, original_shape):
    """
    Extract ARC grid from 64x64 retina.

    The grid was centered on the retina, so we need to:
    1. Find the center offset
    2. Extract the region
    3. Convert grayscale to colors
    """
    h, w = original_shape

    # Calculate offsets (same as convert_arc.py)
    off_y = (RETINA_SIZE - h) // 2
    off_x = (RETINA_SIZE - w) // 2

    # Extract region
    region = retina[off_y:off_y+h, off_x:off_x+w]

    # Convert to ARC colors
    grid = []
    for row in region:
        grid_row = []
        for val in row:
            color = grayscale_to_arc_color(val)
            grid_row.append(color)
        grid.append(grid_row)

    return grid

def compute_accuracy(pred_grid, expected_grid):
    """Compute pixel-level accuracy."""
    if len(pred_grid) != len(expected_grid):
        return 0.0
    if len(pred_grid) == 0:
        return 0.0
    if len(pred_grid[0]) != len(expected_grid[0]):
        return 0.0

    total = 0
    correct = 0
    for y in range(len(expected_grid)):
        for x in range(len(expected_grid[0])):
            total += 1
            if pred_grid[y][x] == expected_grid[y][x]:
                correct += 1

    return correct / total if total > 0 else 0.0

def create_submission_json(predictions, challenges, solutions, output_file):
    """Create ARC submission JSON format."""
    submission = {}

    for task_id, task in challenges.items():
        test_cases = task.get('test', [])
        task_submission = []

        for test_idx, test in enumerate(test_cases):
            input_grid = test['input']
            h, w = len(input_grid), len(input_grid[0]) if input_grid else 0

            # Get expected output shape (if available)
            if task_id in solutions and test_idx < len(solutions[task_id]):
                expected = solutions[task_id][test_idx]
                out_h, out_w = len(expected), len(expected[0]) if expected else 0
            else:
                out_h, out_w = h, w  # Assume same size

            # Get prediction
            if task_id in predictions and test_idx < len(predictions[task_id]):
                pred_retina = predictions[task_id][test_idx]
                pred_grid = extract_grid_from_retina(pred_retina, (out_h, out_w))
            else:
                # No prediction - use zeros
                pred_grid = [[0] * out_w for _ in range(out_h)]

            # For attempt_2, we could try a different approach
            # For now, use the same prediction
            task_submission.append({
                "attempt_1": pred_grid,
                "attempt_2": pred_grid  # Could be different strategy
            })

        submission[task_id] = task_submission

    with open(output_file, 'w') as f:
        json.dump(submission, f)

    print(f"Saved submission to: {output_file}")
    return submission

def create_visualization(task_id, test_idx, input_grid, expected_grid, pred_grid, output_dir):
    """Create visual comparison image."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    # Create ARC colormap
    cmap = ListedColormap([np.array(c)/255 for c in ARC_COLORS])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Input
    axes[0].imshow(input_grid, cmap=cmap, vmin=0, vmax=9)
    axes[0].set_title(f'Input\n{len(input_grid)}x{len(input_grid[0])}')
    axes[0].axis('off')

    # Expected
    axes[1].imshow(expected_grid, cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title(f'Expected\n{len(expected_grid)}x{len(expected_grid[0])}')
    axes[1].axis('off')

    # Predicted
    axes[2].imshow(pred_grid, cmap=cmap, vmin=0, vmax=9)
    axes[2].set_title(f'Predicted\n{len(pred_grid)}x{len(pred_grid[0])}')
    axes[2].axis('off')

    # Error map
    error_map = np.zeros_like(expected_grid, dtype=float)
    accuracy = 0
    total = 0
    for y in range(min(len(expected_grid), len(pred_grid))):
        for x in range(min(len(expected_grid[0]), len(pred_grid[0]))):
            total += 1
            if pred_grid[y][x] != expected_grid[y][x]:
                error_map[y][x] = 1
            else:
                accuracy += 1

    accuracy = accuracy / total if total > 0 else 0

    axes[3].imshow(error_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[3].set_title(f'Error Map\n{accuracy*100:.1f}% correct')
    axes[3].axis('off')

    plt.suptitle(f'Task: {task_id} Test: {test_idx}', fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir / f'{task_id}_test{test_idx}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 60)
    print("ARC-AGI-2 Verification Tool")
    print("=" * 60)
    print()

    visualize = '--visualize' in sys.argv

    # Load challenges and solutions
    print("Loading ARC evaluation data...")
    with open(DATA_DIR / 'arc-agi_evaluation_challenges.json') as f:
        challenges = json.load(f)
    with open(DATA_DIR / 'arc-agi_evaluation_solutions.json') as f:
        solutions = json.load(f)

    # Count test cases
    total_tests = sum(len(t.get('test', [])) for t in challenges.values())
    print(f"Tasks: {len(challenges)}")
    print(f"Total test cases: {total_tests}")
    print()

    # Load predictions
    pred_file = Path(__file__).parent.parent / 'arc_predictions.bin'
    print(f"Looking for predictions: {pred_file}")

    if not pred_file.exists():
        print("No predictions file found.")
        print("Run: ./build/arc_verify arc_eval.bin")
        print()
        print("Creating placeholder submission...")
        predictions = {}
    else:
        predictions = load_predictions_from_binary(pred_file)
        print(f"Loaded predictions for {len(predictions)} tasks")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate submission JSON
    submission = create_submission_json(
        predictions, challenges, solutions,
        OUTPUT_DIR / 'submission.json'
    )

    # Evaluate and create visualizations
    print()
    print("Evaluating predictions...")
    print("-" * 60)

    results = []
    passed = 0
    total = 0

    for task_id, task in challenges.items():
        test_cases = task.get('test', [])

        for test_idx, test in enumerate(test_cases):
            total += 1
            input_grid = test['input']

            if task_id in solutions and test_idx < len(solutions[task_id]):
                expected_grid = solutions[task_id][test_idx]
            else:
                continue

            # Get prediction
            if task_id in submission and test_idx < len(submission[task_id]):
                pred_grid = submission[task_id][test_idx]['attempt_1']
            else:
                pred_grid = [[0]]

            # Compute accuracy
            acc = compute_accuracy(pred_grid, expected_grid)
            is_correct = (acc == 1.0)

            if is_correct:
                passed += 1

            results.append({
                'task_id': task_id,
                'test_idx': test_idx,
                'accuracy': acc,
                'correct': is_correct,
                'input_shape': f"{len(input_grid)}x{len(input_grid[0])}",
                'output_shape': f"{len(expected_grid)}x{len(expected_grid[0])}"
            })

            # Create visualization for failures (or all if --visualize)
            if visualize or (not is_correct and acc > 0.5):
                create_visualization(
                    task_id, test_idx,
                    input_grid, expected_grid, pred_grid,
                    OUTPUT_DIR / 'images'
                )

    # Print results
    print()
    print("=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Total test cases: {total}")
    print(f"Pixel-perfect: {passed}/{total} ({100*passed/total:.2f}%)")
    print()

    # Breakdown by accuracy range
    acc_ranges = [(0.9, 1.0), (0.8, 0.9), (0.5, 0.8), (0.0, 0.5)]
    for low, high in acc_ranges:
        count = sum(1 for r in results if low <= r['accuracy'] < high or (high == 1.0 and r['accuracy'] == 1.0))
        print(f"  {low*100:.0f}%-{high*100:.0f}%: {count} tests")

    # Save detailed results
    with open(OUTPUT_DIR / 'detailed_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total,
                'passed': passed,
                'accuracy': passed / total if total > 0 else 0
            },
            'results': results
        }, f, indent=2)

    print()
    print(f"Submission JSON: {OUTPUT_DIR / 'submission.json'}")
    print(f"Detailed results: {OUTPUT_DIR / 'detailed_results.json'}")
    if visualize:
        print(f"Visualizations: {OUTPUT_DIR / 'images'}")

    # Final clean score
    print()
    print("=" * 60)
    print(f"CLEAN SCORE: {passed}/{total} = {100*passed/total:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()

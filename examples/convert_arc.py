#!/usr/bin/env python3
"""
Phase 13: ARC Data Converter

Converts ARC-AGI JSON format to optimized binary format for C++ ingestion.

Features:
- Centers small grids on 64x64 Retina
- Maps 10 ARC colors to distinct grayscale voltages
- Packs into flat binary for high-speed loading
"""

import json
import struct
import numpy as np
import os
import sys

# Configuration
RETINA_SIZE = 64

# Color mapping: 10 ARC colors to grayscale voltages
# These values trigger the Vision System's boundary detectors
COLOR_MAP = {
    0: 0,    # Black (background)
    1: 28,   # Blue
    2: 56,   # Red
    3: 84,   # Green
    4: 112,  # Yellow
    5: 140,  # Gray
    6: 168,  # Magenta
    7: 196,  # Orange
    8: 224,  # Azure
    9: 252   # Maroon
}


def pad_grid(grid):
    """Centers the ARC grid on the 64x64 Retina."""
    h, w = len(grid), len(grid[0]) if grid else 0

    if h == 0 or w == 0:
        return bytes(RETINA_SIZE * RETINA_SIZE)

    if h > RETINA_SIZE or w > RETINA_SIZE:
        print(f"  Warning: Grid size {h}x{w} exceeds Retina. Cropping.")
        h = min(h, RETINA_SIZE)
        w = min(w, RETINA_SIZE)

    # Create empty retina (black background)
    retina = np.zeros((RETINA_SIZE, RETINA_SIZE), dtype=np.uint8)

    # Calculate offsets to center
    off_y = (RETINA_SIZE - h) // 2
    off_x = (RETINA_SIZE - w) // 2

    # Fill retina with color-mapped values
    for y in range(h):
        for x in range(w):
            val = grid[y][x]
            retina[off_y + y][off_x + x] = COLOR_MAP.get(val, 0)

    return retina.flatten().tobytes()


def process_file(challenges_path, solutions_path, output_file):
    """Process a single ARC challenge/solution file pair."""
    print(f"Processing {challenges_path}...")

    if not os.path.exists(challenges_path):
        print(f"  Error: {challenges_path} not found")
        return 0

    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    solutions = {}
    if os.path.exists(solutions_path):
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)
        print(f"  Loaded {len(solutions)} solutions")
    else:
        print(f"  No solutions file found (test set)")

    # Binary Format:
    # [Magic: 4 bytes "BARC"]
    # [NumTasks: 4 bytes uint32]
    # For each Task:
    #   [ID: 8 bytes, null-padded]
    #   [NumTrain: 4 bytes uint32]
    #   [NumTest: 4 bytes uint32]
    #   [Train pairs: Input (4096 bytes) + Output (4096 bytes)]...
    #   [Test pairs: Input (4096 bytes) + Output (4096 bytes)]...

    with open(output_file, 'wb') as out:
        # Header "BARC" (Bitterbot ARC)
        out.write(b'BARC')
        out.write(struct.pack('I', len(challenges)))

        for task_id, task in challenges.items():
            # Task ID (Pad to 8 bytes)
            id_bytes = task_id.encode('utf-8')[:8].ljust(8, b'\x00')
            out.write(id_bytes)

            train_pairs = task.get('train', [])
            test_pairs = task.get('test', [])

            # Counts
            out.write(struct.pack('II', len(train_pairs), len(test_pairs)))

            # Train Data
            for pair in train_pairs:
                out.write(pad_grid(pair['input']))
                out.write(pad_grid(pair['output']))

            # Test Data
            for i, pair in enumerate(test_pairs):
                out.write(pad_grid(pair['input']))

                # Look for solution
                if task_id in solutions and i < len(solutions[task_id]):
                    out.write(pad_grid(solutions[task_id][i]))
                else:
                    # If no solution (hidden test set), write zeros
                    out.write(bytes(RETINA_SIZE * RETINA_SIZE))

    print(f"  Wrote {len(challenges)} tasks to {output_file}")
    return len(challenges)


def main():
    # Default data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'arc-agi-2')

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print("=" * 50)
    print("Phase 13: ARC Data Converter")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print()

    total_tasks = 0

    # Convert training set
    train_challenges = os.path.join(data_dir, 'arc-agi_training_challenges.json')
    train_solutions = os.path.join(data_dir, 'arc-agi_training_solutions.json')
    train_output = os.path.join(os.path.dirname(__file__), '..', 'arc_training.bin')
    total_tasks += process_file(train_challenges, train_solutions, train_output)

    # Convert evaluation set
    eval_challenges = os.path.join(data_dir, 'arc-agi_evaluation_challenges.json')
    eval_solutions = os.path.join(data_dir, 'arc-agi_evaluation_solutions.json')
    eval_output = os.path.join(os.path.dirname(__file__), '..', 'arc_eval.bin')
    total_tasks += process_file(eval_challenges, eval_solutions, eval_output)

    # Convert test set (no solutions)
    test_challenges = os.path.join(data_dir, 'arc-agi_test_challenges.json')
    test_solutions = ''  # No solutions for test set
    test_output = os.path.join(os.path.dirname(__file__), '..', 'arc_test.bin')
    if os.path.exists(test_challenges):
        total_tasks += process_file(test_challenges, test_solutions, test_output)

    print()
    print("=" * 50)
    print(f"Conversion Complete: {total_tasks} total tasks")
    print("=" * 50)


if __name__ == "__main__":
    main()

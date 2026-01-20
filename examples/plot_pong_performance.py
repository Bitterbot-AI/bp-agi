#!/usr/bin/env python3
"""
PONG PERFORMANCE VISUALIZER
============================

Visualizes the brain's learning during Pong conditioning.

Creates charts showing:
  1. Synaptic weight growth over time
  2. Hit rate progression
  3. Coach vs autonomous performance comparison

Usage:
    python3 plot_pong_performance.py [pong_training.csv]

Author: BP-AGI Project
"""

import sys
import csv

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Will generate text-based visualization.")


def load_csv(filename):
    """Load CSV data into a dictionary of lists."""
    data = {
        'Tick': [],
        'CoachEnabled': [],
        'BallX': [],
        'BallY': [],
        'PaddleX': [],
        'Action': [],
        'LeftWeight': [],
        'RightWeight': [],
        'Hits': [],
        'Misses': [],
        'HitRate': [],
    }

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['Tick'].append(int(row['Tick']))
            data['CoachEnabled'].append(int(row['CoachEnabled']))
            data['BallX'].append(int(row['BallX']))
            data['BallY'].append(int(row['BallY']))
            data['PaddleX'].append(int(row['PaddleX']))
            data['Action'].append(int(row['Action']))
            data['LeftWeight'].append(int(row['LeftWeight']))
            data['RightWeight'].append(int(row['RightWeight']))
            data['Hits'].append(int(row['Hits']))
            data['Misses'].append(int(row['Misses']))
            data['HitRate'].append(float(row['HitRate']))

    return data


def text_visualization(data):
    """Create ASCII visualization."""
    print("\n" + "=" * 60)
    print("PONG LEARNING VISUALIZATION (Text Mode)")
    print("=" * 60)

    # Find phase boundary
    coach_ticks = [i for i, c in enumerate(data['CoachEnabled']) if c == 1]
    test_ticks = [i for i, c in enumerate(data['CoachEnabled']) if c == 0]

    coach_end = max(coach_ticks) if coach_ticks else 0
    test_start = min(test_ticks) if test_ticks else len(data['Tick'])

    print(f"\nCoach Phase: Ticks 0-{coach_end}")
    print(f"Test Phase: Ticks {test_start}-{len(data['Tick'])}")

    # Weight progression
    print("\n" + "-" * 60)
    print("SYNAPTIC WEIGHT GROWTH")
    print("-" * 60)

    max_weight = max(max(data['LeftWeight']), max(data['RightWeight']), 1)

    # Sample every 100 ticks
    print("\nTick    | Left | Right | Bar")
    print("--------|------|-------|" + "-" * 40)

    for i in range(0, len(data['Tick']), 100):
        tick = data['Tick'][i]
        left = data['LeftWeight'][i]
        right = data['RightWeight'][i]

        # Create bar
        left_bar = int((left / max_weight) * 20) if max_weight > 0 else 0
        right_bar = int((right / max_weight) * 20) if max_weight > 0 else 0

        phase = "C" if data['CoachEnabled'][i] else "T"
        bar = "[" + "#" * left_bar + "-" * (20 - left_bar) + "|" + "*" * right_bar + "-" * (20 - right_bar) + "]"

        print(f"{tick:6} {phase}| {left:4} | {right:5} | {bar}")

    # Hit rate progression
    print("\n" + "-" * 60)
    print("HIT RATE PROGRESSION")
    print("-" * 60)

    print("\nTick    | Hits | Miss | Rate  | Bar")
    print("--------|------|------|-------|" + "-" * 25)

    for i in range(0, len(data['Tick']), 100):
        if i >= len(data['Tick']):
            break

        tick = data['Tick'][i]
        hits = data['Hits'][i]
        misses = data['Misses'][i]
        rate = data['HitRate'][i]

        phase = "C" if data['CoachEnabled'][i] else "T"
        bar_len = int(rate * 20)
        bar = "[" + "#" * bar_len + "-" * (20 - bar_len) + "]"

        print(f"{tick:6} {phase}| {hits:4} | {misses:4} | {rate:5.1%} | {bar}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if test_ticks:
        final_left = data['LeftWeight'][-1]
        final_right = data['RightWeight'][-1]
        final_rate = data['HitRate'][-1]
        final_hits = data['Hits'][-1]
        final_misses = data['Misses'][-1]

        print(f"\nFinal Synaptic Weights:")
        print(f"  Left Motor:  {final_left}")
        print(f"  Right Motor: {final_right}")
        print(f"\nFinal Performance (Test Phase):")
        print(f"  Hits: {final_hits}, Misses: {final_misses}")
        print(f"  Hit Rate: {final_rate:.1%}")

        if final_rate > 0.9:
            print("\n  EXCELLENT: >90% hit rate achieved!")
        elif final_rate > 0.7:
            print("\n  GOOD: Learning detected, >70% hit rate")
        elif final_rate > 0.5:
            print("\n  PARTIAL: Some learning, better than random")
        else:
            print("\n  NEEDS MORE TRAINING")


def matplotlib_visualization(data):
    """Create matplotlib visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    fig.suptitle("PONG LEARNING: Pavlovian Conditioning Results",
                 fontsize=14, fontweight='bold')

    ticks = data['Tick']

    # Find phase boundary
    coach_ticks = [i for i, c in enumerate(data['CoachEnabled']) if c == 1]
    test_start = max(coach_ticks) if coach_ticks else 0

    # ========================================
    # Panel 1: Synaptic Weights
    # ========================================
    ax1 = axes[0]
    ax1.set_ylabel('Synaptic Weight', fontsize=10)
    ax1.set_title('Motor Neuron Connection Strength (STDP Learning)')

    ax1.plot(ticks, data['LeftWeight'], color='#E74C3C', linewidth=2,
             label='Left Motor', alpha=0.8)
    ax1.plot(ticks, data['RightWeight'], color='#3498DB', linewidth=2,
             label='Right Motor', alpha=0.8)

    ax1.axvline(x=ticks[test_start] if test_start < len(ticks) else ticks[-1],
                color='gray', linestyle='--', alpha=0.7, label='Coach Off')

    ax1.fill_between(ticks, 0, data['LeftWeight'], color='#E74C3C', alpha=0.2)
    ax1.fill_between(ticks, 0, data['RightWeight'], color='#3498DB', alpha=0.2)

    ax1.legend(loc='upper left')
    ax1.set_ylim(bottom=0)

    # Phase labels
    if test_start > 0 and test_start < len(ticks):
        mid_coach = ticks[test_start // 2]
        mid_test = ticks[(test_start + len(ticks)) // 2]
        ax1.text(mid_coach, ax1.get_ylim()[1] * 0.9, 'COACHING',
                ha='center', fontsize=10, fontweight='bold', color='#27AE60')
        ax1.text(mid_test, ax1.get_ylim()[1] * 0.9, 'AUTONOMOUS',
                ha='center', fontsize=10, fontweight='bold', color='#E74C3C')

    # ========================================
    # Panel 2: Hit Rate
    # ========================================
    ax2 = axes[1]
    ax2.set_ylabel('Hit Rate', fontsize=10)
    ax2.set_title('Performance Over Time')

    ax2.plot(ticks, data['HitRate'], color='#27AE60', linewidth=2, alpha=0.8)
    ax2.fill_between(ticks, 0, data['HitRate'], color='#27AE60', alpha=0.3)

    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random Baseline')
    ax2.axhline(y=0.9, color='#F39C12', linestyle='--', alpha=0.7, label='90% Target')

    ax2.axvline(x=ticks[test_start] if test_start < len(ticks) else ticks[-1],
                color='gray', linestyle='--', alpha=0.7)

    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')

    # ========================================
    # Panel 3: Actions
    # ========================================
    ax3 = axes[2]
    ax3.set_ylabel('Motor Activity', fontsize=10)
    ax3.set_xlabel('Time (Ticks)', fontsize=10)
    ax3.set_title('Motor Neuron Firing (0=Left, 1=Right, 2=None)')

    # Plot actions as a scatter
    action_colors = ['#E74C3C' if a == 0 else '#3498DB' if a == 1 else '#95A5A6'
                     for a in data['Action']]
    ax3.scatter(ticks, data['Action'], c=action_colors, s=2, alpha=0.5)

    ax3.axvline(x=ticks[test_start] if test_start < len(ticks) else ticks[-1],
                color='gray', linestyle='--', alpha=0.7)

    ax3.set_ylim(-0.5, 2.5)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Left', 'Right', 'None'])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_file = "pong_performance.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved visualization to: {output_file}")

    plt.show()


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "pong_training.csv"

    try:
        data = load_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'")
        print("Run the test_pong simulation first.")
        sys.exit(1)

    print(f"Loaded {len(data['Tick'])} ticks from {csv_file}")

    if HAS_MATPLOTLIB:
        matplotlib_visualization(data)
    else:
        text_visualization(data)

    # Always print summary
    text_visualization(data)


if __name__ == "__main__":
    main()

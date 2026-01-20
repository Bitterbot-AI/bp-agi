#!/usr/bin/env python3
"""
OPERANT CONDITIONING LEARNING CURVE VISUALIZER
===============================================

Visualizes the brain's learning during operant conditioning.

Creates charts showing:
  1. Hit rate progression over epochs (the learning curve)
  2. Synaptic weight growth for LEFT and RIGHT motor neurons
  3. Comparison of early vs late performance

The key insight: No coach, no replay buffer, just eligibility traces.

Usage:
    python3 plot_learning_curve.py [operant_learning.csv]

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


def calculate_epoch_stats(data, epoch_size=500):
    """Calculate hit rate for each epoch."""
    epochs = []
    hit_rates = []
    left_weights = []
    right_weights = []

    total_ticks = len(data['Tick'])
    num_epochs = total_ticks // epoch_size

    for i in range(num_epochs):
        start_idx = i * epoch_size
        end_idx = (i + 1) * epoch_size

        # Get hits/misses at epoch boundaries
        if i == 0:
            epoch_hits = data['Hits'][end_idx - 1]
            epoch_misses = data['Misses'][end_idx - 1]
        else:
            epoch_hits = data['Hits'][end_idx - 1] - data['Hits'][start_idx - 1]
            epoch_misses = data['Misses'][end_idx - 1] - data['Misses'][start_idx - 1]

        total = epoch_hits + epoch_misses
        rate = epoch_hits / total if total > 0 else 0

        epochs.append(i + 1)
        hit_rates.append(rate)
        left_weights.append(data['LeftWeight'][end_idx - 1])
        right_weights.append(data['RightWeight'][end_idx - 1])

    return epochs, hit_rates, left_weights, right_weights


def text_visualization(data, epoch_size=500):
    """Create ASCII visualization."""
    print("\n" + "=" * 70)
    print("OPERANT CONDITIONING LEARNING CURVE (Text Mode)")
    print("The Bitterbot Secret Sauce: Learning from Consequences")
    print("=" * 70)

    epochs, hit_rates, left_weights, right_weights = calculate_epoch_stats(data, epoch_size)

    # Learning curve
    print("\n" + "-" * 70)
    print("LEARNING CURVE - Hit Rate by Epoch")
    print("-" * 70)

    print("\nEpoch | Hits | Rate  | Learning Curve")
    print("------|------|-------|" + "-" * 50)

    for i, (epoch, rate, lw, rw) in enumerate(zip(epochs, hit_rates, left_weights, right_weights)):
        bar_len = int(rate * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)

        # Color code based on improvement
        if rate > 0.9:
            status = " ** EXPERT **"
        elif rate > 0.7:
            status = " * Learning! *"
        elif rate > 0.5:
            status = " (improving)"
        else:
            status = " (random)"

        print(f"{epoch:5} | L{lw:3} R{rw:3} | {rate:5.1%} | [{bar}]{status}")

    # Weight progression
    print("\n" + "-" * 70)
    print("SYNAPTIC WEIGHT GROWTH")
    print("-" * 70)

    print("\nEpoch | Left | Right | Balance")
    print("------|------|-------|" + "-" * 40)

    for epoch, lw, rw in zip(epochs, left_weights, right_weights):
        max_w = max(abs(lw), abs(rw), 1)
        left_bar = int((lw / max_w) * 15) if max_w > 0 else 0
        right_bar = int((rw / max_w) * 15) if max_w > 0 else 0

        balance = "[" + "#" * abs(left_bar) + "|" + "*" * abs(right_bar) + "]"
        print(f"{epoch:5} | {lw:4} | {rw:5} | {balance}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    first_rate = hit_rates[0] if hit_rates else 0
    last_rate = hit_rates[-1] if hit_rates else 0
    improvement = last_rate - first_rate

    print(f"\nStarting Hit Rate: {first_rate:.1%}")
    print(f"Final Hit Rate:    {last_rate:.1%}")
    print(f"Improvement:       {improvement:+.1%}")

    if last_rate > 0.9:
        print("\n  EXCELLENT: >90% hit rate achieved!")
        print("  The brain learned to play Pong from pure trial-and-error!")
    elif last_rate > 0.7:
        print("\n  GOOD: >70% hit rate - clear learning detected!")
    elif improvement > 0.2:
        print("\n  LEARNING: Improvement detected, may need more training")
    else:
        print("\n  NEEDS MORE TRAINING")

    # The secret sauce
    print("\n" + "-" * 70)
    print("THE BITTERBOT SECRET SAUCE")
    print("-" * 70)
    print("""
No Replay Buffer. No Gradient Descent. No Matrix Math.

Just:
  - 1 byte eligibility trace per synapse
  - Integer multiply + divide for reward
  - Biological temporal credit assignment

The brain learned cause-and-effect purely from:
  - Action fires motor neuron -> sets eligibility trace
  - Ball hits paddle -> reward floods eligible synapses
  - Ball misses -> punishment weakens eligible synapses
""")


def matplotlib_visualization(data, epoch_size=500):
    """Create matplotlib visualization."""
    epochs, hit_rates, left_weights, right_weights = calculate_epoch_stats(data, epoch_size)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    fig.suptitle("OPERANT CONDITIONING: Learning from Consequences\n"
                 "No Coach, No Replay Buffer - Just Eligibility Traces",
                 fontsize=14, fontweight='bold')

    # ========================================
    # Panel 1: Learning Curve
    # ========================================
    ax1 = axes[0]
    ax1.set_ylabel('Hit Rate', fontsize=10)
    ax1.set_title('The Learning Curve - Trial and Error')

    ax1.plot(epochs, hit_rates, color='#27AE60', linewidth=3,
             marker='o', markersize=6, label='Hit Rate')
    ax1.fill_between(epochs, 0, hit_rates, color='#27AE60', alpha=0.3)

    # Reference lines
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random Baseline (50%)')
    ax1.axhline(y=0.9, color='#F39C12', linestyle='--', alpha=0.7, label='Target (90%)')

    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right')

    # Annotate start and end
    if hit_rates:
        ax1.annotate(f'Start: {hit_rates[0]:.0%}',
                    xy=(epochs[0], hit_rates[0]),
                    xytext=(epochs[0] + 1, hit_rates[0] + 0.15),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray'))
        ax1.annotate(f'End: {hit_rates[-1]:.0%}',
                    xy=(epochs[-1], hit_rates[-1]),
                    xytext=(epochs[-1] - 2, hit_rates[-1] - 0.15),
                    fontsize=9, ha='right',
                    arrowprops=dict(arrowstyle='->', color='gray'))

    # ========================================
    # Panel 2: Synaptic Weights
    # ========================================
    ax2 = axes[1]
    ax2.set_ylabel('Synaptic Weight', fontsize=10)
    ax2.set_title('Motor Neuron Weight Growth (Eligibility + Reward)')

    ax2.plot(epochs, left_weights, color='#E74C3C', linewidth=2,
             marker='s', markersize=4, label='Left Motor', alpha=0.8)
    ax2.plot(epochs, right_weights, color='#3498DB', linewidth=2,
             marker='^', markersize=4, label='Right Motor', alpha=0.8)

    ax2.fill_between(epochs, 0, left_weights, color='#E74C3C', alpha=0.2)
    ax2.fill_between(epochs, 0, right_weights, color='#3498DB', alpha=0.2)

    ax2.legend(loc='upper left')
    ax2.set_ylim(bottom=0)

    # ========================================
    # Panel 3: Improvement Rate
    # ========================================
    ax3 = axes[2]
    ax3.set_ylabel('Improvement', fontsize=10)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_title('Learning Rate (Change from Previous Epoch)')

    if len(hit_rates) > 1:
        improvements = [0] + [hit_rates[i] - hit_rates[i-1] for i in range(1, len(hit_rates))]
        colors = ['#27AE60' if x >= 0 else '#E74C3C' for x in improvements]
        ax3.bar(epochs, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax3.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_file = "learning_curve.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved visualization to: {output_file}")

    plt.show()


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "operant_learning.csv"

    try:
        data = load_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'")
        print("Run the test_operant simulation first.")
        sys.exit(1)

    print(f"Loaded {len(data['Tick'])} ticks from {csv_file}")

    # Determine epoch size from data
    total_ticks = len(data['Tick'])
    epoch_size = 500 if total_ticks <= 10000 else 1000

    if HAS_MATPLOTLIB:
        matplotlib_visualization(data, epoch_size)
    else:
        text_visualization(data, epoch_size)

    # Always print summary
    text_visualization(data, epoch_size)


if __name__ == "__main__":
    main()

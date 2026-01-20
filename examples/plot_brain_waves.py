#!/usr/bin/env python3
"""
THE GRAND FINALE VISUALIZER
============================

Visualizes the brain's internal state during the "First Contact" simulation.

Creates a 3-panel plot showing:
  1. Input Status - What the brain is "seeing"
  2. Column Activity - When the brain "shouts" (spikes)
  3. Memory Trace - The "subconscious" priming state

Usage:
    python3 plot_brain_waves.py [brain_activity.csv]

Author: BP-AGI Project
"""

import sys
import csv

# Try to import matplotlib, provide helpful message if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Will generate text-based visualization.")

def load_csv(filename):
    """Load CSV data into a dictionary of lists."""
    data = {
        'Tick': [],
        'Phase': [],
        'Input': [],
        'RequestNeuron_Fired': [],
        'Column0_Activity': [],
        'Column0_Memory': [],
        'Column1_Activity': [],
        'Column1_Memory': [],
        'Column0_Allocated': [],
        'Column1_Allocated': [],
    }

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['Tick'].append(int(row['Tick']))
            data['Phase'].append(row['Phase'])
            data['Input'].append(row['Input'])
            data['RequestNeuron_Fired'].append(int(row['RequestNeuron_Fired']))
            data['Column0_Activity'].append(int(row['Column0_Activity']))
            data['Column0_Memory'].append(int(row['Column0_Memory']))
            data['Column1_Activity'].append(int(row['Column1_Activity']))
            data['Column1_Memory'].append(int(row['Column1_Memory']))
            data['Column0_Allocated'].append(int(row['Column0_Allocated']))
            data['Column1_Allocated'].append(int(row['Column1_Allocated']))

    return data


def text_visualization(data):
    """Create ASCII visualization when matplotlib is not available."""
    print("\n" + "=" * 70)
    print("THE GRAND FINALE: BRAIN WAVE VISUALIZATION (Text Mode)")
    print("=" * 70)

    # Phase boundaries
    phases = [
        (0, 100, 'Awakening'),
        (100, 200, 'First Contact'),
        (200, 300, 'The Vanishing'),
        (300, 400, 'The Return'),
        (400, 500, 'The Stranger'),
        (500, 600, 'The Dream'),
    ]

    for start, end, name in phases:
        print(f"\n{'='*70}")
        print(f"  {name.upper()} (Ticks {start}-{end})")
        print(f"{'='*70}")

        # Get data for this phase
        phase_ticks = [i for i, t in enumerate(data['Tick']) if start <= t < end]

        if not phase_ticks:
            continue

        # Count fires in this phase
        col0_fires = sum(data['Column0_Activity'][i] for i in phase_ticks)
        col1_fires = sum(data['Column1_Activity'][i] for i in phase_ticks)
        request_fires = sum(data['RequestNeuron_Fired'][i] for i in phase_ticks)

        # Get memory range
        col0_mem_max = max(data['Column0_Memory'][i] for i in phase_ticks)
        col1_mem_max = max(data['Column1_Memory'][i] for i in phase_ticks)

        # Input for this phase
        input_type = data['Input'][phase_ticks[0]]

        print(f"\n  Input: {input_type}")
        print(f"  Column 0 (Triangle) fires: {col0_fires}")
        print(f"  Column 1 (Square) fires:   {col1_fires}")
        print(f"  Request Neuron fires:      {request_fires}")
        print(f"  Column 0 peak memory:      {col0_mem_max}")
        print(f"  Column 1 peak memory:      {col1_mem_max}")

        # ASCII activity chart (sampled every 10 ticks)
        print("\n  Activity Timeline (every 10 ticks):")
        print("  " + "-" * 52)
        print("  Tick: ", end="")
        for t in range(start, end, 10):
            print(f"{t:5}", end="")
        print()

        # Column 0 activity
        print("  Col0: ", end="")
        for t in range(start, end, 10):
            idx = [i for i, tick in enumerate(data['Tick']) if tick == t]
            if idx:
                fires = sum(data['Column0_Activity'][i] for i in range(idx[0], min(idx[0]+10, len(data['Tick']))))
                if fires > 0:
                    print("  |||", end="")
                else:
                    print("    .", end="")
            else:
                print("     ", end="")
        print()

        # Column 1 activity
        print("  Col1: ", end="")
        for t in range(start, end, 10):
            idx = [i for i, tick in enumerate(data['Tick']) if tick == t]
            if idx:
                fires = sum(data['Column1_Activity'][i] for i in range(idx[0], min(idx[0]+10, len(data['Tick']))))
                if fires > 0:
                    print("  |||", end="")
                else:
                    print("    .", end="")
            else:
                print("     ", end="")
        print()

        # Memory trace (bar chart)
        print("\n  Memory Trace:")
        max_mem = max(col0_mem_max, col1_mem_max, 1)
        for level in range(5, 0, -1):
            threshold = (level / 5) * max_mem
            print("  ", end="")
            for t in range(start, end, 10):
                idx = [i for i, tick in enumerate(data['Tick']) if tick == t]
                if idx:
                    mem0 = data['Column0_Memory'][idx[0]]
                    mem1 = data['Column1_Memory'][idx[0]]
                    if mem0 >= threshold and mem1 >= threshold:
                        print("  +X+", end="")
                    elif mem0 >= threshold:
                        print("  [#]", end="")
                    elif mem1 >= threshold:
                        print("  <O>", end="")
                    else:
                        print("     ", end="")
                else:
                    print("     ", end="")
            print()
        print("  " + "-" * 52)
        print("  Legend: [#]=Col0  <O>=Col1  +X+=Both")


def matplotlib_visualization(data):
    """Create matplotlib visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 2]})

    fig.suptitle("THE GRAND FINALE: A Day in the Life of an Artificial Mind",
                 fontsize=14, fontweight='bold', y=0.98)

    # Color scheme
    colors = {
        'Silence': '#2C3E50',
        'Triangle': '#E74C3C',
        'Square': '#3498DB',
        'Column0': '#E74C3C',
        'Column1': '#3498DB',
        'Request': '#F39C12',
        'Memory0': '#E74C3C',
        'Memory1': '#3498DB',
    }

    # Phase boundaries
    phases = [
        (0, 100, 'Awakening', '#34495E'),
        (100, 200, 'First Contact', '#E74C3C'),
        (200, 300, 'The Vanishing', '#9B59B6'),
        (300, 400, 'The Return', '#27AE60'),
        (400, 500, 'The Stranger', '#3498DB'),
        (500, 600, 'The Dream', '#2C3E50'),
    ]

    ticks = data['Tick']

    # ========================================
    # PANEL 1: Input Status
    # ========================================
    ax1 = axes[0]
    ax1.set_ylabel('Input', fontsize=10)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Silence', 'Triangle', 'Square'])

    input_map = {'Silence': 0, 'Triangle': 1, 'Square': 2}

    for i in range(len(ticks) - 1):
        input_type = data['Input'][i]
        input_val = input_map.get(input_type, 0)
        color = colors.get(input_type, '#7F8C8D')
        ax1.axvspan(ticks[i], ticks[i+1],
                    ymin=(input_val) / 3,
                    ymax=(input_val + 1) / 3,
                    alpha=0.8, color=color)

    for start, end, name, color in phases:
        ax1.axvspan(start, end, alpha=0.15, color=color)
        ax1.text((start + end) / 2, 2.3, name, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color)

    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=2, color='gray', linestyle='--', alpha=0.3)

    # ========================================
    # PANEL 2: Column Activity (Spikes)
    # ========================================
    ax2 = axes[1]
    ax2.set_ylabel('Activity\n(Spikes)', fontsize=10)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Request', 'Col 0\n(Triangle)', 'Col 1\n(Square)'])

    for start, end, name, color in phases:
        ax2.axvspan(start, end, alpha=0.08, color=color)

    # Plot spikes
    for i, t in enumerate(ticks):
        if data['RequestNeuron_Fired'][i]:
            ax2.plot([t, t], [-0.3, 0.3], color=colors['Request'], linewidth=2, alpha=0.9)
        if data['Column0_Activity'][i]:
            ax2.plot([t, t], [0.7, 1.3], color=colors['Column0'], linewidth=1.5, alpha=0.8)
        if data['Column1_Activity'][i]:
            ax2.plot([t, t], [1.7, 2.3], color=colors['Column1'], linewidth=1.5, alpha=0.8)

    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=2, color='gray', linestyle='--', alpha=0.3)

    # ========================================
    # PANEL 3: Memory Trace
    # ========================================
    ax3 = axes[2]
    ax3.set_ylabel('Memory Trace\n(Primed Neurons)', fontsize=10)
    ax3.set_xlabel('Time (Ticks)', fontsize=10)

    for start, end, name, color in phases:
        ax3.axvspan(start, end, alpha=0.08, color=color)

    ax3.fill_between(ticks, 0, data['Column0_Memory'],
                     alpha=0.4, color=colors['Memory0'], label='Column 0 (Triangle)')
    ax3.plot(ticks, data['Column0_Memory'],
             color=colors['Memory0'], linewidth=1.5, alpha=0.9)

    ax3.fill_between(ticks, 0, data['Column1_Memory'],
                     alpha=0.4, color=colors['Memory1'], label='Column 1 (Square)')
    ax3.plot(ticks, data['Column1_Memory'],
             color=colors['Memory1'], linewidth=1.5, alpha=0.9)

    ax3.legend(loc='upper right', fontsize=9)

    max_memory = max(max(data['Column0_Memory']), max(data['Column1_Memory']))
    ax3.set_ylim(0, max(max_memory * 1.2, 10))

    plt.xlim(0, 600)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = "brain_waves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved visualization to: {output_file}")

    plt.show()


def print_summary(data):
    """Print summary statistics."""
    print("\n" + "=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Total ticks: {len(data['Tick'])}")
    print(f"Column 0 (Triangle) fires: {sum(data['Column0_Activity'])}")
    print(f"Column 1 (Square) fires: {sum(data['Column1_Activity'])}")
    print(f"Request neuron fires: {sum(data['RequestNeuron_Fired'])}")
    print(f"Peak Column 0 memory: {max(data['Column0_Memory'])}")
    print(f"Peak Column 1 memory: {max(data['Column1_Memory'])}")

    print("\n" + "-" * 50)
    print("BEHAVIORAL VERIFICATION")
    print("-" * 50)

    # Check First Contact
    first_contact_fires = any(
        data['Phase'][i] == 'FirstContact' and data['Column0_Activity'][i]
        for i in range(len(data['Tick']))
    )
    if first_contact_fires:
        print("[PASS] First Contact: Triangle learned and recognized")
    else:
        print("[FAIL] First Contact: Triangle not recognized")

    # Check Vanishing
    vanishing_memory = [
        data['Column0_Memory'][i]
        for i in range(len(data['Tick']))
        if data['Phase'][i] == 'Vanishing'
    ]
    avg_vanishing = sum(vanishing_memory) / len(vanishing_memory) if vanishing_memory else 0
    if avg_vanishing > 0:
        print(f"[PASS] The Vanishing: Memory persists (avg {avg_vanishing:.1f} primed neurons)")
    else:
        print("[FAIL] The Vanishing: Memory did not persist")

    # Check Return
    return_indices = [i for i in range(len(data['Tick'])) if data['Phase'][i] == 'Return']
    return_fires = [i for i in return_indices if data['Column0_Activity'][i]]
    if return_fires:
        first_fire_tick = data['Tick'][return_fires[0]]
        relative_tick = first_fire_tick - 300
        if relative_tick < 10:
            print(f"[PASS] The Return: Hot start achieved ({relative_tick} ticks)")
        else:
            print(f"[WARN] The Return: Slower start ({relative_tick} ticks)")
    else:
        print("[FAIL] The Return: Triangle not recognized")

    # Check Stranger
    stranger_fires = any(
        data['Phase'][i] == 'Stranger' and data['Column1_Activity'][i]
        for i in range(len(data['Tick']))
    )
    if stranger_fires:
        print("[PASS] The Stranger: Square learned in separate column")
    else:
        print("[FAIL] The Stranger: Square not learned")

    # Check Dream
    dream_memory = [
        data['Column0_Memory'][i]
        for i in range(len(data['Tick']))
        if data['Phase'][i] == 'Dream'
    ]
    avg_dream = sum(dream_memory) / len(dream_memory) if dream_memory else 0
    print(f"[INFO] The Dream: Average memory trace = {avg_dream:.1f} primed neurons")

    print("\n" + "=" * 50)
    print("\"The mind is not a vessel to be filled,")
    print(" but a fire to be kindled.\" - Plutarch")
    print("=" * 50)


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "brain_activity.csv"

    try:
        data = load_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'")
        print("Run the grand_finale simulation first:")
        print("  ./grand_finale")
        sys.exit(1)

    print(f"Loaded {len(data['Tick'])} ticks from {csv_file}")

    if HAS_MATPLOTLIB:
        matplotlib_visualization(data)
    else:
        text_visualization(data)

    print_summary(data)


if __name__ == "__main__":
    main()

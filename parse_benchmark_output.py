#!/usr/bin/env python3
"""
Parse Honeybee/Dragonfly benchmark stdout into JSON results.

Usage:
    python parse_benchmark_output.py benchmark_output.txt > results.json
    # Or pipe directly:
    ./arc_honeybee arc_eval.bin | tee output.txt
    python parse_benchmark_output.py output.txt -o honeybee_results.json
"""

import re
import json
import sys
import argparse

def parse_benchmark_output(text):
    """Parse benchmark stdout and extract results."""
    results = {}

    # Pattern for Honeybee output:
    # [  1/120] 0934a4d8 (4 train)... 80.5%  (1121092ms)
    # [  2/120] 135a2760 (2 train)... 99.4% [PASS] (250870ms)

    honeybee_pattern = r'\[\s*(\d+)/(\d+)\]\s+(\w+)\s+\((\d+)\s+train\)\.+\s+([\d.]+)%\s*(\[PASS\])?\s*\((\d+)ms\)'

    # Pattern for Dragonfly output (2-attempt):
    # [  1/120] 0934a4d8 (4 train) Try1: 81% -> Try2: 81% (787750ms)
    # [  2/120] 135a2760 (2 train) 99.4% [PASS] (384333ms)

    dragonfly_pattern = r'\[\s*(\d+)/(\d+)\]\s+(\w+)\s+\((\d+)\s+train\)\s+(?:Try1:\s*([\d.]+)%\s*->\s*Try2:\s*([\d.]+)%|([\d.]+)%\s*(\[PASS\])?)\s*\((\d+)ms\)'

    for line in text.split('\n'):
        # Try Dragonfly pattern first (more specific)
        match = re.search(dragonfly_pattern, line)
        if match:
            task_num, total, task_id, num_train, try1, try2, single_score, passed, time_ms = match.groups()

            if try1 and try2:
                # 2-attempt result
                accuracy = max(float(try1), float(try2)) / 100.0
                passed = accuracy >= 1.0
            else:
                # Single attempt (passed on first try)
                accuracy = float(single_score) / 100.0
                passed = passed is not None

            results[task_id] = {
                'task_num': int(task_num),
                'num_train': int(num_train),
                'accuracy': accuracy,
                'passed': passed,
                'time_ms': int(time_ms),
                'attempt1': float(try1) / 100.0 if try1 else accuracy,
                'attempt2': float(try2) / 100.0 if try2 else accuracy,
            }
            continue

        # Try Honeybee pattern
        match = re.search(honeybee_pattern, line)
        if match:
            task_num, total, task_id, num_train, accuracy, passed, time_ms = match.groups()

            results[task_id] = {
                'task_num': int(task_num),
                'num_train': int(num_train),
                'accuracy': float(accuracy) / 100.0,
                'passed': passed is not None,
                'time_ms': int(time_ms),
            }

    return results

def main():
    parser = argparse.ArgumentParser(description='Parse benchmark output to JSON')
    parser.add_argument('input', nargs='?', default='-',
                        help='Input file (or - for stdin)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output JSON file (default: stdout)')
    args = parser.parse_args()

    # Read input
    if args.input == '-':
        text = sys.stdin.read()
    else:
        with open(args.input) as f:
            text = f.read()

    # Parse
    results = parse_benchmark_output(text)

    # Summary
    if results:
        passed = sum(1 for r in results.values() if r['passed'])
        total = len(results)
        avg_acc = sum(r['accuracy'] for r in results.values()) / total * 100

        print(f"# Parsed {total} tasks: {passed} passed ({100*passed/total:.1f}%), avg {avg_acc:.1f}%",
              file=sys.stderr)

    # Output
    output_json = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"# Saved to: {args.output}", file=sys.stderr)
    else:
        print(output_json)

if __name__ == '__main__':
    main()

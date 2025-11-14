#!/usr/bin/env python3
"""Summarize emissions and energy from CodeCarbon's CSV.

Usage:
    python scripts/calculate_emissions.py IMDB/emissions.csv

Produces a brief summary printed to stdout and writes `scripts/emissions_summary.csv`.
"""
import sys
import pandas as pd
from pathlib import Path


def summarize(path):
    df = pd.read_csv(path)
    if df.empty:
        print('No data in', path)
        return

    # group stats by project
    grp = df.groupby('project_name').agg(
        runs=('run_id', 'count'),
        mean_energy_kwh=('energy_consumed', 'mean'),
        std_energy_kwh=('energy_consumed', 'std'),
        mean_emissions_kg=('emissions', 'mean'),
        std_emissions_kg=('emissions', 'std'),
        mean_duration_s=('duration', 'mean'),
    ).reset_index()

    # compute pairwise ratios if two projects present
    out = grp.copy()
    if len(grp) >= 2:
        # pick the first two for ratio example
        a, b = grp.iloc[0], grp.iloc[1]
        ratio_energy = b['mean_energy_kwh'] / a['mean_energy_kwh'] if a['mean_energy_kwh'] > 0 else float('inf')
        pct_increase_energy = (ratio_energy - 1.0) * 100
        ratio_emissions = b['mean_emissions_kg'] / a['mean_emissions_kg'] if a['mean_emissions_kg'] > 0 else float('inf')
        pct_increase_emissions = (ratio_emissions - 1.0) * 100
        print(f"Comparison sample: {a['project_name']} -> {b['project_name']}")
        print(f"  Energy: {ratio_energy:.2f}x ({pct_increase_energy:.1f}% increase)")
        print(f"  Emissions: {ratio_emissions:.2f}x ({pct_increase_emissions:.1f}% increase)")

    # convert emissions to grams for readability
    out['mean_emissions_g'] = out['mean_emissions_kg'] * 1000
    out['std_emissions_g'] = out['std_emissions_kg'] * 1000

    # save summary
    out_path = Path('scripts') / 'emissions_summary.csv'
    out.to_csv(out_path, index=False)
    print('\nSaved summary to', out_path)
    print('\nSummary:')
    print(out.to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/calculate_emissions.py path/to/emissions.csv')
        sys.exit(1)
    summarize(sys.argv[1])

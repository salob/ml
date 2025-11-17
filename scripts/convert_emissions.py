#!/usr/bin/env python3
"""Convert emissions.csv to use more readable units.

Converts:
- emissions (kg) → emissions_g (grams)
- energy_consumed (kWh) → energy_consumed_wh (Watt-hours)
- cpu_energy, gpu_energy, ram_energy (kWh) → _wh (Watt-hours)

Usage:
    python scripts/convert_emissions.py

Reads: logs/emissions.csv
Writes: logs/emissions_readable.csv
"""
import pandas as pd
from pathlib import Path


def convert_emissions():
    input_path = Path('logs') / 'emissions.csv'
    output_path = Path('logs') / 'emissions_readable.csv'
    
    # Read the emissions data
    df = pd.read_csv(input_path)
    
    if df.empty:
        print(f'No data in {input_path}')
        return
    
    # Create a copy for conversion
    df_readable = df.copy()
    
    # Convert emissions from kg to grams
    df_readable['emissions_g'] = df['emissions'] * 1000
    
    # Convert energy from kWh to Watt-hours (Wh)
    # 1 kWh = 1000 Wh
    df_readable['energy_consumed_wh'] = df['energy_consumed'] * 1000
    df_readable['cpu_energy_wh'] = df['cpu_energy'] * 1000
    df_readable['gpu_energy_wh'] = df['gpu_energy'] * 1000
    df_readable['ram_energy_wh'] = df['ram_energy'] * 1000
    
    # Drop the original kg/kWh columns to avoid confusion
    columns_to_drop = [
        'emissions',
        'energy_consumed',
        'cpu_energy',
        'gpu_energy',
        'ram_energy'
    ]
    df_readable = df_readable.drop(columns=columns_to_drop)
    
    # Reorder columns to put readable units near the front
    # Get all column names
    cols = df_readable.columns.tolist()
    
    # Define preferred order for key columns
    priority_cols = [
        'timestamp',
        'project_name',
        'run_id',
        'duration',
        'emissions_g',
        'energy_consumed_wh',
        'cpu_energy_wh',
        'gpu_energy_wh',
        'ram_energy_wh'
    ]
    
    # Build new column order: priority columns first, then the rest
    remaining_cols = [col for col in cols if col not in priority_cols]
    new_order = [col for col in priority_cols if col in cols] + remaining_cols
    df_readable = df_readable[new_order]
    
    # Save to CSV
    df_readable.to_csv(output_path, index=False)
    
    print(f'Converted {len(df_readable)} rows from {input_path}')
    print(f'Saved readable version to {output_path}')
    print('\nConversions applied:')
    print('  - emissions (kg) → emissions_g (grams)')
    print('  - energy_consumed (kWh) → energy_consumed_wh (Watt-hours)')
    print('  - cpu/gpu/ram_energy (kWh) → _wh (Watt-hours)')
    print('\nSample of converted data:')
    print(df_readable[['project_name', 'duration', 'emissions_g', 'energy_consumed_wh']].head().to_string(index=False))


if __name__ == '__main__':
    convert_emissions()

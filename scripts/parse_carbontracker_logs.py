#!/usr/bin/env python3
"""
Parse CarbonTracker log files and create a readable CSV

Usage: python scripts/parse_carbontracker_logs.py
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime


def parse_carbontracker_log(log_path):
    """Parse a single CarbonTracker output log file"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract model name and timestamp from filename
    # Format: ct_imdb_MODEL__PID.TIMESTAMP_carbontracker_output.log
    filename = log_path.name
    match = re.match(r'ct_imdb_(\w+)__[\d.]+_([\dT-]+Z)_carbontracker_output\.log', filename)
    
    if not match:
        return None
    
    model_name = match.group(1)
    timestamp_str = match.group(2)
    
    # Parse timestamp (format: 2025-11-15T193450Z)
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H%M%SZ')
    except:
        timestamp = None
    
    # Extract actual consumption data (not predicted)
    # Look for "Actual consumption" section specifically
    actual_section = re.search(r'Actual consumption.*?Time:\s*([\d:]+).*?Energy:\s*([\d.]+)\s*kWh.*?CO2eq:\s*([\d.]+)\s*g', content, re.DOTALL)
    
    # If no "Actual consumption" section, try to find any consumption data
    if actual_section:
        time_str = actual_section.group(1)
        energy_kwh = float(actual_section.group(2))
        co2_g = float(actual_section.group(3))
    else:
        # Fallback: find first occurrence (for logs without prediction)
        time_match = re.search(r'Time:\s*([\d:]+)', content)
        energy_match = re.search(r'Energy:\s*([\d.]+)\s*kWh', content)
        co2_match = re.search(r'CO2eq:\s*([\d.]+)\s*g', content)
        
        if not (time_match and energy_match and co2_match):
            return None
        
        time_str = time_match.group(1)
        energy_kwh = float(energy_match.group(1))
        co2_g = float(co2_match.group(1))
    
    carbon_intensity_match = re.search(r'Average carbon intensity during training was ([\d.]+) gCO2eq/kWh', content)
    
    if not carbon_intensity_match:
        return None
    
    carbon_intensity = float(carbon_intensity_match.group(1))
    
    # Parse time duration (format: H:MM:SS or MM:SS)
    time_parts = time_str.split(':')
    if len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
    elif len(time_parts) == 2:
        hours = 0
        minutes, seconds = map(int, time_parts)
    else:
        return None
    
    duration_seconds = hours * 3600 + minutes * 60 + seconds
    
    # Convert to readable units
    energy_wh = energy_kwh * 1000  # kWh to Wh
    
    # CarbonTracker applies PUE=1.58 by default (matching CodeCarbon)
    pue = 1.58
    
    # Parse the detailed .log file for GPU/CPU power breakdown
    detailed_log_path = log_path.parent / log_path.name.replace('_output.log', '.log')
    gpu_power = 0.0
    cpu_power = 0.0
    
    if detailed_log_path.exists():
        try:
            with open(detailed_log_path, 'r') as f:
                detailed_content = f.read()
            
            # Extract all GPU and CPU power measurements per epoch
            gpu_powers = re.findall(r'Average power usage \(W\) for gpu: ([\d.]+)', detailed_content)
            cpu_powers = re.findall(r'Average power usage \(W\) for cpu: ([\d.]+)', detailed_content)
            
            if gpu_powers:
                gpu_power = sum(float(p) for p in gpu_powers) / len(gpu_powers)
            if cpu_powers:
                cpu_power = sum(float(p) for p in cpu_powers) / len(cpu_powers)
        except Exception as e:
            print(f"Warning: Could not parse detailed log {detailed_log_path.name}: {e}")
    
    return {
        'timestamp': timestamp.isoformat() if timestamp else timestamp_str,
        'model_name': f'IMDB_{model_name.title()}',
        'duration_seconds': duration_seconds,
        'duration_formatted': time_str,
        'energy_kwh': energy_kwh,
        'energy_wh': energy_wh,
        'co2_g': co2_g,
        'pue_applied': pue,
        'carbon_intensity_gco2_per_kwh': carbon_intensity,
        'gpu_power_w': gpu_power,
        'cpu_power_w': cpu_power,
        'log_file': filename
    }


def parse_all_carbontracker_logs():
    """Parse all CarbonTracker output logs in logs directory"""
    logs_dir = Path('carbontracker_logs')
    
    # Find all CarbonTracker output log files
    log_files = sorted(logs_dir.glob('ct_imdb_*_carbontracker_output.log'))
    
    print(f"Found {len(log_files)} CarbonTracker log files")
    
    data = []
    for log_file in log_files:
        print(f"Parsing {log_file.name}...")
        parsed = parse_carbontracker_log(log_file)
        if parsed:
            data.append(parsed)
        else:
            print(f"Could not parse {log_file.name}")
    
    if not data:
        print("No data parsed!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Save to CSV
    output_path = Path('logs') / 'carbontracker_readable.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nParsed {len(df)} runs from CarbonTracker logs")
    print(f"Saved to {output_path}")
    print(f"\nBoth CodeCarbon and CarbonTracker now use PUE=1.58")
    
    # Print summary by model
    print("\nSummary by model:")
    summary = df.groupby('model_name').agg({
        'duration_seconds': ['count', 'mean'],
        'energy_wh': 'mean',
        'co2_g': 'mean'
    }).round(4)
    print(summary)
    
    # Show first few rows
    print("\nSample data:")
    print(df[['timestamp', 'model_name', 'duration_formatted', 'energy_wh', 'co2_g']].head(10))


if __name__ == '__main__':
    parse_all_carbontracker_logs()

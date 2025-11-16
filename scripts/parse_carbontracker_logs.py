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
    
    # CarbonTracker applies PUE=1.58 by default
    # For laptops/local machines, PUE should be 1.0
    # Provide both raw (with PUE) and adjusted (without PUE) values
    pue = 1.58
    energy_kwh_adjusted = energy_kwh / pue
    energy_wh_adjusted = energy_kwh_adjusted * 1000
    co2_g_adjusted = co2_g / pue
    
    return {
        'timestamp': timestamp.isoformat() if timestamp else timestamp_str,
        'model_name': f'IMDB_{model_name.title()}',
        'duration_seconds': duration_seconds,
        'duration_formatted': time_str,
        'energy_kwh_raw': energy_kwh,
        'energy_wh_raw': energy_wh,
        'co2_g_raw': co2_g,
        'pue_applied': pue,
        'energy_kwh_adjusted': energy_kwh_adjusted,
        'energy_wh_adjusted': energy_wh_adjusted,
        'co2_g_adjusted': co2_g_adjusted,
        'carbon_intensity_gco2_per_kwh': carbon_intensity,
        'log_file': filename
    }


def parse_all_carbontracker_logs():
    """Parse all CarbonTracker output logs in logs directory"""
    logs_dir = Path('logs')
    
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
            print(f"  ⚠️  Could not parse {log_file.name}")
    
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
    
    print(f"\n✅ Parsed {len(df)} runs from CarbonTracker logs")
    print(f"📊 Saved to {output_path}")
    print(f"\nℹ️  CarbonTracker applies PUE=1.58 by default (data center assumption)")
    print(f"   For local/laptop runs, use 'adjusted' columns (PUE=1.0)")
    
    # Print summary by model (using adjusted values)
    print("\n📈 Summary by model (PUE-adjusted for local machine):")
    summary = df.groupby('model_name').agg({
        'duration_seconds': ['count', 'mean'],
        'energy_wh_adjusted': 'mean',
        'co2_g_adjusted': 'mean'
    }).round(4)
    print(summary)
    
    # Show first few rows
    print("\n📋 Sample data (PUE-adjusted):")
    print(df[['timestamp', 'model_name', 'duration_formatted', 'energy_wh_adjusted', 'co2_g_adjusted']].head(10))


if __name__ == '__main__':
    parse_all_carbontracker_logs()

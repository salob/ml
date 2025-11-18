#!/usr/bin/env python3
"""
Multiple Experiment Runner

This script runs each model multiple times with different seeds to collect
statistics for comparison. Useful for generating data for the comparison report.

Usage: python run_multiple_experiments.py [--runs N]
"""

import subprocess
import os
import sys
import time
from datetime import datetime
import argparse

def run_experiment(script_path, run_number, total_runs, seed):
    """Run a single experiment script"""
    print(f"\n{'='*60}")
    print(f"Running {script_path} - Run {run_number}/{total_runs} (seed: {seed})")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Pass the seed directly as command line argument
    env = os.environ.copy()
    env['EXPERIMENT_RUN'] = str(run_number)
    
    try:
        result = subprocess.run([
            sys.executable, script_path, '--seed', str(seed)
        ], env=env, capture_output=False, text=True, check=True)
        
        duration = time.time() - start_time
        print(f"\n✓ Completed in {duration:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments for model comparison')
    parser.add_argument('--runs', type=int, default=5, 
                       help='Number of runs for each model (default: 5)')
    parser.add_argument('--seed', type=int, default=2025,
                       help='Starting seed value (default: 2025)')
    parser.add_argument('--models', nargs='+', 
                       choices=['logreg', 'dense', 'transformer', 'all'],
                       default=['all'],
                       help='Which models to run (default: all)')
    
    args = parser.parse_args()
    
    # Define model scripts
    scripts = {
        'logreg': 'IMDB/imdb_logreg.py',
        'dense': 'IMDB/imdb_dense.py', 
        'transformer': 'IMDB/imdb_transformer.py'
    }
    
    # Determine which scripts to run
    if 'all' in args.models:
        scripts_to_run = scripts
    else:
        scripts_to_run = {k: v for k, v in scripts.items() if k in args.models}
    
    print(f"Running {args.runs} experiments for each model: {list(scripts_to_run.keys())}")
    print(f"Total experiments: {len(scripts_to_run) * args.runs}")
    print(f"Starting seed: {args.seed}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check that all scripts exist
    missing_scripts = []
    for name, path in scripts_to_run.items():
        if not os.path.exists(path):
            missing_scripts.append(f"{name}: {path}")
    
    if missing_scripts:
        print("Error: The following scripts were not found:")
        for script in missing_scripts:
            print(f"  - {script}")
        sys.exit(1)
    
    # Ensure logs directory exists
    os.makedirs("codecarbon_logs", exist_ok=True)
    
    # Run experiments
    results = {}
    overall_start = time.time()
    
    for model_name, script_path in scripts_to_run.items():
        results[model_name] = []
        print(f"\n{'#'*60}")
        print(f"STARTING {model_name.upper()} EXPERIMENTS")
        print(f"{'#'*60}")
        
        for run in range(1, args.runs + 1):
            current_seed = args.seed + (run - 1)  # Reset seed sequence for each model
            success = run_experiment(script_path, run, args.runs, current_seed)
            results[model_name].append(success)
            
            # Small delay between runs
            if run < args.runs:
                print("Waiting 5 seconds before next run...")
                time.sleep(5)
    
    # Summary
    total_duration = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for model_name, run_results in results.items():
        successful = sum(run_results)
        total = len(run_results)
        print(f"{model_name:12}: {successful}/{total} successful")
    
    total_successful = sum(sum(run_results) for run_results in results.values())
    total_experiments = sum(len(run_results) for run_results in results.values())
    
    print(f"{'Overall':<12}: {total_successful}/{total_experiments} successful")
    
    if total_successful > 0:
        print(f"\nNow run: python scripts/generate_comparison_report.py")
        print("to generate the comparison report with all collected data.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick test script to generate a comparison report with current data
"""

import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

if __name__ == "__main__":
    from scripts.generate_comparison_report import generate_pdf_report
    
    print("Generating comparison report...")
    try:
        generate_pdf_report()
        print("\nSuccess! Check the logs/ directory for the PDF report.")
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)
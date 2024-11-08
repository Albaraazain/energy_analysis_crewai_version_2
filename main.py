# main.py

import argparse
import json
import logging
from datetime import datetime, timedelta
import sys
from typing import Dict
import random

from src.core.analyzer import EnergyAnalyzer
from src.utils.helpers import setup_logging

def generate_sample_data(months: int = 12) -> Dict[str, float]:
    """Generate sample energy consumption data."""
    current_date = datetime.now()
    data = {}

    # Generate realistic-looking consumption data with seasonal variation
    for i in range(months):
        date = current_date - timedelta(days=30*i)
        month_str = date.strftime("%Y-%m")

        # Base consumption (higher in winter/summer months)
        month_num = date.month
        if month_num in [12, 1, 2]:  # Winter
            base = random.uniform(900, 1200)
        elif month_num in [6, 7, 8]:  # Summer
            base = random.uniform(800, 1000)
        else:  # Spring/Fall
            base = random.uniform(500, 700)

        # Add some random variation
        consumption = round(base + random.uniform(-50, 50), 2)
        data[month_str] = consumption

    return dict(sorted(data.items()))

def read_data_file(file_path: str) -> Dict[str, float]:
    """Read energy consumption data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error reading data file: {str(e)}")
        raise

def save_results(results: Dict, output_file: str):
    """Save analysis results to a JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Energy Consumption Analysis Tool')
    parser.add_argument('--input', '-i', help='Input JSON file with consumption data')
    parser.add_argument('--output', '-o', help='Output file for analysis results')
    parser.add_argument('--sample', '-s', action='store_true', help='Use sample data')
    parser.add_argument('--months', '-m', type=int, default=12, help='Number of months for sample data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    try:
        # Initialize analyzer
        analyzer = EnergyAnalyzer()

        # Get input data
        if args.sample:
            logging.info(f"Generating sample data for {args.months} months")
            data = generate_sample_data(args.months)
        elif args.input:
            logging.info(f"Reading data from {args.input}")
            data = read_data_file(args.input)
        else:
            parser.error("Either --input or --sample is required")

        # Run analysis
        logging.info("Starting analysis")
        results = analyzer.analyze(data)

        # Save or display results
        if args.output:
            save_results(results, args.output)
        else:
            print(json.dumps(results, indent=2))

        logging.info("Analysis completed successfully")

    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
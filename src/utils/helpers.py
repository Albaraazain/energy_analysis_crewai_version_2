# src/utils/helpers.py

import logging
import sys
from typing import Dict, Any
import json

def setup_logging(verbose: bool = False):
    """Configure logging settings."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def format_consumption_data(data: Dict[str, float]) -> str:
    """Format consumption data for display."""
    sorted_data = dict(sorted(data.items()))
    max_month_len = max(len(month) for month in sorted_data.keys())
    max_value_len = max(len(f"{value:.2f}") for value in sorted_data.values())

    output = ["Monthly Consumption Data:"]
    output.append("-" * (max_month_len + max_value_len + 10))

    for month, value in sorted_data.items():
        output.append(f"{month:<{max_month_len}} | {value:>{max_value_len}.2f} kWh")

    return "\n".join(output)

def format_results(results: Dict[str, Any]) -> str:
    """Format analysis results for display."""
    output = ["Energy Analysis Results", "=" * 50]

    # Format analysis section
    if "analysis" in results:
        output.append("\nAnalysis:")
        output.append("-" * 20)
        analysis = results["analysis"]

        if "basic_stats" in analysis:
            output.append("\nBasic Statistics:")
            for key, value in analysis["basic_stats"].items():
                output.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")

        if "patterns" in analysis:
            output.append("\nIdentified Patterns:")
            for key, value in analysis["patterns"].items():
                output.append(f"  {key.replace('_', ' ').title()}: {value}")

    # Format recommendations section
    if "recommendations" in results:
        output.append("\nRecommendations:")
        output.append("-" * 20)
        recommendations = results["recommendations"]

        for category, items in recommendations.items():
            output.append(f"\n{category.replace('_', ' ').title()}:")
            for item in items:
                output.append(f"  • {item}")

    return "\n".join(output)

def validate_json_file(file_path: str) -> bool:
    """Validate JSON file format."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return False

        for month, value in data.items():
            if not isinstance(month, str) or not isinstance(value, (int, float)):
                return False

        return True
    except Exception:
        return False
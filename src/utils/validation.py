# src/utils/validation.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataValidator:
    """Utilities for data validation"""

    @staticmethod
    def validate_energy_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Validate energy consumption data"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check required columns
        required_columns = ['timestamp', 'consumption']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results['valid'] = False
            validation_results['errors'].append(
                f"Missing required columns: {missing_columns}"
            )

        # Check data types
        if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            validation_results['valid'] = False
            validation_results['errors'].append(
                "Timestamp column must be datetime type"
            )

        if 'consumption' in data.columns and not pd.api.types.is_numeric_dtype(data['consumption']):
            validation_results['valid'] = False
            validation_results['errors'].append(
                "Consumption column must be numeric type"
            )

        # Check for negative values
        if 'consumption' in data.columns and (data['consumption'] < 0).any():
            validation_results['valid'] = False
            validation_results['errors'].append(
                "Negative consumption values found"
            )

        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            validation_results['warnings'].append(
                f"Missing values found: {missing_values.to_dict()}"
            )

        # Check for duplicate timestamps
        if 'timestamp' in data.columns and data['timestamp'].duplicated().any():
            validation_results['warnings'].append(
                "Duplicate timestamps found"
            )

        return validation_results

    @staticmethod
    def validate_analysis_results(results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis results"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        required_fields = ['metrics', 'patterns', 'insights']
        missing_fields = [field for field in required_fields if field not in results]

        if missing_fields:
            validation_results['valid'] = False
            validation_results['errors'].append(
                f"Missing required fields: {missing_fields}"
            )

        if 'metrics' in results and not isinstance(results['metrics'], dict):
            validation_results['valid'] = False
            validation_results['errors'].append(
                "Metrics must be a dictionary"
            )

        return validation_results
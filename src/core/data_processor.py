# src/core/data_processor.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
from .types import EnergyData

class DataProcessor:
    """Process and transform energy consumption data"""

    @staticmethod
    def normalize_data(data: Dict[str, float]) -> List[EnergyData]:
        """Convert raw data to standardized format"""
        normalized = []
        for date_str, value in data.items():
            normalized.append({
                'timestamp': datetime.strptime(date_str, "%Y-%m"),
                'consumption': float(value),
                'memory_metadata': {}
            })
        return normalized

    @staticmethod
    def to_dataframe(data: List[EnergyData]) -> pd.DataFrame:
        """Convert data to pandas DataFrame"""
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features to DataFrame"""
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['season'] = df.index.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        return df
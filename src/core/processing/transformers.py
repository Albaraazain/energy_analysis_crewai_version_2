# src/core/processing/transformers.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .base import DataProcessor, ProcessingResult

class TimeSeriesTransformer(DataProcessor):
    """Handles time series data transformations"""

    def process(self, data: Dict[str, Any]) -> ProcessingResult:
        """Process time series data"""
        try:
            # Validate input data
            validation_errors = self._validate_data_structure(data)
            if validation_errors:
                return ProcessingResult(
                    timestamp=datetime.now(),
                    data={},
                    metadata={'errors': validation_errors},
                    validation_status=False,
                    errors=validation_errors
                )

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Apply transformations
            processed_df = self._apply_transformations(df)

            # Generate memory_metadata
            metadata = self._generate_metadata(processed_df)

            return ProcessingResult(
                timestamp=datetime.now(),
                data=processed_df.to_dict(orient='records'),
                metadata=metadata,
                validation_status=True
            )
        except Exception as e:
            return ProcessingResult(
                timestamp=datetime.now(),
                data={},
                metadata={'error': str(e)},
                validation_status=False,
                errors=[str(e)]
            )

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply time series transformations"""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Resample to regular intervals if needed
        if self.config.get('resample', False):
            interval = self.config.get('resample_interval', '1H')
            df = df.resample(interval).mean()

        # Fill missing values
        df = self._handle_missing_values(df)

        # Add time-based features
        df = self._add_time_features(df)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in time series"""
        method = self.config.get('missing_value_method', 'interpolate')

        if method == 'interpolate':
            df = df.interpolate(method='time')
        elif method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'mean':
            df = df.fillna(df.mean())

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to DataFrame"""
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['is_weekend'] = df.index.dayofweek.isin([5, 6])

        # Add season
        df['season'] = df.index.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })

        return df

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate time series data"""
        return len(self._validate_data_structure(data)) == 0

    def _generate_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate memory_metadata for processed data"""
        return {
            'start_date': df.index.min().isoformat(),
            'end_date': df.index.max().isoformat(),
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'time_span': (df.index.max() - df.index.min()).days,
            'frequency': self._detect_frequency(df)
        }

    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect time series frequency"""
        try:
            return pd.infer_freq(df.index)
        except ValueError:
            time_diff = df.index.to_series().diff().median()
            if time_diff < timedelta(hours=1):
                return 'sub-hourly'
            elif time_diff < timedelta(days=1):
                return 'hourly'
            elif time_diff < timedelta(weeks=1):
                return 'daily'
            elif time_diff < timedelta(days=31):
                return 'weekly'
            else:
                return 'monthly'
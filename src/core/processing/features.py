# src/core/processing/features.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from .base import DataProcessor, ProcessingResult

class FeatureEngineer(DataProcessor):
    """Advanced feature engineering for energy consumption data"""

    def process(self, data: Dict[str, Any]) -> ProcessingResult:
        """Process data and generate advanced features"""
        try:
            df = pd.DataFrame(data)

            # Basic statistical features
            df = self._add_statistical_features(df)

            # Rolling window features
            df = self._add_rolling_features(df)

            # Lag features
            df = self._add_lag_features(df)

            # Domain-specific features
            df = self._add_energy_specific_features(df)

            return ProcessingResult(
                timestamp=datetime.now(),
                data=df.to_dict(orient='records'),
                metadata=self._generate_feature_metadata(df),
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

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Calculate rolling statistics
        windows = [24, 168]  # 24 hours, 1 week

        for window in windows:
            rolling = df['consumption'].rolling(window=window)
            prefix = f'rolling_{window}h'

            df[f'{prefix}_mean'] = rolling.mean()
            df[f'{prefix}_std'] = rolling.std()
            df[f'{prefix}_min'] = rolling.min()
            df[f'{prefix}_max'] = rolling.max()
            df[f'{prefix}_range'] = df[f'{prefix}_max'] - df[f'{prefix}_min']
            df[f'{prefix}_zscore'] = (
                                             df['consumption'] - df[f'{prefix}_mean']
                                     ) / df[f'{prefix}_std'].replace(0, 1)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        # Define windows for different patterns
        windows = {
            'hourly': 24,
            'daily': 7,
            'weekly': 4
        }

        for name, window in windows.items():
            # Rolling mean and variation
            df[f'{name}_mean'] = df['consumption'].rolling(
                window=window, min_periods=1
            ).mean()

            df[f'{name}_var'] = df['consumption'].rolling(
                window=window, min_periods=1
            ).var()

            # Rolling quantiles
            for q in [0.25, 0.5, 0.75]:
                df[f'{name}_q{int(q*100)}'] = df['consumption'].rolling(
                    window=window, min_periods=1
                ).quantile(q)

            # Rolling trend
            df[f'{name}_trend'] = self._calculate_rolling_trend(
                df['consumption'], window
            )

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        # Define lag periods
        lag_periods = [1, 24, 168]  # 1 hour, 1 day, 1 week

        for lag in lag_periods:
            # Simple lag
            df[f'lag_{lag}'] = df['consumption'].shift(lag)

            # Difference with lag
            df[f'diff_{lag}'] = df['consumption'] - df[f'lag_{lag}']

            # Percent change
            df[f'pct_change_{lag}'] = df['consumption'].pct_change(lag)

            # Rate of change
            df[f'rate_change_{lag}'] = df[f'diff_{lag}'] / lag

        return df

    def _add_energy_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific energy features"""
        # Peak vs Off-peak consumption
        df['is_peak_hour'] = df.index.hour.isin(range(9, 20))  # 9 AM to 8 PM
        df['peak_consumption'] = np.where(
            df['is_peak_hour'],
            df['consumption'],
            0
        )
        df['offpeak_consumption'] = np.where(
            df['is_peak_hour'],
            0,
            df['consumption']
        )

        # Daily load factor
        df['daily_max'] = df.groupby(df.index.date)['consumption'].transform('max')
        df['daily_avg'] = df.groupby(df.index.date)['consumption'].transform('mean')
        df['load_factor'] = df['daily_avg'] / df['daily_max']

        # Consumption patterns
        df['baseload'] = df.groupby(df.index.date)['consumption'].transform('min')
        df['peak_load'] = df['consumption'] - df['baseload']

        # Variability metrics
        df['daily_volatility'] = df.groupby(df.index.date)['consumption'].transform(
            lambda x: x.std() / x.mean() if x.mean() != 0 else 0
        )

        return df

    def _calculate_rolling_trend(self, series: pd.Series,
                                 window: int) -> pd.Series:
        """Calculate rolling trend using linear regression"""
        def calculate_slope(x):
            if len(x) < 2:
                return 0
            try:
                slope, _, _, _, _ = stats.linregress(range(len(x)), x)
                return slope
            except:
                return 0

        return series.rolling(window=window, min_periods=2).apply(
            calculate_slope
        )

    def _generate_feature_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate memory_metadata about engineered features"""
        return {
            'total_features': len(df.columns),
            'feature_types': {
                'statistical': len([col for col in df.columns if 'rolling' in col]),
                'lag': len([col for col in df.columns if 'lag' in col]),
                'energy_specific': len([
                    col for col in df.columns
                    if any(x in col for x in ['peak', 'load', 'base'])
                ])
            },
            'missing_values': df.isnull().sum().to_dict(),
            'feature_correlations': self._calculate_correlations(df)
        }

    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations with target variable"""
        correlations = {}
        for column in df.columns:
            if column != 'consumption':
                try:
                    corr = df[column].corr(df['consumption'])
                    if not np.isnan(corr):
                        correlations[column] = corr
                except:
                    continue
        return correlations

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate input data for feature engineering"""
        if not isinstance(data, dict):
            return False

        required_fields = ['consumption']
        return all(field in data for field in required_fields)
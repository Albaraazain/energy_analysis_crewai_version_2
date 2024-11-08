# src/core/processing/preprocessing.py
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .base import DataProcessor, ProcessingResult

class DataPreprocessor(DataProcessor):
    """Advanced data preprocessing and normalization"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scalers = {}
        self.scaling_method = config.get('scaling_method', 'standard')
        self.scaling_params = {}

    def process(self, data: Dict[str, Any]) -> ProcessingResult:
        """Process and normalize data"""
        try:
            df = pd.DataFrame(data)

            # Handle missing values
            df = self._handle_missing_values(df)

            # Remove outliers
            df = self._handle_outliers(df)

            # Scale numerical features
            df = self._scale_features(df)

            # Encode categorical variables
            df = self._encode_categorical(df)

            return ProcessingResult(
                timestamp=datetime.now(),
                data=df.to_dict(orient='records'),
                metadata=self._generate_preprocessing_metadata(df),
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

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with advanced strategies"""
        strategies = {
            'numerical': {
                'method': self.config.get('numerical_imputation', 'interpolate'),
                'columns': df.select_dtypes(include=[np.number]).columns
            },
            'categorical': {
                'method': self.config.get('categorical_imputation', 'mode'),
                'columns': df.select_dtypes(include=['object', 'category']).columns
            }
        }

        for data_type, strategy in strategies.items():
            if strategy['columns'].empty:
                continue

            if strategy['method'] == 'interpolate':
                df[strategy['columns']] = df[strategy['columns']].interpolate(
                    method='time'
                )
            elif strategy['method'] == 'mean':
                df[strategy['columns']] = df[strategy['columns']].fillna(
                    df[strategy['columns']].mean()
                )
            elif strategy['method'] == 'mode':
                df[strategy['columns']] = df[strategy['columns']].fillna(
                    df[strategy['columns']].mode().iloc[0]
                )
            elif strategy['method'] == 'ffill':
                df[strategy['columns']] = df[strategy['columns']].fillna(
                    method='ffill'
                )

        return df
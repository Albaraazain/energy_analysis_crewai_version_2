# src/core/processing/quality.py
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from .base import DataProcessor, ProcessingResult

class DataQualityChecker(DataProcessor):
    """Advanced data quality assessment and validation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.quality_checks = {
            'completeness': self._check_completeness,
            'consistency': self._check_consistency,
            'validity': self._check_validity,
            'timeliness': self._check_timeliness,
            'accuracy': self._check_accuracy
        }
        self.quality_thresholds = config.get('quality_thresholds', {
            'completeness': 0.95,  # 95% data completeness required
            'consistency': 0.90,   # 90% consistency required
            'validity': 0.98,      # 98% valid values required
            'timeliness': 24,      # Maximum 24 hours delay
            'accuracy': 0.95       # 95% accuracy required
        })

    def process(self, data: Dict[str, Any]) -> ProcessingResult:
        """Perform comprehensive data quality assessment"""
        try:
            df = pd.DataFrame(data)

            # Run all quality checks
            quality_results = {}
            for check_name, check_func in self.quality_checks.items():
                quality_results[check_name] = check_func(df)

            # Generate quality score
            quality_score = self._calculate_quality_score(quality_results)

            # Generate quality report
            quality_report = self._generate_quality_report(
                df, quality_results, quality_score
            )

            return ProcessingResult(
                timestamp=datetime.now(),
                data=quality_report,
                metadata={
                    'quality_score': quality_score,
                    'checks_passed': self._get_passed_checks(quality_results)
                },
                validation_status=quality_score >= self.config.get(
                    'minimum_quality_score', 0.8
                )
            )
        except Exception as e:
            return ProcessingResult(
                timestamp=datetime.now(),
                data={},
                metadata={'error': str(e)},
                validation_status=False,
                errors=[str(e)]
            )

    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness"""
        completeness_results = {
            'overall': 1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'by_column': (1 - df.isnull().sum() / len(df)).to_dict(),
            'missing_patterns': self._analyze_missing_patterns(df)
        }

        return {
            'score': completeness_results['overall'],
            'details': completeness_results,
            'passed': completeness_results['overall'] >=
                      self.quality_thresholds['completeness']
        }

    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency"""
        consistency_checks = {
            'value_range': self._check_value_ranges(df),
            'logical_rules': self._check_logical_rules(df),
            'temporal_consistency': self._check_temporal_consistency(df)
        }

        consistency_score = np.mean([
            check['score'] for check in consistency_checks.values()
        ])

        return {
            'score': consistency_score,
            'details': consistency_checks,
            'passed': consistency_score >= self.quality_thresholds['consistency']
        }

    def _check_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity"""
        validity_results = {
            'format_checks': self._check_data_formats(df),
            'range_checks': self._check_value_validity(df),
            'relationship_checks': self._check_relationship_validity(df)
        }

        validity_score = np.mean([
            check['score'] for check in validity_results.values()
        ])

        return {
            'score': validity_score,
            'details': validity_results,
            'passed': validity_score >= self.quality_thresholds['validity']
        }

    def _check_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data timeliness"""
        if 'timestamp' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            return {
                'score': 0,
                'details': {'error': 'No timestamp information available'},
                'passed': False
            }

        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['timestamp'])
        current_time = pd.Timestamp.now()

        max_delay = (current_time - timestamps.max()).total_seconds() / 3600  # in hours

        timeliness_score = 1.0 if max_delay <= self.quality_thresholds['timeliness'] else \
            max(0, 1 - (max_delay - self.quality_thresholds['timeliness']) / 24)

        return {
            'score': timeliness_score,
            'details': {
                'max_delay_hours': max_delay,
                'data_coverage': {
                    'start': timestamps.min().isoformat(),
                    'end': timestamps.max().isoformat()
                }
            },
            'passed': timeliness_score >= 0.8
        }

    def _check_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data accuracy"""
        accuracy_checks = {
            'statistical_validity': self._check_statistical_validity(df),
            'outlier_assessment': self._check_outliers(df),
            'precision_assessment': self._check_precision(df)
        }

        accuracy_score = np.mean([
            check['score'] for check in accuracy_checks.values()
        ])

        return {
            'score': accuracy_score,
            'details': accuracy_checks,
            'passed': accuracy_score >= self.quality_thresholds['accuracy']
        }
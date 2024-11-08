# src/core/analytics/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    """Model for analysis results"""
    timestamp: datetime
    metrics: Dict[str, float]
    patterns: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

class BaseAnalyzer(ABC):
    """Abstract base class for analytics components"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.8)

    @abstractmethod
    async def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Perform analysis on the data"""
        pass

    def validate_results(self, result: AnalysisResult) -> bool:
        """Validate analysis results"""
        return (
                result.confidence >= self.confidence_threshold and
                bool(result.patterns) and
                bool(result.insights)
        )

    async def _calculate_confidence(self, metrics: Dict[str, float],
                                    sample_size: int) -> float:
        """Calculate confidence score for analysis"""
        try:
            # Base confidence on sample size and metric stability
            sample_factor = min(1.0, sample_size / 100)
            metric_stability = np.mean([
                self._calculate_metric_stability(values)
                for values in metrics.values()
                if isinstance(values, (list, np.ndarray))
            ])

            return float(sample_factor * metric_stability)
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _calculate_metric_stability(self, values: np.ndarray) -> float:
        """Calculate stability score for a metric"""
        try:
            # Use coefficient of variation as stability measure
            cv = stats.variation(values)
            return 1 / (1 + cv)  # Transform to 0-1 scale
        except Exception:
            return 0.0
# src/utils/metrics.py
from typing import Dict, Any, List
from datetime import datetime
import statistics

class PerformanceMetrics:
    """Track and analyze agent and system performance"""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'response_times': [],
            'token_usage': [],
            'memory_usage': [],
            'success_rate': []
        }

    def add_metric(self, metric_type: str, value: float):
        """Add a new metric measurement"""
        if metric_type in self.metrics:
            self.metrics[metric_type].append(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get statistical summary of metrics"""
        summary = {}
        for metric_type, values in self.metrics.items():
            if values:
                summary[metric_type] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
        return summary
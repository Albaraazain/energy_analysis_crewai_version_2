# src/core/analytics/statistical.py
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from .base import BaseAnalyzer, AnalysisResult

class StatisticalAnalyzer(BaseAnalyzer):
    """Advanced statistical analysis component"""

    async def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Perform comprehensive statistical analysis"""
        try:
            # Basic statistics
            basic_stats = self._calculate_basic_stats(data)

            # Advanced statistical analysis
            advanced_stats = {
                'distribution': self._analyze_distribution(data),
                'stationarity': self._test_stationarity(data),
                'seasonality': self._analyze_seasonality(data),
                'correlations': self._analyze_correlations(data)
            }

            # Pattern detection
            patterns = self._detect_patterns(data)

            # Generate insights
            insights = self._generate_insights(basic_stats, advanced_stats, patterns)

            # Calculate confidence
            confidence = await self._calculate_confidence(basic_stats, len(data))

            return AnalysisResult(
                timestamp=datetime.now(),
                metrics={**basic_stats, **advanced_stats},
                patterns=patterns,
                insights=insights,
                confidence=confidence,
                metadata={
                    'analysis_type': 'statistical',
                    'data_points': len(data),
                    'features_analyzed': list(data.columns)
                }
            )
        except Exception as e:
            print(f"Error in statistical analysis: {str(e)}")
            raise

    def _calculate_basic_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive basic statistics"""
        stats_dict = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            series = data[column]
            stats_dict[column] = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                'range': float(series.max() - series.min()),
                'cv': float(series.std() / series.mean() if series.mean() != 0 else 0)
            }
        return stats_dict

    def _analyze_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distribution characteristics"""
        distributions = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            series = data[column]
            # Normality test
            _, p_value = stats.normaltest(series)

            # Fit best distribution
            distributions[column] = {
                'is_normal': bool(p_value > 0.05),
                'best_fit': self._find_best_distribution(series),
                'percentiles': {
                    str(p): float(np.percentile(series, p))
                    for p in [5, 25, 50, 75, 95]
                }
            }
        return distributions

    def _find_best_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """Find the best-fitting distribution"""
        distributions = [
            stats.norm, stats.gamma, stats.lognorm,
            stats.expon, stats.weibull_min
        ]

        best_dist = None
        best_params = None
        best_sse = np.inf

        for distribution in distributions:
            try:
                # Fit distribution
                params = distribution.fit(data)

                # Calculate SSE
                theoretical_pdf = distribution.pdf(
                    np.sort(data),
                    *params[:-2],
                    loc=params[-2],
                    scale=params[-1]
                )
                sse = np.sum((np.histogram(data, bins='auto')[0] - theoretical_pdf)**2)

                if sse < best_sse:
                    best_dist = distribution.name
                    best_params = params
                    best_sse = sse
            except Exception:
                continue

        return {
            'distribution': best_dist,
            'parameters': list(best_params) if best_params is not None else None,
            'sse': float(best_sse)
        }

    def _test_stationarity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        from statsmodels.tsa.stattools import adfuller

        stationarity_results = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            series = data[column]
            try:
                result = adfuller(series)
                stationarity_results[column] = {
                    'is_stationary': bool(result[1] < 0.05),
                    'adf_statistic': float(result[0]),
                    'p_value': float(result[1]),
                    'critical_values': {
                        str(key): float(value)
                        for key, value in result[4].items()
                    }
                }
            except Exception as e:
                print(f"Error testing stationarity for {column}: {str(e)}")
                stationarity_results[column] = {
                    'error': str(e)
                }

        return stationarity_results

    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in the data"""
        seasonality_results = {}

        for column in data.select_dtypes(include=[np.number]).columns:
            try:
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    data[column],
                    period=self._estimate_period(data[column]),
                    extrapolate_trend='freq'
                )

                seasonality_results[column] = {
                    'seasonal_strength': float(
                        np.std(decomposition.seasonal) /
                        np.std(decomposition.resid)
                    ),
                    'trend_strength': float(
                        np.std(decomposition.trend) /
                        np.std(decomposition.resid)
                    ),
                    'seasonal_peaks': self._find_seasonal_peaks(
                        decomposition.seasonal
                    ),
                    'seasonal_troughs': self._find_seasonal_troughs(
                        decomposition.seasonal
                    )
                }
            except Exception as e:
                print(f"Error analyzing seasonality for {column}: {str(e)}")
                seasonality_results[column] = {
                    'error': str(e)
                }

        return seasonality_results
# src/core/analytics/patterns.py
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from .base import BaseAnalyzer, AnalysisResult

class PatternAnalyzer(BaseAnalyzer):
    """Advanced pattern recognition component"""

    async def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Perform comprehensive pattern analysis"""
        try:
            # Identify patterns
            time_patterns = self._analyze_time_patterns(data)
            consumption_patterns = self._analyze_consumption_patterns(data)
            behavioral_patterns = self._analyze_behavioral_patterns(data)
            anomaly_patterns = self._detect_anomalies(data)

            # Combine patterns
            patterns = self._combine_patterns(
                time_patterns,
                consumption_patterns,
                behavioral_patterns,
                anomaly_patterns
            )

            # Generate insights
            insights = self._generate_pattern_insights(patterns)

            # Calculate metrics
            metrics = self._calculate_pattern_metrics(patterns)

            # Calculate confidence
            confidence = await self._calculate_confidence(metrics, len(data))

            return AnalysisResult(
                timestamp=datetime.now(),
                metrics=metrics,
                patterns=patterns,
                insights=insights,
                confidence=confidence,
                metadata={
                    'analysis_type': 'pattern_recognition',
                    'pattern_types': list(patterns.keys())
                }
            )
        except Exception as e:
            print(f"Error in pattern analysis: {str(e)}")
            raise

    def _analyze_time_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time-based patterns"""
        return {
            'daily': self._analyze_daily_patterns(data),
            'weekly': self._analyze_weekly_patterns(data),
            'monthly': self._analyze_monthly_patterns(data),
            'seasonal': self._analyze_seasonal_patterns(data)
        }

    def _analyze_consumption_patterns(self,
                                      data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consumption patterns"""
        consumption = data['consumption'].values

        # Find peaks and troughs
        peaks, _ = signal.find_peaks(consumption)
        troughs, _ = signal.find_peaks(-consumption)

        # Identify cycles
        cycles = self._identify_cycles(consumption)

        # Cluster consumption levels
        clusters = self._cluster_consumption(consumption)

        return {
            'peaks': {
                'indices': peaks.tolist(),
                'values': consumption[peaks].tolist(),
                'frequency': len(peaks) / len(consumption)
            },
            'troughs': {
                'indices': troughs.tolist(),
                'values': consumption[troughs].tolist(),
                'frequency': len(troughs) / len(consumption)
            },
            'cycles': cycles,
            'clusters': clusters
        }

    def _analyze_behavioral_patterns(self,
                                     data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavioral patterns"""
        # Create time features
        data['hour'] = data.index.hour
        data['day'] = data.index.dayofweek
        data['month'] = data.index.month

        # Identify usage profiles
        profiles = self._identify_usage_profiles(data)

        # Detect routine patterns
        routines = self._detect_routines(data)

        # Analyze consistency
        consistency = self._analyze_usage_consistency(data)

        return {
            'profiles': profiles,
            'routines': routines,
            'consistency': consistency
        }

    def _identify_cycles(self, data: np.ndarray) -> Dict[str, Any]:
        """Identify cyclical patterns in the data"""
        try:
            # Perform FFT
            fft_result = np.fft.fft(data)
            frequencies = np.fft.fftfreq(len(data))

            # Find dominant frequencies
            dominant_indices = np.argsort(np.abs(fft_result))[-5:]

            cycles = []
            for idx in dominant_indices:
                if frequencies[idx] > 0:  # Only positive frequencies
                    period = 1 / frequencies[idx]
                    amplitude = np.abs(fft_result[idx])
                    cycles.append({
                        'period': float(period),
                        'amplitude': float(amplitude),
                        'frequency': float(frequencies[idx])
                    })

            return {
                'dominant_cycles': cycles,
                'total_cycles': len(cycles)
            }
        except Exception as e:
            print(f"Error identifying cycles: {str(e)}")
            return {'error': str(e)}

    def _cluster_consumption(self, data: np.ndarray) -> Dict[str, Any]:
        """Cluster consumption levels"""
        try:
            # Prepare data for clustering
            X = StandardScaler().fit_transform(data.reshape(-1, 1))

            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=0.3, min_samples=5).fit(X)

            # Analyze clusters
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            clusters = []
            for i in range(n_clusters):
                cluster_data = data[labels == i]
                clusters.append({
                    'mean': float(cluster_data.mean()),
                    'std': float(cluster_data.std()),
                    'size': int(len(cluster_data)),
                    'min': float(cluster_data.min()),
                    'max': float(cluster_data.max())
                })

            return {
                'n_clusters': n_clusters,
                'clusters': clusters,
                'noise_points': int(sum(labels == -1))
            }
        except Exception as e:
            print(f"Error clustering consumption: {str(e)}")
            return {'error': str(e)}

    def _detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalous patterns"""
        anomalies = {
            'point_anomalies': self._detect_point_anomalies(data),
            'contextual_anomalies': self._detect_contextual_anomalies(data),
            'seasonal_anomalies': self._detect_seasonal_anomalies(data)
        }

        return {
            'anomalies': anomalies,
            'total_anomalies': sum(len(a) for a in anomalies.values())
        }

    def _detect_point_anomalies(self,
                                data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect point anomalies using statistical methods"""
        consumption = data['consumption']
        mean = consumption.mean()
        std = consumption.std()
        threshold = 3  # Number of standard deviations

        anomalies = []
        for idx, value in consumption.items():
            z_score = abs(value - mean) / std
            if z_score > threshold:
                anomalies.append({
                    'timestamp': idx.isoformat(),
                    'value': float(value),
                    'z_score': float(z_score),
                    'type': 'point'
                })

        return anomalies

    def _detect_contextual_anomalies(self,
                                     data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect contextual anomalies considering time context"""
        anomalies = []

        # Group by context (e.g., hour of day, day of week)
        for context, group in data.groupby([
            data.index.hour, data.index.dayofweek
        ]):
            hour, day = context
            consumption = group['consumption']
            mean = consumption.mean()
            std = consumption.std()

            for idx, value in consumption.items():
                z_score = abs(value - mean) / std
                if z_score > 2.5:  # Lower threshold for contextual anomalies
                    anomalies.append({
                        'timestamp': idx.isoformat(),
                        'value': float(value),
                        'z_score': float(z_score),
                        'context': {
                            'hour': hour,
                            'day': day
                        },
                        'type': 'contextual'
                    })

        return anomalies
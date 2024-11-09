from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np
from .base import BaseAgent, AgentResult
from .interfaces import IAnalysisAgent
from crewai import Agent

class DataAnalystAgent(BaseAgent, IAnalysisAgent):
    """Enhanced Data Analyst Agent with advanced analysis capabilities"""

    def create_agent(self) -> Agent:
        """Create and configure the Data Analyst agent"""
        return Agent(
            role='Energy Data Analyst',
            goal='Analyze energy consumption data and identify meaningful patterns',
            backstory="""You are an experienced energy data analyst with expertise in 
            residential energy consumption patterns. You excel at interpreting usage 
            data and identifying optimization opportunities.""",
            verbose=self.config.get('verbose', True),
            allow_delegation=True,
            llm=self.llm
        )

    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the provided energy consumption data"""
        try:
            df = pd.DataFrame(data['data'])

            analysis_results = {
                'statistics': self._calculate_statistics(df),
                'patterns': self._analyze_patterns(df),
                'anomalies': self._detect_anomalies(df)
            }

            return analysis_results
        except Exception as e:
            raise Exception(f"Error in data analysis: {str(e)}")

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate the analysis results"""
        required_keys = ['statistics', 'patterns', 'anomalies']

        # Check if all required keys are present
        if not all(key in results for key in required_keys):
            return False

        # Validate statistics
        statistics = results.get('statistics', {})
        if not isinstance(statistics, dict) or 'basic' not in statistics:
            return False

        # Validate patterns
        patterns = results.get('patterns', {})
        if not isinstance(patterns, dict) or 'seasonal' not in patterns:
            return False

        # Validate anomalies
        anomalies = results.get('anomalies', [])
        if not isinstance(anomalies, list):
            return False

        return True

    async def process(self, data: Dict[str, Any]) -> AgentResult:
        """Process energy consumption data with enhanced analysis"""
        try:
            # Perform data analysis
            analysis_results = await self.analyze_data(data)

            # Validate results
            if not self.validate_results(analysis_results):
                raise ValueError("Invalid analysis results")

            # Generate insights from the analysis
            insights = self._generate_insights(
                pd.DataFrame(data['data']),
                analysis_results['patterns'],
                analysis_results['anomalies']
            )

            return AgentResult(
                status='success',
                data={
                    'statistics': analysis_results['statistics'],
                    'patterns': analysis_results['patterns'],
                    'anomalies': analysis_results['anomalies'],
                    'insights': insights,
                    'metrics': {
                        'records_processed': len(pd.DataFrame(data['data'])),
                        'analysis_completion': 100
                    }
                },
                metadata={
                    'processing_time': datetime.now().isoformat(),
                    'data_quality': self._assess_data_quality(pd.DataFrame(data['data']))
                }
            )

        except Exception as e:
            return AgentResult(
                status='error',
                data={},
                metadata={'error_type': type(e).__name__},
                error=str(e)
            )

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate enhanced statistical measures"""
        consumption = df['consumption']
        return {
            'basic': {
                'mean': float(consumption.mean()),
                'median': float(consumption.median()),
                'std': float(consumption.std()),
                'min': float(consumption.min()),
                'max': float(consumption.max())
            },
            'distribution': {
                'quartiles': {k: float(v) for k, v in
                              consumption.quantile([0.25, 0.5, 0.75]).to_dict().items()},
                'skewness': float(consumption.skew()),
                'kurtosis': float(consumption.kurtosis())
            },
            'trends': {
                'rolling_mean': [float(x) for x in
                                 consumption.rolling(window=3).mean().fillna(0).tolist()],
                'cumulative_sum': [float(x) for x in consumption.cumsum().tolist()]
            }
        }

    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify consumption patterns with enhanced detection"""
        return {
            'seasonal': self._analyze_seasonal_patterns(df),
            'trends': self._analyze_trends(df),
            'cycles': self._analyze_cycles(df)
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies using multiple methods"""
        anomalies = []

        # Z-score based detection
        z_scores = (df['consumption'] - df['consumption'].mean()) / df['consumption'].std()
        anomaly_indices = abs(z_scores) > 2

        for idx in df[anomaly_indices].index:
            anomalies.append({
                'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'value': float(df.loc[idx, 'consumption']),
                'z_score': float(z_scores[idx]),
                'type': 'statistical',
                'severity': 'high' if abs(z_scores[idx]) > 3 else 'medium'
            })

        return anomalies

    def _generate_insights(self, df: pd.DataFrame, patterns: Dict[str, Any],
                           anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable insights from analysis"""
        insights = []

        # Trend-based insights
        if 'trends' in patterns:
            trend = patterns['trends']
            insights.append({
                'type': 'trend',
                'description': f"Consumption showing {trend['direction']} trend",
                'significance': trend.get('significance', 0.0),
                'action_items': self._generate_trend_recommendations(trend)
            })

        # Seasonal insights
        if 'seasonal' in patterns:
            seasonal = patterns['seasonal']
            insights.append({
                'type': 'seasonal',
                'description': f"Identified {len(seasonal.get('peaks', {}))} seasonal patterns",
                'patterns': seasonal.get('averages', {}),
                'action_items': self._generate_seasonal_recommendations(seasonal)
            })

        # Anomaly insights
        if anomalies:
            insights.append({
                'type': 'anomaly',
                'description': f"Detected {len(anomalies)} anomalies",
                'anomalies': anomalies,
                'action_items': self._generate_anomaly_recommendations(anomalies)
            })

        return insights

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of input data"""
        return {
            'completeness': {k: float(v) for k, v in
                             (1 - df.isnull().sum() / len(df)).to_dict().items()},
            'date_range': {
                'start': df.index.min().isoformat() if hasattr(df.index.min(), 'isoformat')
                else str(df.index.min()),
                'end': df.index.max().isoformat() if hasattr(df.index.max(), 'isoformat')
                else str(df.index.max()),
                'duration_days': float((df.index.max() - df.index.min()).total_seconds() / 86400)
                if hasattr(df.index.max(), 'total_seconds') else 0
            },
            'value_range': {
                'min': float(df['consumption'].min()),
                'max': float(df['consumption'].max())
            }
        }

    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in consumption"""
        try:
            # Add season column if datetime index exists
            if hasattr(df.index, 'month'):
                df['season'] = df.index.month.map({
                    12: 'winter', 1: 'winter', 2: 'winter',
                    3: 'spring', 4: 'spring', 5: 'spring',
                    6: 'summer', 7: 'summer', 8: 'summer',
                    9: 'fall', 10: 'fall', 11: 'fall'
                })
            else:
                df['season'] = 'unknown'

            seasonal_means = df.groupby('season')['consumption'].mean()

            # Convert to standard Python types for serialization
            averages = {k: float(v) for k, v in seasonal_means.to_dict().items()}

            max_season = seasonal_means.idxmax()
            min_season = seasonal_means.idxmin()

            return {
                'averages': averages,
                'peaks': {
                    'highest': {
                        'season': max_season,
                        'value': float(seasonal_means[max_season])
                    },
                    'lowest': {
                        'season': min_season,
                        'value': float(seasonal_means[min_season])
                    }
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'averages': {},
                'peaks': {'highest': {}, 'lowest': {}}
            }

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consumption trends"""
        try:
            x = np.arange(len(df))
            y = df['consumption'].values
            z = np.polyfit(x, y, 1)

            trend_direction = 'increasing' if z[0] > 0 else 'decreasing'
            trend_significance = abs(z[0]) / df['consumption'].mean()

            return {
                'direction': trend_direction,
                'slope': float(z[0]),
                'significance': float(trend_significance),
                'linear_fit': {
                    'slope': float(z[0]),
                    'intercept': float(z[1])
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'direction': 'unknown',
                'slope': 0.0,
                'significance': 0.0
            }

    def _analyze_cycles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cyclical patterns in consumption"""
        try:
            if hasattr(df.index, 'month'):
                monthly_means = df.groupby(df.index.month)['consumption'].mean()
                cycle_strength = float(monthly_means.std() / monthly_means.mean())
            else:
                monthly_means = df.groupby(0)['consumption'].mean()
                cycle_strength = 0.0

            return {
                'monthly_patterns': {k: float(v) for k, v in monthly_means.to_dict().items()},
                'cycle_strength': {
                    'monthly': cycle_strength
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'monthly_patterns': {},
                'cycle_strength': {'monthly': 0.0}
            }

    def _generate_trend_recommendations(self, trend: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trends"""
        recommendations = []
        if trend.get('direction') == 'increasing':
            recommendations.extend([
                "Review recent changes in consumption patterns",
                "Check for equipment efficiency issues",
                "Consider energy audit to identify causes"
            ])
        else:
            recommendations.extend([
                "Continue monitoring for sustained improvement",
                "Document successful energy-saving measures",
                "Share best practices across similar use cases"
            ])
        return recommendations

    def _generate_seasonal_recommendations(self, seasonal: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on seasonal patterns"""
        recommendations = []
        peaks = seasonal.get('peaks', {})
        if peaks:
            highest = peaks.get('highest', {}).get('season')
            if highest == 'summer':
                recommendations.extend([
                    "Optimize cooling system efficiency",
                    "Consider solar shading options",
                    "Review insulation effectiveness"
                ])
            elif highest == 'winter':
                recommendations.extend([
                    "Check heating system efficiency",
                    "Inspect insulation and weatherization",
                    "Consider programmable thermostat upgrades"
                ])
        return recommendations

    def _generate_anomaly_recommendations(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on anomalies"""
        recommendations = []
        high_severity = any(a.get('severity') == 'high' for a in anomalies)
        if high_severity:
            recommendations.extend([
                "Conduct immediate equipment inspection",
                "Review operational procedures",
                "Implement real-time monitoring alerts"
            ])
        else:
            recommendations.extend([
                "Schedule routine equipment maintenance",
                "Monitor patterns for recurrence",
                "Document anomaly patterns for future reference"
            ])
        return recommendations
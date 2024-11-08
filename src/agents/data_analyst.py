# src/agents/data_analyst.py
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
from .base import BaseAgent, AgentResult
from ..core.types import EnergyData
from crewai import Agent

class DataAnalystAgent(BaseAgent):
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

    async def process(self, data: Dict[str, Any]) -> AgentResult:
        """Process energy consumption data with enhanced analysis"""
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(data['data'])

            # Perform basic statistical analysis
            basic_stats = self._calculate_statistics(df)

            # Identify patterns and trends
            patterns = self._analyze_patterns(df)

            # Identify anomalies
            anomalies = self._detect_anomalies(df)

            # Generate insights
            insights = self._generate_insights(df, patterns, anomalies)

            return AgentResult(
                status='success',
                data={
                    'statistics': basic_stats,
                    'patterns': patterns,
                    'anomalies': anomalies,
                    'insights': insights,
                    'metrics': {
                        'records_processed': len(df),
                        'analysis_completion': 100
                    }
                },
                metadata={
                    'processing_time': datetime.now().isoformat(),
                    'data_quality': self._assess_data_quality(df)
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
                'mean': consumption.mean(),
                'median': consumption.median(),
                'std': consumption.std(),
                'min': consumption.min(),
                'max': consumption.max()
            },
            'distribution': {
                'quartiles': consumption.quantile([0.25, 0.5, 0.75]).to_dict(),
                'skewness': consumption.skew(),
                'kurtosis': consumption.kurtosis()
            },
            'trends': {
                'rolling_mean': consumption.rolling(window=3).mean().tolist(),
                'cumulative_sum': consumption.cumsum().tolist()
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
                'timestamp': idx.isoformat(),
                'value': df.loc[idx, 'consumption'],
                'z_score': z_scores[idx],
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
                'significance': trend['significance'],
                'action_items': self._generate_trend_recommendations(trend)
            })

        # Seasonal insights
        if 'seasonal' in patterns:
            seasonal = patterns['seasonal']
            insights.append({
                'type': 'seasonal',
                'description': f"Identified {len(seasonal['patterns'])} seasonal patterns",
                'patterns': seasonal['patterns'],
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
            'completeness': (1 - df.isnull().sum() / len(df)).to_dict(),
            'date_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat(),
                'duration_days': (df.index.max() - df.index.min()).days
            },
            'value_range': {
                'min': df['consumption'].min(),
                'max': df['consumption'].max()
            }
        }

    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in consumption"""
        seasonal_patterns = {}

        # Group by season
        df['season'] = df.index.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })

        seasonal_means = df.groupby('season')['consumption'].mean()
        seasonal_patterns['averages'] = seasonal_means.to_dict()

        # Identify peak seasons
        max_season = seasonal_means.idxmax()
        min_season = seasonal_means.idxmin()

        seasonal_patterns['peaks'] = {
            'highest': {
                'season': max_season,
                'value': seasonal_means[max_season]
            },
            'lowest': {
                'season': min_season,
                'value': seasonal_means[min_season]
            }
        }

        return seasonal_patterns

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consumption trends"""
        # Simple linear trend
        import numpy as np
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

    def _analyze_cycles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cyclical patterns in consumption"""
        # Monthly patterns
        monthly_means = df.groupby(df.index.month)['consumption'].mean()

        return {
            'monthly_patterns': monthly_means.to_dict(),
            'cycle_strength': {
                'monthly': float(monthly_means.std() / monthly_means.mean())
            }
        }
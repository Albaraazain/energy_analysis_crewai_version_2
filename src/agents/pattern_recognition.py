# src/agents/pattern_recognition.py
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from crewai import Agent
from scipy import stats
from .base import BaseAgent, AgentResult
from .interfaces import IAnalysisAgent
from ..core.types import EnergyData

class PatternRecognitionAgent(BaseAgent, IAnalysisAgent):
    """Specialized agent for identifying complex patterns in energy consumption"""

    def create_agent(self) -> Agent:
        return Agent(
            role='Pattern Recognition Specialist',
            goal='Identify complex patterns and trends in energy consumption data',
            backstory="""You are a specialized pattern recognition expert with deep 
            expertise in time series analysis and energy consumption behavior. Your 
            focus is on identifying subtle patterns that might be missed by basic 
            statistical analysis.""",
            verbose=self.config.get('verbose', True),
            allow_delegation=True,
            llm=self.llm
        )

    async def process(self, data: Dict[str, Any]) -> AgentResult:
        """Process data to identify patterns"""
        try:
            df = pd.DataFrame(data['data'])

            # Comprehensive pattern analysis
            patterns = {
                'time_based': self._analyze_time_patterns(df),
                'behavioral': self._analyze_behavioral_patterns(df),
                'cyclical': self._analyze_cyclical_patterns(df),
                'correlations': self._analyze_correlations(df)
            }

            # Pattern significance assessment
            significant_patterns = self._assess_pattern_significance(patterns)

            return AgentResult(
                status='success',
                data={
                    'patterns': patterns,
                    'significant_patterns': significant_patterns,
                    'metrics': self._calculate_pattern_metrics(patterns)
                },
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'analysis_quality': self._assess_analysis_quality(patterns)
                }
            )

        except Exception as e:
            return AgentResult(
                status='error',
                data={},
                metadata={'error_type': type(e).__name__},
                error=str(e)
            )

    def _analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time-based patterns"""
        return {
            'daily': self._analyze_daily_patterns(df),
            'weekly': self._analyze_weekly_patterns(df),
            'monthly': self._analyze_monthly_patterns(df),
            'seasonal': self._analyze_seasonal_patterns(df)
        }

    def _analyze_behavioral_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        return {
            'usage_profiles': self._identify_usage_profiles(df),
            'peak_usage': self._analyze_peak_usage(df),
            'consistency': self._analyze_usage_consistency(df)
        }

    def _analyze_cyclical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cyclical patterns using advanced techniques"""
        # Perform seasonal decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose

        try:
            decomposition = seasonal_decompose(
                df['consumption'],
                period=12,  # Monthly data
                extrapolate_trend='freq'
            )

            return {
                'seasonal': {
                    'values': decomposition.seasonal.tolist(),
                    'strength': float(np.std(decomposition.seasonal) /
                                      np.std(df['consumption']))
                },
                'trend': {
                    'values': decomposition.trend.tolist(),
                    'direction': 'increasing' if decomposition.trend[-1] >
                                                 decomposition.trend[0] else 'decreasing'
                },
                'residual': {
                    'values': decomposition.resid.tolist(),
                    'volatility': float(np.std(decomposition.resid))
                }
            }
        except Exception as e:
            return {
                'error': str(e),
                'fallback': self._simple_cycle_analysis(df)
            }

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different factors"""
        correlations = {}

        # Temperature correlation (if available)
        if 'temperature' in df.columns:
            correlations['temperature'] = {
                'correlation': float(df['consumption'].corr(df['temperature'])),
                'significance': self._calculate_correlation_significance(
                    df['consumption'], df['temperature']
                )
            }

        # Time-based correlations
        correlations['time_of_day'] = self._analyze_time_correlation(df)
        correlations['day_of_week'] = self._analyze_dow_correlation(df)

        return correlations

    def _calculate_correlation_significance(self, x: pd.Series,
                                            y: pd.Series) -> Dict[str, float]:
        """Calculate the significance of a correlation"""
        correlation = x.corr(y)
        n = len(x)
        t_stat = correlation * np.sqrt((n-2)/(1-correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))

        return {
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
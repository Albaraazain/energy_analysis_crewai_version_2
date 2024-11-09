from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from .base import BaseAgent, AgentResult
from .interfaces import IAnalysisAgent
from crewai import Agent

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

    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the provided data for patterns"""
        try:
            df = pd.DataFrame(data['data'])

            patterns = {
                'time_based': self._analyze_time_patterns(df),
                'behavioral': self._analyze_behavioral_patterns(df),
                'cyclical': self._analyze_cyclical_patterns(df),
                'correlations': self._analyze_correlations(df)
            }

            # Assess pattern significance
            significant_patterns = self._assess_pattern_significance(patterns)

            return {
                'patterns': patterns,
                'significant_patterns': significant_patterns,
                'metrics': self._calculate_pattern_metrics(patterns)
            }
        except Exception as e:
            raise Exception(f"Error in pattern analysis: {str(e)}")

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate analysis results"""
        try:
            # Check for required top-level keys
            required_keys = ['patterns', 'significant_patterns', 'metrics']
            if not all(key in results for key in required_keys):
                return False

            # Validate patterns structure
            patterns = results.get('patterns', {})
            required_pattern_types = ['time_based', 'behavioral', 'cyclical', 'correlations']
            if not all(key in patterns for key in required_pattern_types):
                return False

            # Validate significant patterns
            significant_patterns = results.get('significant_patterns', [])
            if not isinstance(significant_patterns, (list, dict)):
                return False

            # Validate metrics
            metrics = results.get('metrics', {})
            if not isinstance(metrics, dict):
                return False

            return True
        except Exception:
            return False

    async def process(self, data: Dict[str, Any]) -> AgentResult:
        """Process data to identify patterns"""
        try:
            # Perform pattern analysis
            analysis_results = await self.analyze_data(data)

            # Validate results
            if not self.validate_results(analysis_results):
                raise ValueError("Invalid analysis results")

            return AgentResult(
                status='success',
                data=analysis_results,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'analysis_quality': self._assess_analysis_quality(analysis_results)
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
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    raise ValueError("No timestamp information available")

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                df['consumption'],
                period=24,  # Daily seasonality
                extrapolate_trend='freq'
            )

            return {
                'seasonal': {
                    'values': [float(x) for x in decomposition.seasonal.tolist()],
                    'strength': float(np.std(decomposition.seasonal) /
                                      np.std(df['consumption']))
                },
                'trend': {
                    'values': [float(x) for x in decomposition.trend.tolist()],
                    'direction': 'increasing' if decomposition.trend[-1] >
                                                 decomposition.trend[0] else 'decreasing'
                },
                'residual': {
                    'values': [float(x) for x in decomposition.resid.fillna(0).tolist()],
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

    def _analyze_daily_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze daily consumption patterns"""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    return {'error': 'No timestamp information available'}

            hourly_avg = df.groupby(df.index.hour)['consumption'].mean()

            # Find peak and off-peak hours
            peak_hour = hourly_avg.idxmax()
            off_peak_hour = hourly_avg.idxmin()

            return {
                'hourly_averages': {str(k): float(v) for k, v in hourly_avg.items()},
                'peak_hour': {
                    'hour': int(peak_hour),
                    'consumption': float(hourly_avg[peak_hour])
                },
                'off_peak_hour': {
                    'hour': int(off_peak_hour),
                    'consumption': float(hourly_avg[off_peak_hour])
                },
                'variability': float(hourly_avg.std() / hourly_avg.mean())
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly consumption patterns"""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    return {'error': 'No timestamp information available'}

            daily_avg = df.groupby(df.index.dayofweek)['consumption'].mean()

            # Map day numbers to names
            day_names = {
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            }

            return {
                'daily_averages': {day_names[k]: float(v) for k, v in daily_avg.items()},
                'weekday_avg': float(daily_avg[daily_avg.index < 5].mean()),
                'weekend_avg': float(daily_avg[daily_avg.index >= 5].mean()),
                'week_pattern': 'weekend_heavy' if daily_avg[daily_avg.index >= 5].mean() >
                                                   daily_avg[daily_avg.index < 5].mean()
                else 'weekday_heavy'
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_monthly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly consumption patterns"""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    return {'error': 'No timestamp information available'}

            monthly_avg = df.groupby(df.index.month)['consumption'].mean()

            # Find peak and low consumption months
            peak_month = monthly_avg.idxmax()
            low_month = monthly_avg.idxmin()

            return {
                'monthly_averages': {str(k): float(v) for k, v in monthly_avg.items()},
                'peak_month': {
                    'month': int(peak_month),
                    'consumption': float(monthly_avg[peak_month])
                },
                'low_month': {
                    'month': int(low_month),
                    'consumption': float(monthly_avg[low_month])
                },
                'seasonality_strength': float(monthly_avg.std() / monthly_avg.mean())
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal consumption patterns"""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    return {'error': 'No timestamp information available'}

            # Define seasons
            df['season'] = df.index.month.map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            })

            seasonal_avg = df.groupby('season')['consumption'].mean()

            return {
                'seasonal_averages': {k: float(v) for k, v in seasonal_avg.items()},
                'highest_season': {
                    'season': str(seasonal_avg.idxmax()),
                    'consumption': float(seasonal_avg.max())
                },
                'lowest_season': {
                    'season': str(seasonal_avg.idxmin()),
                    'consumption': float(seasonal_avg.min())
                },
                'seasonal_variation': float(seasonal_avg.std() / seasonal_avg.mean())
            }
        except Exception as e:
            return {'error': str(e)}

    def _identify_usage_profiles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify distinct usage profiles"""
        try:
            hourly_patterns = df.groupby(df.index.hour)['consumption'].mean()

            # Define profile characteristics
            morning_consumption = hourly_patterns[6:9].mean()
            midday_consumption = hourly_patterns[11:14].mean()
            evening_consumption = hourly_patterns[17:20].mean()
            night_consumption = hourly_patterns[[22, 23, 0, 1, 2, 3]].mean()

            profiles = []
            if morning_consumption > midday_consumption:
                profiles.append('morning_heavy')
            if evening_consumption > midday_consumption:
                profiles.append('evening_heavy')
            if night_consumption > 0.7 * hourly_patterns.mean():
                profiles.append('night_active')

            return {
                'identified_profiles': profiles,
                'profile_metrics': {
                    'morning_intensity': float(morning_consumption / hourly_patterns.mean()),
                    'midday_intensity': float(midday_consumption / hourly_patterns.mean()),
                    'evening_intensity': float(evening_consumption / hourly_patterns.mean()),
                    'night_intensity': float(night_consumption / hourly_patterns.mean())
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_peak_usage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze peak usage patterns"""
        try:
            daily_max = df.groupby(df.index.date)['consumption'].max()
            daily_avg = df.groupby(df.index.date)['consumption'].mean()
            peak_ratio = daily_max / daily_avg

            return {
                'average_peak_ratio': float(peak_ratio.mean()),
                'peak_variability': float(peak_ratio.std()),
                'peak_trend': 'increasing' if np.polyfit(range(len(peak_ratio)),
                                                         peak_ratio.values, 1)[0] > 0
                else 'decreasing'
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_usage_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consistency in usage patterns"""
        try:
            hourly_patterns = df.groupby([df.index.date,
                                          df.index.hour])['consumption'].mean().unstack()

            # Calculate pattern consistency scores
            pattern_correlation = hourly_patterns.corr().values
            consistency_score = float(np.mean(pattern_correlation))

            return {
                'consistency_score': consistency_score,
                'pattern_stability': 'high' if consistency_score > 0.8 else
                'medium' if consistency_score > 0.6 else 'low',
                'variation_coefficient': float(hourly_patterns.std().mean() /
                                               hourly_patterns.mean().mean())
            }
        except Exception as e:
            return {'error': str(e)}

    def _simple_cycle_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform simple cycle analysis when decomposition fails"""
        try:
            daily_avg = df.groupby(df.index.date)['consumption'].mean()
            return {
                'daily_cycle': {
                    'mean': float(daily_avg.mean()),
                    'std': float(daily_avg.std()),
                    'trend': 'increasing' if np.polyfit(range(len(daily_avg)),
                                                        daily_avg.values, 1)[0] > 0
                    else 'decreasing'
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_time_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation with time of day"""
        try:
            hour_correlation = stats.pointbiserialr(
                df.index.hour,
                df['consumption']
            )
            return {
                'correlation': float(hour_correlation[0]),
                'p_value': float(hour_correlation[1])
            }
        except Exception:
            return {'correlation': 0.0, 'p_value': 1.0}

    def _analyze_dow_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation with day of week"""
        try:
            dow_correlation = stats.pointbiserialr(
                df.index.dayofweek,
                df['consumption']
            )
            return {
                'correlation': float(dow_correlation[0]),
                'p_value': float(dow_correlation[1])
            }
        except Exception:
            return {'correlation': 0.0, 'p_value': 1.0}

    def _assess_pattern_significance(self, patterns: Dict[str, Any]) -> List[str]:
        """Assess and identify significant patterns"""
        significant_patterns = []

        # Check time-based patterns
        time_patterns = patterns.get('time_based', {})
        for period, data in time_patterns.items():
            if isinstance(data, dict) and data.get('variability', 0) > 0.2:
                significant_patterns.append(f"Significant {period} pattern detected")

        # Check behavioral patterns
        behavioral = patterns.get('behavioral', {})
        usage_profiles = behavioral.get('usage_profiles', {}).get('identified_profiles', [])
        for profile in usage_profiles:
            significant_patterns.append(f"Distinct {profile} usage profile identified")

        return significant_patterns

    def _calculate_pattern_metrics(self, patterns: Dict[str, Any]) -> Dict[str, float]:
        """Calculate pattern strength metrics"""
        metrics = {}

        # Calculate average pattern strengths
        if 'cyclical' in patterns:
            cyclical = patterns['cyclical']
            if 'seasonal' in cyclical:
                metrics['seasonal_strength'] = float(cyclical['seasonal'].get('strength', 0))

        # Add other relevant metrics
        metrics['pattern_count'] = len(self._assess_pattern_significance(patterns))

        return metrics

    def _assess_analysis_quality(self, patterns: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of pattern analysis"""
        return {
            'completeness': 1.0,  # Placeholder
            'confidence': 0.8,    # Placeholder
            'reliability': 0.9    # Placeholder
        }
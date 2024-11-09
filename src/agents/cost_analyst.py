from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from crewai import Agent
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .base import BaseAgent, AgentResult
from .interfaces import IAnalysisAgent, IRecommendationAgent

class CostAnalystAgent(BaseAgent, IAnalysisAgent, IRecommendationAgent):
    """Specialized agent for analyzing energy costs and providing financial insights"""

    def create_agent(self) -> Agent:
        return Agent(
            role='Energy Cost Analyst',
            goal='Analyze energy costs and provide financial optimization recommendations',
            backstory="""You are an expert in energy cost analysis and financial 
            optimization. Your expertise lies in identifying cost-saving opportunities 
            and optimizing rate plans based on consumption patterns.""",
            verbose=self.config.get('verbose', True),
            allow_delegation=True,
            llm=self.llm
        )

    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the provided data for cost insights"""
        try:
            # Create DataFrame and setup datetime index
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Verify the data is properly formatted
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Failed to create proper datetime index")

            analysis_results = {
                'rate_analysis': self._analyze_rate_plans(df),
                'cost_patterns': self._analyze_cost_patterns(df),
                'savings_opportunities': self._identify_savings(df),
                'budget_projections': self._generate_projections(df)
            }

            return analysis_results
        except Exception as e:
            raise Exception(f"Error in cost analysis: {str(e)}")

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate the analysis results"""
        required_keys = ['rate_analysis', 'cost_patterns', 'savings_opportunities', 'budget_projections']

        # Check if all required keys are present
        if not all(key in results for key in required_keys):
            return False

        # Validate rate analysis results
        rate_analysis = results.get('rate_analysis', {})
        if not isinstance(rate_analysis, dict) or 'plan_comparisons' not in rate_analysis:
            return False

        # Validate cost patterns
        cost_patterns = results.get('cost_patterns', {})
        if not isinstance(cost_patterns, dict) or 'monthly_trends' not in cost_patterns:
            return False

        return True

    async def process(self, data: Dict[str, Any]) -> AgentResult:
        """Process consumption data for cost analysis"""
        try:
            # Perform data analysis
            analysis_results = await self.analyze_data(data)

            # Validate results
            if not self.validate_results(analysis_results):
                raise ValueError("Invalid analysis results")

            # Generate recommendations
            recommendations = await self.generate_recommendations(analysis_results)

            return AgentResult(
                status='success',
                data={
                    'analysis': analysis_results,
                    'recommendations': recommendations,
                    'metrics': self._calculate_cost_metrics(analysis_results)
                },
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

    def _analyze_rate_plans(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze different rate plans and their impact"""
        # Define common rate plans
        rate_plans = {
            'standard': {'base_rate': 0.12, 'peak_rate': 0.20},
            'time_of_use': {
                'off_peak': 0.08,
                'mid_peak': 0.12,
                'peak': 0.24
            },
            'tiered': {
                'tier1_limit': 1000,
                'tier1_rate': 0.10,
                'tier2_rate': 0.15
            }
        }

        results = {}
        for plan_name, plan_rates in rate_plans.items():
            results[plan_name] = self._calculate_plan_costs(df, plan_rates)

        return {
            'plan_comparisons': results,
            'optimal_plan': self._determine_optimal_plan(results),
            'potential_savings': self._calculate_potential_savings(results)
        }

    def _analyze_cost_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in energy costs"""
        return {
            'monthly_trends': self._analyze_monthly_costs(df),
            'peak_cost_periods': self._identify_peak_costs(df),
            'cost_drivers': self._identify_cost_drivers(df)
        }

    def _identify_savings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify potential savings opportunities"""
        return {
            'rate_optimization': self._analyze_rate_optimization(df),
            'usage_optimization': self._analyze_usage_optimization(df),
            'peak_reduction': self._analyze_peak_reduction(df)
        }

    def _generate_projections(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate cost projections"""
        try:
            # Prepare data for forecasting
            costs = df['consumption'] * df.get('rate', 0.12)

            # Fit model
            model = ExponentialSmoothing(
                costs,
                seasonal_periods=12,
                trend='add',
                seasonal='add'
            ).fit()

            # Generate forecasts
            forecast_periods = 12
            forecast = model.forecast(forecast_periods)

            return {
                'monthly_forecasts': forecast.to_dict(),
                'annual_projection': float(forecast.sum()),
                'confidence_intervals': self._calculate_forecast_intervals(model, forecast_periods)
            }
        except Exception:
            # Fallback to simple moving average if advanced forecasting fails
            return self._simple_projection(costs)

    def _calculate_plan_costs(self, df: pd.DataFrame, rates: Dict[str, float]) -> Dict[str, float]:
        """Calculate costs under different rate plans"""
        consumption = df['consumption'].fillna(0)

        if 'peak_rate' in rates:  # Standard plan
            base_cost = consumption * rates['base_rate']
            peak_hours = df.index.hour.isin(range(14, 19))
            peak_cost = consumption[peak_hours] * (rates['peak_rate'] - rates['base_rate'])
            total_cost = base_cost.sum() + peak_cost.sum()

        elif 'off_peak' in rates:  # Time of use plan
            peak_hours = df.index.hour.isin(range(14, 19))
            mid_peak_hours = df.index.hour.isin(range(11, 14)) | df.index.hour.isin(range(19, 21))

            peak_cost = consumption[peak_hours] * rates['peak']
            mid_peak_cost = consumption[mid_peak_hours] * rates['mid_peak']
            off_peak_cost = consumption[~(peak_hours | mid_peak_hours)] * rates['off_peak']

            total_cost = peak_cost.sum() + mid_peak_cost.sum() + off_peak_cost.sum()

        else:  # Tiered plan
            tier1_consumption = consumption.clip(upper=rates['tier1_limit'])
            tier2_consumption = consumption - tier1_consumption

            total_cost = (tier1_consumption * rates['tier1_rate']).sum() + \
                         (tier2_consumption * rates['tier2_rate']).sum()

        return {
            'total_cost': float(total_cost),
            'average_monthly_cost': float(total_cost / df.index.nunique()),
            'peak_cost_percentage': float(peak_cost.sum() / total_cost * 100) if 'peak_cost' in locals() else 0
        }

    async def generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost-saving recommendations"""
        recommendations = {
            'immediate_actions': self._generate_immediate_actions(analysis),
            'long_term_strategies': self._generate_long_term_strategies(analysis),
            'rate_plan_recommendations': self._generate_rate_recommendations(analysis)
        }

        return await self.prioritize_recommendations(recommendations)

    async def prioritize_recommendations(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize recommendations based on impact and feasibility"""
        prioritized = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }

        for category, items in recommendations.items():
            if isinstance(items, list):
                for item in items:
                    priority = self._assess_recommendation_priority(item)
                    prioritized[f'{priority}_priority'].append({
                        'category': category,
                        'recommendation': item,
                        'impact': self._calculate_recommendation_impact(item),
                        'feasibility': self._assess_recommendation_feasibility(item)
                    })

        return prioritized

    def _generate_immediate_actions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate list of immediate actions based on analysis"""
        return [
            "Switch to optimal rate plan",
            "Adjust peak usage patterns",
            "Implement basic energy-saving measures"
        ]

    def _generate_long_term_strategies(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate long-term strategic recommendations"""
        return [
            "Install energy monitoring system",
            "Upgrade to energy-efficient equipment",
            "Develop comprehensive energy management plan"
        ]

    def _generate_rate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate rate plan recommendations"""
        return [
            "Evaluate time-of-use rate options",
            "Consider demand response programs",
            "Review peak demand charges"
        ]

    def _calculate_recommendation_impact(self, recommendation: str) -> float:
        """Calculate the potential impact of a recommendation"""
        # Placeholder for impact calculation logic
        return 0.8

    def _assess_recommendation_feasibility(self, recommendation: str) -> float:
        """Assess the feasibility of implementing a recommendation"""
        # Placeholder for feasibility assessment logic
        return 0.7

    def _assess_recommendation_priority(self, recommendation: str) -> str:
        """Assess the priority of a recommendation"""
        # Placeholder for priority assessment logic
        return 'high'

    def _calculate_cost_metrics(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key cost metrics"""
        return {
            'average_cost': 0.0,
            'cost_variance': 0.0,
            'savings_potential': 0.0
        }

    def _assess_analysis_quality(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of the analysis"""
        return {
            'completeness': 1.0,
            'reliability': 0.9,
            'accuracy': 0.85
        }

    def _simple_projection(self, costs: pd.Series) -> Dict[str, Any]:
        """Generate simple cost projections using moving averages"""
        return {
            'monthly_forecasts': costs.rolling(window=3).mean().tail(12).to_dict(),
            'annual_projection': float(costs.mean() * 12),
            'confidence_intervals': None
        }

    def _calculate_forecast_intervals(self, model: ExponentialSmoothing, periods: int) -> Dict[str, List[float]]:
        """Calculate forecast confidence intervals"""
        # Placeholder for confidence interval calculation
        return {
            'lower_95': [0.0] * periods,
            'upper_95': [0.0] * periods
        }

    def _analyze_rate_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze rate optimization opportunities"""
        return {'potential_savings': 0.0}

    def _analyze_usage_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze usage optimization opportunities"""
        return {'potential_savings': 0.0}

    def _analyze_peak_reduction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze peak reduction opportunities"""
        return {'potential_savings': 0.0}

    def _analyze_monthly_costs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly cost trends"""
        return {'trend': 'stable'}

    def _identify_peak_costs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify peak cost periods"""
        return {'peak_periods': []}

    def _identify_cost_drivers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify main cost drivers"""
        return {'primary_drivers': []}

    def _determine_optimal_plan(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the optimal rate plan"""
        return {'optimal_plan': 'standard'}

    def _calculate_potential_savings(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate potential savings"""
        return {'annual_savings': 0.0}
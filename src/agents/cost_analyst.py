# src/agents/cost_analyst.py
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from crewai import Agent

from .base import BaseAgent, AgentResult
from .interfaces import IAnalysisAgent, IRecommendationAgent
from ..core.types import EnergyData

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

    async def process(self, data: Dict[str, Any]) -> AgentResult:
        """Process consumption data for cost analysis"""
        try:
            df = pd.DataFrame(data['data'])

            # Comprehensive cost analysis
            cost_analysis = {
                'rate_analysis': self._analyze_rate_plans(df),
                'cost_patterns': self._analyze_cost_patterns(df),
                'savings_opportunities': self._identify_savings(df),
                'budget_projections': self._generate_projections(df)
            }

            # Generate recommendations
            recommendations = await self.generate_recommendations(cost_analysis)

            return AgentResult(
                status='success',
                data={
                    'analysis': cost_analysis,
                    'recommendations': recommendations,
                    'metrics': self._calculate_cost_metrics(cost_analysis)
                },
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'analysis_quality': self._assess_analysis_quality(cost_analysis)
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

    async def generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost-saving recommendations"""
        recommendations = {
            'immediate_actions': self._generate_immediate_actions(analysis),
            'long_term_strategies': self._generate_long_term_strategies(analysis),
            'rate_plan_recommendations': self._generate_rate_recommendations(analysis)
        }

        return await self.prioritize_recommendations(recommendations)

    def _calculate_plan_costs(self, df: pd.DataFrame,
                              rates: Dict[str, float]) -> Dict[str, float]:
        """Calculate costs under different rate plans"""
        if 'peak_rate' in rates:  # Standard plan
            return self._calculate_standard_plan(df, rates)
        elif 'off_peak' in rates:  # Time of use plan
            return self._calculate_tou_plan(df, rates)
        else:  # Tiered plan
            return self._calculate_tiered_plan(df, rates)

    def _calculate_standard_plan(self, df: pd.DataFrame,
                                 rates: Dict[str, float]) -> Dict[str, float]:
        """Calculate costs for standard rate plan"""
        consumption = df['consumption']
        base_cost = consumption * rates['base_rate']

        # Assume peak hours are 2-6 PM on weekdays
        peak_mask = (df.index.hour.isin(range(14, 19))) & (df.index.dayofweek < 5)
        peak_cost = consumption[peak_mask] * (rates['peak_rate'] - rates['base_rate'])

        total_cost = base_cost.sum() + peak_cost.sum()

        return {
            'total_cost': float(total_cost),
            'average_monthly_cost': float(total_cost / df.index.nunique()),
            'peak_cost_percentage': float(peak_cost.sum() / total_cost * 100)
        }

    async def prioritize_recommendations(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize recommendations based on impact and feasibility"""
        prioritized = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }

        for category, items in recommendations.items():
            for item in items:
                priority = self._assess_recommendation_priority(item)
                prioritized[f'{priority}_priority'].append({
                    'category': category,
                    'recommendation': item,
                    'impact': self._calculate_recommendation_impact(item),
                    'feasibility': self._assess_recommendation_feasibility(item)
                })

        return prioritized

    def _generate_projections(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate cost projections"""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        try:
            # Prepare data for forecasting
            costs = df['consumption'] * df.get('rate', 0.12)  # Use default rate if not provided

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
        except Exception as e:
            # Fallback to simple moving average
            return self._simple_projection(costs)
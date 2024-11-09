import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from crewai import Agent, LLM
from src.agents.cost_analyst import CostAnalystAgent

def prepare_dataframe(data):
    """Helper function to prepare DataFrame with proper datetime index"""
    df = pd.DataFrame(data['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

@pytest.mark.asyncio
class TestCostAnalystAgent:

    async def test_initialization(self, mock_llm, base_config):
        """Test CostAnalystAgent initialization"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)
        assert agent.llm == mock_llm
        assert agent.config == base_config
        assert hasattr(agent, '_agent') and agent._agent is None

    async def test_create_agent(self, mock_llm, base_config):
        """Test agent creation with correct role and configuration"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)
        crew_agent = agent.create_agent()
        assert crew_agent.role == 'Energy Cost Analyst'
        assert isinstance(crew_agent.llm, LLM)
        assert crew_agent.verbose == base_config['verbose']

    async def test_analyze_data(self, mock_llm, base_config, sample_energy_data):
        """Test cost analysis functionality"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Prepare data with proper datetime index
        data_with_index = {'data': sample_energy_data['data']}
        analysis_results = await agent.analyze_data(data_with_index)

        # Verify analysis structure
        assert isinstance(analysis_results, dict)
        assert 'rate_analysis' in analysis_results
        assert 'cost_patterns' in analysis_results
        assert 'savings_opportunities' in analysis_results
        assert 'budget_projections' in analysis_results

        # Verify rate analysis
        rate_analysis = analysis_results['rate_analysis']
        assert 'plan_comparisons' in rate_analysis
        assert 'optimal_plan' in rate_analysis
        assert 'potential_savings' in rate_analysis

        # Verify cost patterns
        cost_patterns = analysis_results['cost_patterns']
        assert 'monthly_trends' in cost_patterns
        assert 'peak_cost_periods' in cost_patterns
        assert 'cost_drivers' in cost_patterns

    async def test_validate_results(self, mock_llm, base_config):
        """Test results validation"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Test with valid results
        valid_results = {
            'rate_analysis': {
                'plan_comparisons': {
                    'standard': {'total_cost': 1000.0},
                    'time_of_use': {'total_cost': 950.0}
                },
                'optimal_plan': {'name': 'time_of_use'},
                'potential_savings': {'annual': 50.0}
            },
            'cost_patterns': {
                'monthly_trends': {'trend': 'stable'},
                'peak_cost_periods': [],
                'cost_drivers': []
            },
            'savings_opportunities': {
                'rate_optimization': {'potential_savings': 50.0},
                'usage_optimization': {'potential_savings': 30.0},
                'peak_reduction': {'potential_savings': 20.0}
            },
            'budget_projections': {
                'monthly_forecasts': {},
                'annual_projection': 12000.0
            }
        }
        assert agent.validate_results(valid_results) is True

        # Test with missing required keys
        invalid_results = {
            'rate_analysis': {'plan_comparisons': {}},
            # Missing required sections
        }
        assert agent.validate_results(invalid_results) is False

    async def test_process(self, mock_llm, base_config, sample_energy_data):
        """Test complete processing pipeline"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Prepare data with proper datetime index
        data_with_index = {'data': sample_energy_data['data']}
        result = await agent.process(data_with_index)

        assert result.status == 'success', f"Process failed with error: {result.error}"
        assert result.error is None

        # Verify data structure
        assert 'analysis' in result.data
        assert 'recommendations' in result.data
        assert 'metrics' in result.data

        analysis = result.data['analysis']
        assert 'rate_analysis' in analysis
        assert 'cost_patterns' in analysis
        assert 'savings_opportunities' in analysis
        assert 'budget_projections' in analysis

    async def test_rate_plan_analysis(self, mock_llm, base_config, sample_dataframe):
        """Test rate plan analysis calculations"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Ensure DataFrame has datetime index
        df = sample_dataframe.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        rate_analysis = agent._analyze_rate_plans(df)

        assert 'plan_comparisons' in rate_analysis
        assert 'optimal_plan' in rate_analysis
        assert 'potential_savings' in rate_analysis

        plan_comparisons = rate_analysis['plan_comparisons']
        assert 'standard' in plan_comparisons
        assert 'time_of_use' in plan_comparisons
        assert 'tiered' in plan_comparisons

        for plan_name, comparison in plan_comparisons.items():
            assert 'total_cost' in comparison
            assert 'average_monthly_cost' in comparison
            assert isinstance(comparison['total_cost'], float)
            assert comparison['total_cost'] >= 0

    async def test_calculate_plan_costs(self, mock_llm, base_config, sample_dataframe):
        """Test cost calculation for different rate plans"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Ensure DataFrame has datetime index
        df = sample_dataframe.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        # Test standard plan
        standard_rates = {'base_rate': 0.12, 'peak_rate': 0.20}
        standard_costs = agent._calculate_plan_costs(df, standard_rates)
        assert isinstance(standard_costs, dict)
        assert all(key in standard_costs for key in ['total_cost', 'average_monthly_cost', 'peak_cost_percentage'])

        # Test time-of-use plan
        tou_rates = {'off_peak': 0.08, 'mid_peak': 0.12, 'peak': 0.24}
        tou_costs = agent._calculate_plan_costs(df, tou_rates)
        assert isinstance(tou_costs, dict)
        assert tou_costs['total_cost'] > 0

        # Test tiered plan
        tiered_rates = {'tier1_limit': 1000, 'tier1_rate': 0.10, 'tier2_rate': 0.15}
        tiered_costs = agent._calculate_plan_costs(df, tiered_rates)
        assert isinstance(tiered_costs, dict)
        assert tiered_costs['total_cost'] > 0

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, mock_llm, base_config):
        """Test recommendation generation"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        analysis = {
            'rate_analysis': {
                'optimal_plan': 'time_of_use',
                'potential_savings': {'annual': 500.0}
            },
            'cost_patterns': {
                'peak_cost_periods': [{'hour': 14, 'cost': 0.24}]
            }
        }

        recommendations = await agent.generate_recommendations(analysis)

        assert isinstance(recommendations, dict)
        for priority in ['high_priority', 'medium_priority', 'low_priority']:
            assert priority in recommendations
            assert isinstance(recommendations[priority], list)

            if recommendations[priority]:  # If there are recommendations
                for rec in recommendations[priority]:
                    assert 'category' in rec
                    assert 'recommendation' in rec
                    assert 'impact' in rec
                    assert 'feasibility' in rec

    async def test_error_handling(self, mock_llm, base_config):
        """Test error handling with invalid input data"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Test with invalid data structure
        invalid_data = {'data': 'not_a_list'}
        result = await agent.process(invalid_data)
        assert result.status == 'error'
        assert result.error is not None

        # Test with missing consumption data
        invalid_data = {'data': [{'timestamp': '2024-01-01T00:00:00'}]}
        result = await agent.process(invalid_data)
        assert result.status == 'error'
        assert result.error is not None

    async def test_cost_projections(self, mock_llm, base_config, sample_dataframe):
        """Test cost projection calculations"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Ensure DataFrame has datetime index
        df = sample_dataframe.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        projections = agent._generate_projections(df)

        assert 'monthly_forecasts' in projections
        assert 'annual_projection' in projections
        assert isinstance(projections['annual_projection'], float)
        assert projections['annual_projection'] > 0

    async def test_savings_opportunities(self, mock_llm, base_config, sample_dataframe):
        """Test identification of savings opportunities"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Ensure DataFrame has datetime index
        df = sample_dataframe.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        savings = agent._identify_savings(df)

        assert 'rate_optimization' in savings
        assert 'usage_optimization' in savings
        assert 'peak_reduction' in savings

        for category in savings.values():
            assert 'potential_savings' in category
            assert isinstance(category['potential_savings'], (int, float))
            assert category['potential_savings'] >= 0

    async def test_prioritize_recommendations(self, mock_llm, base_config):
        """Test recommendation prioritization"""
        agent = CostAnalystAgent(llm=mock_llm, config=base_config)

        recommendations = {
            'immediate_actions': [
                "Switch to optimal rate plan",
                "Adjust peak usage patterns"
            ],
            'long_term_strategies': [
                "Install energy monitoring system",
                "Upgrade to energy-efficient equipment"
            ]
        }

        prioritized = await agent.prioritize_recommendations(recommendations)

        assert 'high_priority' in prioritized
        assert 'medium_priority' in prioritized
        assert 'low_priority' in prioritized

        # Check structure of prioritized recommendations
        for priority_level in prioritized.values():
            assert isinstance(priority_level, list)
            for item in priority_level:
                assert 'category' in item
                assert 'recommendation' in item
                assert 'impact' in item
                assert 'feasibility' in item
                
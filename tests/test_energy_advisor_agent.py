import pytest
from src.agents.energy_advisor import EnergyAdvisorAgent

@pytest.mark.asyncio
class TestEnergyAdvisorAgent:

    async def test_create_agent(self, mock_llm):
        """Test energy advisor agent creation"""
        agent = EnergyAdvisorAgent.create_agent(mock_llm)
        assert agent.role == 'Energy Efficiency Advisor'
        assert agent.verbose is True
        assert agent.allow_delegation is False

    async def test_generate_season_recommendations(self):
        """Test seasonal recommendations generation"""
        seasonal_data = {
            'summer': 150.0,
            'winter': 120.0,
            'spring': 100.0,
            'fall': 110.0
        }

        recommendations = EnergyAdvisorAgent.generate_season_recommendations(seasonal_data)

        assert 'priority_season' in recommendations
        assert 'recommendations' in recommendations
        assert 'general_recommendations' in recommendations

        # Verify recommendations for highest consumption season
        assert recommendations['priority_season'] == 'summer'  # Based on mock data
        assert len(recommendations['recommendations']) > 0
        assert len(recommendations['general_recommendations']) > 0

        # Verify recommendation content
        summer_recs = recommendations['recommendations']
        assert any('AC' in rec or 'cooling' in rec.lower() for rec in summer_recs)

    async def test_generate_consumption_recommendations(self):
        """Test consumption-based recommendations"""
        analysis_results = {
            "trend": "increasing",
            "anomalies": [
                {
                    "timestamp": "2024-01-01T14:00:00",
                    "value": 150.0,
                    "type": "peak_usage"
                }
            ]
        }

        recommendations = EnergyAdvisorAgent.generate_consumption_recommendations(analysis_results)

        assert 'immediate_actions' in recommendations
        assert 'long_term_actions' in recommendations
        assert 'behavioral_changes' in recommendations

        # Verify recommendation types for increasing trend
        immediate = recommendations['immediate_actions']
        assert any('audit' in action.lower() for action in immediate)

        long_term = recommendations['long_term_actions']
        assert any('system' in action.lower() for action in long_term)

        behavioral = recommendations['behavioral_changes']
        assert any('monitoring' in action.lower() for action in behavioral)

    async def test_recommendations_for_anomalies(self):
        """Test recommendations specifically for anomalies"""
        analysis_results = {
            "anomalies": [
                {
                    "timestamp": "2024-01-01T14:00:00",
                    "value": 200.0,
                    "severity": "high"
                }
            ]
        }

        recommendations = EnergyAdvisorAgent.generate_consumption_recommendations(analysis_results)
        immediate_actions = recommendations['immediate_actions']

        assert any('investigate' in action.lower() for action in immediate_actions)
        assert any('equipment' in action.lower() for action in immediate_actions)

    async def test_empty_analysis_handling(self):
        """Test handling of empty or minimal analysis results"""
        empty_analysis = {
            "trend": None,
            "anomalies": []
        }

        recommendations = EnergyAdvisorAgent.generate_consumption_recommendations(empty_analysis)

        # Should still provide basic recommendations
        assert all(key in recommendations for key in
                   ['immediate_actions', 'long_term_actions', 'behavioral_changes'])
        assert all(len(recommendations[key]) > 0 for key in recommendations)

    async def test_seasonal_edge_cases(self):
        """Test seasonal recommendations with edge cases"""
        # Test with equal seasonal consumption
        equal_seasons = {
            'summer': 100.0,
            'winter': 100.0,
            'spring': 100.0,
            'fall': 100.0
        }
        equal_recs = EnergyAdvisorAgent.generate_season_recommendations(equal_seasons)
        assert 'priority_season' in equal_recs

        # Test with missing seasons
        partial_seasons = {
            'summer': 100.0,
            'winter': 120.0
        }
        partial_recs = EnergyAdvisorAgent.generate_season_recommendations(partial_seasons)
        assert partial_recs['priority_season'] == 'winter'

    async def test_recommendation_consistency(self):
        """Test consistency of recommendations across multiple calls"""
        analysis_results = {
            "trend": "increasing",
            "anomalies": []
        }

        # Generate recommendations multiple times
        rec_sets = [
            EnergyAdvisorAgent.generate_consumption_recommendations(analysis_results)
            for _ in range(3)
        ]

        # Verify consistency in recommendation structure and content
        for recs in rec_sets:
            assert all(key in recs for key in
                       ['immediate_actions', 'long_term_actions', 'behavioral_changes'])

        # Compare recommendation sets
        base_set = rec_sets[0]
        for other_set in rec_sets[1:]:
            assert len(base_set['immediate_actions']) == len(other_set['immediate_actions'])
            assert len(base_set['long_term_actions']) == len(other_set['long_term_actions'])

    async def test_recommendation_relevance(self):
        """Test relevance of recommendations to input data"""
        # Test high summer consumption
        summer_data = {'summer': 200.0, 'winter': 100.0}
        summer_recs = EnergyAdvisorAgent.generate_season_recommendations(summer_data)
        assert any('cooling' in rec.lower() or 'AC' in rec
                   for rec in summer_recs['recommendations'])

        # Test high winter consumption
        winter_data = {'summer': 100.0, 'winter': 200.0}
        winter_recs = EnergyAdvisorAgent.generate_season_recommendations(winter_data)
        assert any('heating' in rec.lower() or 'insulation' in rec.lower()
                   for rec in winter_recs['recommendations'])
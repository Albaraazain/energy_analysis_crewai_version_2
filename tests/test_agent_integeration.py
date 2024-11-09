import pytest
from datetime import datetime
from src.agents.data_analyst import DataAnalystAgent
from src.agents.cost_analyst import CostAnalystAgent
from src.agents.pattern_recognition import PatternRecognitionAgent
from src.agents.energy_advisor import EnergyAdvisorAgent
from crewai import Crew, Process


@pytest.mark.asyncio
class TestAgentIntegration:

    async def test_full_analysis_pipeline(self, mock_llm, base_config, sample_energy_data):
        """Test complete analysis pipeline with all agents"""
        # Initialize agents
        data_analyst = DataAnalystAgent(llm=mock_llm, config=base_config)
        cost_analyst = CostAnalystAgent(llm=mock_llm, config=base_config)
        pattern_recognition = PatternRecognitionAgent(llm=mock_llm, config=base_config)
        energy_advisor = EnergyAdvisorAgent.create_agent(mock_llm)

        # Process data through data analyst
        data_analysis = await data_analyst.process(sample_energy_data)
        assert data_analysis.status == 'success'

        # Process data through cost analyst
        cost_analysis = await cost_analyst.process(sample_energy_data)
        assert cost_analysis.status == 'success'

        # Process data through pattern recognition
        pattern_analysis = await pattern_recognition.process(sample_energy_data)
        assert pattern_analysis.status == 'success'

        # Verify data flows between agents
        assert all(key in data_analysis.data for key in ['statistics', 'patterns', 'anomalies'])
        assert all(key in cost_analysis.data['analysis'] for key in ['rate_analysis', 'cost_patterns'])
        assert 'patterns' in pattern_analysis.data

    async def test_crew_formation(self, mock_llm, base_config):
        """Test crew formation and task delegation"""
        # Create agents
        data_analyst = DataAnalystAgent(llm=mock_llm, config=base_config)
        cost_analyst = CostAnalystAgent(llm=mock_llm, config=base_config)
        energy_advisor = EnergyAdvisorAgent.create_agent(mock_llm)

        # Create crew
        crew = Crew(
            agents=[data_analyst.agent, cost_analyst.agent, energy_advisor],
            tasks=[],
            process=Process.sequential
        )

        assert len(crew.agents) == 3
        assert crew.process == Process.sequential

    async def test_data_cost_integration(self, mock_llm, base_config, sample_energy_data):
        """Test integration between data analyst and cost analyst"""
        data_analyst = DataAnalystAgent(llm=mock_llm, config=base_config)
        cost_analyst = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Get data analysis
        data_analysis = await data_analyst.process(sample_energy_data)
        assert data_analysis.status == 'success'

        # Use data analysis insights for cost analysis
        cost_input = {
            'data': sample_energy_data['data'],
            'insights': data_analysis.data['insights']
        }
        cost_analysis = await cost_analyst.process(cost_input)
        assert cost_analysis.status == 'success'

        # Verify cost analysis uses data insights
        assert 'analysis' in cost_analysis.data
        assert 'recommendations' in cost_analysis.data

    async def test_pattern_advisor_integration(self, mock_llm, base_config, sample_energy_data):
        """Test integration between pattern recognition and energy advisor"""
        pattern_recognition = PatternRecognitionAgent(llm=mock_llm, config=base_config)

        # Get pattern analysis
        pattern_analysis = await pattern_recognition.process(sample_energy_data)
        assert pattern_analysis.status == 'success'

        # Generate recommendations based on patterns
        significant_patterns = pattern_analysis.data.get('significant_patterns', [])
        seasonal_data = {
            pattern['season']: pattern.get('value', 0)
            for pattern in significant_patterns
            if 'season' in pattern
        }

        recommendations = EnergyAdvisorAgent.generate_season_recommendations(seasonal_data)
        assert 'recommendations' in recommendations
        assert 'general_recommendations' in recommendations

    async def test_error_propagation(self, mock_llm, base_config):
        """Test error handling and propagation between agents"""
        data_analyst = DataAnalystAgent(llm=mock_llm, config=base_config)
        cost_analyst = CostAnalystAgent(llm=mock_llm, config=base_config)

        # Test with invalid data
        invalid_data = {'data': 'invalid'}

        # Check data analyst error
        data_result = await data_analyst.process(invalid_data)
        assert data_result.status == 'error'
        assert data_result.error is not None

        # Verify error propagation to cost analyst
        cost_result = await cost_analyst.process(invalid_data)
        assert cost_result.status == 'error'
        assert cost_result.error is not None

    async def test_recommendation_aggregation(self, mock_llm, base_config, sample_energy_data):
        """Test aggregation of recommendations from multiple agents"""
        cost_analyst = CostAnalystAgent(llm=mock_llm, config=base_config)
        pattern_recognition = PatternRecognitionAgent(llm=mock_llm, config=base_config)

        # Get recommendations from both agents
        cost_analysis = await cost_analyst.process(sample_energy_data)
        pattern_analysis = await pattern_recognition.process(sample_energy_data)

        # Verify each analysis succeeded
        assert cost_analysis.status == 'success'
        assert pattern_analysis.status == 'success'

        # Combine insights for energy advisor
        combined_analysis = {
            "cost_patterns": cost_analysis.data['analysis'].get('cost_patterns', {}),
            "consumption_patterns": pattern_analysis.data.get('patterns', {})
        }

        # Generate final recommendations
        recommendations = EnergyAdvisorAgent.generate_consumption_recommendations(combined_analysis)

        # Verify combined recommendations
        assert 'immediate_actions' in recommendations
        assert 'long_term_actions' in recommendations
        assert 'behavioral_changes' in recommendations

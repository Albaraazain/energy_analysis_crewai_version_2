import pytest
from src.agents.base import BaseAgent, AgentResult
from crewai import Agent
from typing import Dict, Any


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent"""

    def create_agent(self) -> Agent:
        return Agent(
            role='Test Agent',
            goal='Testing',
            backstory='Test agent for unit testing',
            verbose=self.config.get('verbose', True),
            allow_delegation=True,
            llm=self.llm
        )

    async def process(self, data: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            status='success',
            data=data,
            metadata={'test': True}
        )


@pytest.mark.asyncio
class TestBaseAgent:

    async def test_agent_initialization(self, mock_llm, base_config):
        """Test agent initialization"""
        agent = TestAgent(llm=mock_llm, config=base_config)
        assert agent.llm == mock_llm
        assert agent.config == base_config
        assert len(agent.results_history) == 0

    async def test_agent_property(self, mock_llm, base_config):
        """Test agent property creates and caches agent"""
        agent = TestAgent(llm=mock_llm, config=base_config)
        # First access creates the agent
        assert agent.agent is not None
        initial_agent = agent.agent
        # Second access returns cached agent
        assert agent.agent is initial_agent

    async def test_store_result(self, mock_llm, base_config):
        """Test storing agent results"""
        agent = TestAgent(llm=mock_llm, config=base_config)
        result = AgentResult(
            status='success',
            data={'test': 'data'},
            metadata={'test': True}
        )
        agent.store_result(result)
        assert len(agent.results_history) == 1
        assert agent.results_history[0] == result

    async def test_get_history(self, mock_llm, base_config):
        """Test retrieving agent history"""
        agent = TestAgent(llm=mock_llm, config=base_config)
        result1 = AgentResult(status='success', data={'test': 1}, metadata={})
        result2 = AgentResult(status='success', data={'test': 2}, metadata={})

        agent.store_result(result1)
        agent.store_result(result2)

        history = agent.get_history()
        assert len(history) == 2
        assert history[0] == result1
        assert history[1] == result2

    async def test_process_implementation(self, mock_llm, base_config):
        """Test process method implementation"""
        agent = TestAgent(llm=mock_llm, config=base_config)
        test_data = {'test': 'data'}
        result = await agent.process(test_data)

        assert isinstance(result, AgentResult)
        assert result.status == 'success'
        assert result.data == test_data
        assert result.metadata['test'] is True

    async def test_agent_result_model(self):
        """Test AgentResult model validation"""
        # Test valid initialization
        result = AgentResult(
            status='success',
            data={'test': 'data'},
            metadata={'test': True}
        )
        assert result.status == 'success'
        assert result.data['test'] == 'data'
        assert result.metadata['test'] is True
        assert result.error is None

        # Test with error
        result_with_error = AgentResult(
            status='error',
            data={},
            metadata={},
            error='Test error'
        )
        assert result_with_error.status == 'error'
        assert result_with_error.error == 'Test error'

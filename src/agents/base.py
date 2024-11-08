# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from crewai import Agent

class AgentResult(BaseModel):
    """Model for standardized agent results"""
    status: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None

class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, llm, config: Dict[str, Any]):
        self.llm = llm
        self.config = config
        self._agent: Optional[Agent] = None
        self.results_history: List[AgentResult] = []

    @abstractmethod
    def create_agent(self) -> Agent:
        """Create and configure the agent"""
        pass

    @property
    def agent(self) -> Agent:
        """Get or create the agent instance"""
        if not self._agent:
            self._agent = self.create_agent()
        return self._agent

    def store_result(self, result: AgentResult):
        """Store agent execution results"""
        self.results_history.append(result)

    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> AgentResult:
        """Process data and return results"""
        pass

    def get_history(self) -> List[AgentResult]:
        """Get agent execution history"""
        return self.results_history
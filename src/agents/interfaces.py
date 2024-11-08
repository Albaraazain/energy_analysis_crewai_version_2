# src/agents/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from ..core.types import AgentMetadata

class IAnalysisAgent(ABC):
    """Interface for analysis-focused agents"""

    @abstractmethod
    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data analysis"""
        pass

    @abstractmethod
    def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate analysis results"""
        pass

class IRecommendationAgent(ABC):
    """Interface for recommendation-focused agents"""

    @abstractmethod
    async def generate_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis"""
        pass

    @abstractmethod
    def prioritize_recommendations(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize generated recommendations"""
        pass

class IVisualizationAgent(ABC):
    """Interface for visualization-focused agents"""

    @abstractmethod
    async def create_visualizations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create data visualizations"""
        pass

    @abstractmethod
    def export_visualizations(self, visualizations: Dict[str, Any], format: str) -> bytes:
        """Export visualizations in specified format"""
        pass
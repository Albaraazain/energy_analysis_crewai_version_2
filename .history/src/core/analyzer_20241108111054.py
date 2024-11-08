# src/core/analyzer.py

from crewai import Crew, Process, LLM
from typing import Dict, Any
import logging
from datetime import datetime

from config.config import LLM_CONFIG, PROCESS_CONFIG
from src.agents.data_analyst import DataAnalystAgent
from src.agents.energy_advisor import EnergyAdvisorAgent
from src.core.tasks import TaskDefinitions


import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyAnalyzer:
    """
    Core class for energy consumption analysis using CrewAI.
    Version 1: Implementation with two agents and sequential processing.
    """

    def __init__(self):
        """Initialize the EnergyAnalyzer with default configuration."""
        try:
            self.llm = LLM(
                model=LLM_CONFIG["model"],
                base_url=LLM_CONFIG["base_url"],
                api_key=LLM_CONFIG["api_key"],
                temperature=LLM_CONFIG["temperature"]
            )
            logger.info("LLM initialized successfully")

            # Initialize agents
            self.data_analyst = DataAnalystAgent.create_agent(self.llm)
            self.energy_advisor = EnergyAdvisorAgent.create_agent(self.llm)
            logger.info("Agents initialized successfully")

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    def analyze(self, monthly_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze energy consumption data and provide recommendations.

        Args:
            monthly_data: Dictionary of monthly consumption data
                        Format: {'YYYY-MM': consumption_value}

        Returns:
            Dictionary containing analysis results and recommendations
        """
        try:
            # Validate input data
            if not self._validate_data(monthly_data):
                raise ValueError("Invalid data format")

            # Create tasks
            analysis_task = TaskDefinitions.create_analysis_task(
                monthly_data,
                self.data_analyst
            )

            # Create crew for initial analysis
            analysis_crew = Crew(
                agents=[self.data_analyst],
                tasks=[analysis_task],
                verbose=PROCESS_CONFIG["verbose"],
                max_rpm=PROCESS_CONFIG["max_rpm"],
                process=Process.sequential
            )

            # Execute analysis
            logger.info("Starting energy consumption analysis")
            analysis_result = analysis_crew.kickoff()

            # Parse analysis result
            try:
                analysis_data = json.loads(analysis_result.raw)
            except json.JSONDecodeError:
                logger.warning("Failed to parse analysis result as JSON")
                analysis_data = {"raw_analysis": analysis_result.raw}

            # Create recommendation tasks
            recommendation_task = TaskDefinitions.create_recommendation_task(
                analysis_data,
                self.energy_advisor
            )

            # Create crew for recommendations
            recommendation_crew = Crew(
                agents=[self.energy_advisor],
                tasks=[recommendation_task],
                verbose=PROCESS_CONFIG["verbose"],
                max_rpm=PROCESS_CONFIG["max_rpm"],
                process=Process.sequential
            )

            # Generate recommendations
            logger.info("Generating recommendations")
            recommendation_result = recommendation_crew.kickoff()

            # Parse recommendation result
            try:
                recommendation_data = json.loads(recommendation_result.raw)
            except json.JSONDecodeError:
                logger.warning("Failed to parse recommendations as JSON")
                recommendation_data = {"raw_recommendations": recommendation_result.raw}

            # Combine results
            return {
                "analysis": analysis_data,
                "recommendations": recommendation_data,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise

    def _validate_data(self, data: Dict[str, float]) -> bool:
        """Validate input data format."""
        if not data or not isinstance(data, dict):
            return False

        try:
            for month, value in data.items():
                # Validate date format (YYYY-MM)
                if not (isinstance(month, str) and len(month.split('-')) == 2):
                    return False
                # Validate consumption value
                if not (isinstance(value, (int, float)) and value >= 0):
                    return False
            return True
        except Exception:
            return False
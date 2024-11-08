# src/core/tasks.py

from crewai import Task
from typing import Dict, Any
import json

class TaskDefinitions:
    """
    Defines and creates tasks for the energy analysis crew.
    """

    @staticmethod
    def create_analysis_task(data: Dict[str, float], agent: Any) -> Task:
        """
        Create a task for analyzing energy consumption data.

        Args:
            data: Monthly consumption data
            agent: Data Analyst agent instance
        """
        return Task(
            description=f"""Analyze the provided monthly energy consumption data and identify key patterns.
            
            Data to analyze: {json.dumps(data, indent=2)}
            
            Required Analysis:
            1. Calculate basic statistics (average, median, min, max)
            2. Identify seasonal patterns and trends
            3. Detect any anomalies in consumption
            4. Provide monthly comparison insights
            
            Return the analysis in JSON format with the following structure:
            {{
                "basic_stats": {{...}},
                "patterns": {{...}},
                "anomalies": {{...}},
                "insights": [...]
            }}
            """,
            agent=agent,
            expected_output="JSON formatted analysis of energy consumption patterns",
            context_variables={
                "data": data,
                "required_metrics": ["basic_stats", "patterns", "anomalies", "insights"]
            }
        )

    @staticmethod
    def create_recommendation_task(analysis_result: Dict, agent: Any) -> Task:
        """
        Create a task for generating energy-saving recommendations.

        Args:
            analysis_result: Results from the analysis task
            agent: Energy Advisor agent instance
        """
        return Task(
            description=f"""Based on the provided analysis, generate specific and actionable recommendations 
            for reducing energy consumption.
            
            Analysis Results: {json.dumps(analysis_result, indent=2)}
            
            Required Recommendations:
            1. Immediate actions based on identified patterns
            2. Seasonal recommendations
            3. Long-term improvement suggestions
            4. Behavioral changes
            
            Format recommendations in JSON with the following structure:
            {{
                "immediate_actions": [...],
                "seasonal_recommendations": {{...}},
                "long_term_improvements": [...],
                "behavioral_changes": [...]
            }}
            """,
            agent=agent,
            expected_output="JSON formatted energy-saving recommendations",
            context_variables={
                "analysis_result": analysis_result
            }
        )
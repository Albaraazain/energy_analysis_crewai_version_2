# src/agents/energy_advisor.py

from crewai import Agent
from typing import Dict, List
import json

class EnergyAdvisorAgent:
    """
    Agent responsible for providing energy-saving recommendations based on analysis.
    """


    @staticmethod
    def create_agent(llm) -> Agent:
        """
        Create and configure the Energy Advisor agent.

        Args:
            llm: Language model instance to be used by the agent

        Returns:
            Agent: Configured Energy Advisor agent
        """
        return Agent(
            role='Energy Efficiency Advisor',
            goal='Provide actionable recommendations for reducing energy consumption',
            backstory="""You are an experienced energy efficiency advisor who helps 
            homeowners reduce their energy consumption through practical and 
            cost-effective recommendations. You specialize in analyzing consumption 
            patterns and suggesting targeted improvements.""",
            verbose=True,
            llm=llm,
            allow_delegation=False
        )

    @staticmethod
    def generate_season_recommendations(seasonal_data: Dict[str, float]) -> Dict:
        """
        Generate season-specific recommendations based on consumption patterns.

        Args:
            seasonal_data: Dictionary containing seasonal consumption averages

        Returns:
            Dictionary of seasonal recommendations
        """
        recommendations = {
            "summer": [
                "Use ceiling fans to improve air circulation",
                "Install window shades or awnings",
                "Set thermostat to optimal summer temperature (78°F/26°C)",
                "Service your AC system regularly"
            ],
            "winter": [
                "Seal drafts around windows and doors",
                "Reverse ceiling fan direction",
                "Set thermostat to optimal winter temperature (68°F/20°C)",
                "Service your heating system"
            ],
            "spring_fall": [
                "Use natural ventilation when possible",
                "Conduct seasonal maintenance",
                "Adjust thermostat settings for mild weather",
                "Check insulation effectiveness"
            ]
        }

        # Identify the season with the highest consumption
        highest_season = max(seasonal_data.items(), key=lambda x: x[1])[0]

        return {
            "priority_season": highest_season,
            "recommendations": recommendations[highest_season],
            "general_recommendations": [
                "Regular equipment maintenance",
                "Monitor daily consumption patterns",
                "Consider energy-efficient appliance upgrades",
                "Implement a home energy monitoring system"
            ]
        }


    @staticmethod
    def generate_consumption_recommendations(analysis_results: Dict) -> Dict:
        """
        Generate specific recommendations based on consumption analysis.

        Args:
            analysis_results: Dictionary containing consumption analysis results

        Returns:
            Dictionary of targeted recommendations
        """
        recommendations = {
            "immediate_actions": [],
            "long_term_actions": [],
            "behavioral_changes": []
        }

        # Add recommendations based on trend
        if analysis_results.get("trend") == "increasing":
            recommendations["immediate_actions"].extend([
                "Conduct an energy audit",
                "Check for malfunctioning equipment",
                "Review thermostat settings"
            ])
            recommendations["long_term_actions"].extend([
                "Consider smart home energy management system",
                "Plan for energy-efficient appliance upgrades",
                "Evaluate insulation improvements"
            ])

        # Add recommendations for anomalies
        if analysis_results.get("anomalies"):
            recommendations["immediate_actions"].extend([
                "Investigate unusual consumption periods",
                "Check for equipment issues during high-usage months",
                "Consider peak usage timing adjustments"
            ])

        # Add behavioral recommendations
        recommendations["behavioral_changes"].extend([
            "Develop energy-conscious habits",
            "Create a schedule for equipment usage",
            "Regular monitoring of energy consumption",
            "Educate household members about energy conservation"
        ])

        return recommendations

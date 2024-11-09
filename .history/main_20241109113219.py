import os
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process

from src.agents.data_analyst import DataAnalystAgent
from src.agents.cost_analyst import CostAnalystAgent
from src.agents.pattern_recognition import PatternRecognitionAgent
from src.agents.energy_advisor import EnergyAdvisorAgent
from src.core.types import EnergyData

# Load environment variables
load_dotenv()

# Configure Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

def create_llm():
    """Create Groq LLM instance with Llama 3 70B model"""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-groq-70b-8192-tool-use-preview",
        temperature=0.7,
        max_tokens=8192
    )

def load_sample_data(days: int = 30) -> Dict[str, Any]:
    """Generate sample energy consumption data"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='H'
    )
    
    # Generate synthetic data
    data = []
    for date in dates:
        # Simulate daily and seasonal patterns
        hour = date.hour
        day_factor = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)  # Daily pattern
        season_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)  # Seasonal pattern
        
        # Base consumption with some randomness
        consumption = 30 * day_factor * season_factor * (0.9 + 0.2 * np.random.random())
        
        # Simulate temperature based on time of day and season
        base_temp = 68 + 15 * np.sin(2 * np.pi * date.dayofyear / 365)
        temp_variation = 10 * np.sin(2 * np.pi * hour / 24)
        temperature = base_temp + temp_variation
        
        data.append({
            "timestamp": date.isoformat(),
            "consumption": round(consumption, 2),
            "rate": 0.12 + 0.04 * (hour >= 14 and hour <= 19),  # Peak rate during 2-7 PM
            "temperature": round(temperature, 1)
        })
    
    return {"data": data}

async def main():
    """Main execution function"""
    # Initialize LLM
    llm = create_llm()
    
    # Configuration for agents
    config = {
        "verbose": True,
        "tools_enabled": True
    }
    
    # Create agents
    data_analyst = DataAnalystAgent(llm=llm, config=config)
    cost_analyst = CostAnalystAgent(llm=llm, config=config)
    pattern_recognition = PatternRecognitionAgent(llm=llm, config=config)
    energy_advisor = EnergyAdvisorAgent.create_agent(llm=llm)
    
    # Load sample data
    data = load_sample_data(days=30)
    
    try:
        # Process data through different agents
        print("🔍 Starting Data Analysis...")
        data_analysis = await data_analyst.process(data)
        print("✅ Data Analysis Complete\n")
        
        print("💰 Starting Cost Analysis...")
        cost_analysis = await cost_analyst.process(data)
        print("✅ Cost Analysis Complete\n")
        
        print("🎯 Identifying Patterns...")
        pattern_analysis = await pattern_recognition.process(data)
        print("✅ Pattern Recognition Complete\n")
        
        # Create tasks for the crew
        tasks = [
            Task(
                description="Analyze energy consumption patterns and identify anomalies",
                agent=data_analyst.agent
            ),
            Task(
                description="Analyze cost implications and identify savings opportunities",
                agent=cost_analyst.agent
            ),
            Task(
                description="Generate comprehensive energy optimization recommendations",
                agent=energy_advisor
            )
        ]
        
        # Create and run the crew
        crew = Crew(
            agents=[data_analyst.agent, cost_analyst.agent, energy_advisor],
            tasks=tasks,
            process=Process.sequential
        )
        
        print("🤖 Starting Crew Analysis...")
        result = await crew.kickoff()
        
        # Print results
        print("\n📊 Analysis Results:")
        print("-------------------")
        print("\n1. Data Analysis Insights:")
        for insight in data_analysis.data.get('insights', []):
            print(f"- {insight.get('description')}")
        
        print("\n2. Cost Analysis Findings:")
        cost_data = cost_analysis.data.get('analysis', {})
        for category, findings in cost_data.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            if isinstance(findings, dict):
                for key, value in findings.items():
                    print(f"- {key}: {value}")
        
        print("\n3. Pattern Recognition Results:")
        patterns = pattern_analysis.data.get('significant_patterns', [])
        for pattern in patterns:
            print(f"- {pattern}")
        
        print("\n4. Crew Analysis Results:")
        print(result)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
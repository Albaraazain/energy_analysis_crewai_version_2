# config/config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
LLM_CONFIG = {
    "model": "llama3-groq-70b-8192-tool-use-preview",
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": os.getenv("GROQ_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 1.0,
    "stream": False,
    "n": 1
}

# Agent Configuration
AGENT_CONFIG = {
    "data_analyst": {
        "role": "Energy Data Analyst",
        "goal": "Analyze energy consumption patterns and provide detailed insights",
        "backstory": """You are an experienced energy data analyst specializing in 
        residential energy consumption. Your expertise lies in interpreting energy 
        usage data and identifying consumption patterns, anomalies, and potential 
        areas for improvement."""
    },
    "energy_advisor": {
        "role": "Energy Efficiency Advisor",
        "goal": "Provide actionable recommendations for energy savings",
        "backstory": """You are an energy efficiency expert with years of experience 
        in helping homeowners reduce their energy consumption. You provide practical, 
        actionable advice based on consumption patterns and best practices in 
        residential energy efficiency."""
    }
}

# Process Configuration
PROCESS_CONFIG = {
    "verbose": True,
    "max_rpm": 10,
    "language": "en",
    "cache": True
}

MEMORY_CONFIG = {
    "provider": "groq",
    "storage_dir": "data/memory"
}

DATABASE_CONFIG = {
    "long_term_db": "sqlite:///data/long_term_memory.db"
}



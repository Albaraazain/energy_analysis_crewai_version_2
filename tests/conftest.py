from datetime import datetime
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv(override=True)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = Mock(spec=ChatGroq)
    mock.temperature = 0.7
    mock.model_name = "llama3-groq-70b-8192-tool-use-preview"
    return mock

@pytest.fixture
def base_config():
    """Base configuration for testing"""
    return {
        'verbose': True,
        'tools_enabled': True,
        'max_retries': 3,
        'timeout': 300,
        'memory': {
            'type': 'short_term',
            'capacity': 10
        }
    }

@pytest.fixture
def sample_energy_data():
    """Generate sample energy consumption data"""
    dates = pd.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 7),
        freq='H'
    )

    data = []
    for date in dates:
        # Create synthetic data with known patterns
        hour = date.hour
        day_factor = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)

        # Base consumption with controlled randomness
        consumption = 30 * day_factor * (0.9 + 0.2 * np.random.RandomState(42).random())

        # Temperature with daily pattern
        temperature = 68 + 10 * np.sin(2 * np.pi * hour / 24)

        data.append({
            "timestamp": date.isoformat(),
            "consumption": round(consumption, 2),
            "rate": 0.12 + 0.04 * (hour >= 14 and hour <= 19),  # Peak rate 2-7 PM
            "temperature": round(temperature, 1)
        })

    return {"data": data}

@pytest.fixture
def sample_dataframe(sample_energy_data):
    """Convert sample data to DataFrame"""
    df = pd.DataFrame(sample_energy_data['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def mock_agent_result():
    """Mock successful agent result"""
    return {
        'status': 'success',
        'data': {
            'patterns': {},
            'metrics': {},
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'analysis_quality': {
                'completeness': 1.0,
                'reliability': 0.9
            }
        }
    }
# tests/conftest.py
import asyncio
import os
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pytest
from dotenv import load_dotenv

# Configure pytest-asyncio to use asyncio as the default event loop
pytest.register_assert_rewrite('pytest_asyncio')


def pytest_configure(config):
    """
    Pytest configuration function to set up the test environment
    """
    # Load environment variables
    load_dotenv(override=True)

    # Register async marker
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as requiring asyncio"
    )

    # Verify required environment variables
    if not os.getenv('GROQ_API_KEY'):
        raise EnvironmentError(
            "GROQ_API_KEY not found in environment variables. "
            "Please ensure your .env file contains this variable."
        )

# Configure asyncio plugin
def pytest_addoption(parser):
    parser.addini(
        'asyncio_mode',
        'run async tests in "strict" mode',
        type="string",
        default="strict"
    )

# Set up async fixture loop scope
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing"""
    return {
        'llm': {
            'model': 'llama3-groq-70b-8192-tool-use-preview',
            'base_url': 'https://api.groq.com/openai/v1',
            'temperature': 0.7
        },
        'memory': {
            'provider': 'groq',
            'storage_dir': 'test_storage'
        },
        'process': {
            'type': 'hierarchical',
            'verbose': True
        },
        'task_queue': {
            'max_retries': 3,
            'timeout': 300
        }
    }


@pytest.fixture
def sample_energy_data() -> pd.DataFrame:
    """Generate sample energy consumption data"""
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-12-31',
        freq='H'
    )

    # Generate realistic consumption patterns
    base_load = 0.5 + np.random.normal(0, 0.1, len(dates))
    daily_pattern = np.sin(np.pi * dates.hour / 12) * 0.3
    seasonal_pattern = np.sin(np.pi * dates.dayofyear / 182.5) * 0.2

    consumption = (base_load + daily_pattern + seasonal_pattern) * 1000
    consumption = np.maximum(consumption, 0)  # Ensure non-negative values

    return pd.DataFrame({
        'timestamp': dates,
        'consumption': consumption
    })


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""

    class MockLLM:
        async def generate(self, prompts):
            return [{'text': 'Mock response'}]

        async def embed(self, text):
            return [0.1] * 128

    return MockLLM()
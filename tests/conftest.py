# tests/conftest.py
import pytest
from typing import Dict
import os
from pathlib import Path

@pytest.fixture
def sample_config():
    """Provide sample configuration for testing"""
    return {
        'llm': {
            'model': 'test-model',
            'temperature': 0.7
        },
        'memory': {
            'provider': 'test',
            'storage_dir': 'test_storage'
        },
        'process': {
            'type': 'hierarchical',
            'verbose': True
        }
    }

@pytest.fixture
def sample_energy_data():
    """Provide sample energy consumption data"""
    return {
        '2024-01': 1000.0,
        '2024-02': 950.0,
        '2024-03': 800.0
    }

@pytest.fixture
def temp_storage_path(tmp_path):
    """Provide temporary storage path"""
    storage_dir = tmp_path / "test_storage"
    storage_dir.mkdir()
    return storage_dir
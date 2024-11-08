# tests/test_llm_operations.py
import pytest
from datetime import datetime
import os
from dotenv import load_dotenv
from src.core.memory.long_term import LongTermMemory
from src.core.memory.base import MemoryEntry

# Load environment variables
load_dotenv(override=True)

TEST_CONFIG = {
    'database_config': {
        'long_term_db': 'sqlite:///data/test_llm_operations.db'
    },
    'embedder': {
        'model': 'sentence-transformers/all-MiniLM-L6-v2'
    },
    'llm': {
        'model': 'mixtral-8x7b-32768'
    },
    'groq_api_key': os.getenv('GROQ_API_KEY')
}

@pytest.fixture(scope="function")
def memory_instance():
    """Create a test memory instance"""
    os.makedirs('data', exist_ok=True)
    return LongTermMemory(TEST_CONFIG)

@pytest.fixture
def sample_content():
    """Create sample content for testing"""
    return {
        "type": "energy_reading",
        "value": 150.5,
        "unit": "kWh",
        "timestamp": datetime.now().isoformat(),
        "location": "Building A",
        "device_id": "meter_001"
    }

@pytest.mark.asyncio
async def test_summary_generation(memory_instance, sample_content):
    """Test summary generation"""
    print("Starting test_summary_generation")
    summary = await memory_instance._generate_summary(sample_content)
    print(f"Generated summary: {summary}")
    assert isinstance(summary, str)
    assert len(summary) > 0

@pytest.mark.asyncio
async def test_importance_calculation(memory_instance, sample_content):
    """Test importance score calculation"""
    print("Starting test_importance_calculation")
    score = await memory_instance._calculate_importance(sample_content)
    print(f"Calculated importance score: {score}")
    assert isinstance(score, float)
    assert 0 <= score <= 1

@pytest.mark.asyncio
async def test_embedding_generation(memory_instance, sample_content):
    """Test embedding generation"""
    print("Starting test_embedding_generation")
    embedding = await memory_instance._generate_embedding(sample_content)
    print(f"Generated embedding: {embedding}")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, (int, float)) for x in embedding)

@pytest.mark.asyncio
async def test_similarity_calculation(memory_instance, sample_content):
    """Test similarity calculation between embeddings"""
    print("Starting test_similarity_calculation")
    embedding1 = await memory_instance._generate_embedding(sample_content)
    print(f"Generated embedding1: {embedding1}")
    embedding2 = await memory_instance._generate_embedding(sample_content)
    print(f"Generated embedding2: {embedding2}")

    similarity = memory_instance._calculate_similarity(embedding1, embedding2)
    print(f"Calculated similarity: {similarity}")
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1
    assert similarity > 0.9  # Same content should have high similarity

@pytest.mark.asyncio
async def test_store_with_llm_processing(memory_instance):
    """Test storing entry with LLM processing"""
    print("Starting test_store_with_llm_processing")
    entry = MemoryEntry(
        timestamp=datetime.now(),
        content={
            "type": "important_event",
            "description": "Critical system maintenance performed",
            "impact": "high",
            "duration": "2 hours"
        },
        metadata={"priority": "high"},
        source="maintenance_log",
        tags=["maintenance", "critical"]
    )

    success = await memory_instance.store(entry)
    print(f"Store operation success: {success}")
    assert success, "Failed to store entry with LLM processing"

if __name__ == '__main__':
    pytest.main([__file__])
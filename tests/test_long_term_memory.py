# tests/test_long_term_memory.py
import pytest
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from src.core.memory.long_term import LongTermMemory
from src.core.memory.base import MemoryEntry

# Load environment variables at the start
load_dotenv(override=True)

# Verify GROQ_API_KEY is loaded
if not os.getenv('GROQ_API_KEY'):
    raise EnvironmentError("GROQ_API_KEY not found in environment variables")

# Test configuration
TEST_CONFIG = {
    'database_config': {
        'long_term_db': 'sqlite:///data/test_long_term_memory.db'
    },
    'embedder': {
        'model': 'sentence-transformers/all-MiniLM-L6-v2'
    },
    'llm': {
        'model': 'mixtral-8x7b-32768'
    },
    'groq_api_key': os.getenv('GROQ_API_KEY')  # Explicitly pass the API key
}

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

@pytest.fixture(scope="function")
def memory_instance():
    """Create a test memory instance"""
    mem = LongTermMemory(TEST_CONFIG)
    return mem

@pytest.fixture
def sample_entry():
    """Create a sample memory entry"""
    return MemoryEntry(
        timestamp=datetime.now(),
        content={
            "type": "energy_reading",
            "value": 150.5,
            "unit": "kWh"
        },
        metadata={
            "device_id": "meter_001",
            "location": "main_building"
        },
        source="test_device",
        tags=["test", "energy", "consumption"]
    )

@pytest.mark.asyncio
async def test_store_and_retrieve(memory_instance, sample_entry):
    """Test storing and retrieving a memory entry"""
    # Store the entry
    success = await memory_instance.store(sample_entry)
    assert success, "Failed to store memory entry"

    # Retrieve using a relevant query
    memories = await memory_instance.retrieve("energy reading kWh", limit=1)
    assert len(memories) > 0, "Failed to retrieve stored memory"

    retrieved = memories[0]
    assert retrieved.source == sample_entry.source
    assert retrieved.content['value'] == sample_entry.content['value']
    assert retrieved.metadata['device_id'] == sample_entry.metadata['device_id']

    # Cleanup
    await memory_instance.clear()

@pytest.mark.asyncio
async def test_update(memory_instance, sample_entry):
    """Test updating a memory entry"""
    # First store the entry
    await memory_instance.store(sample_entry)

    # Update the entry
    entry_id = str(sample_entry.timestamp.timestamp())
    updates = {
        'content': {
            "type": "energy_reading",
            "value": 160.5,  # Updated value
            "unit": "kWh"
        }
    }

    success = await memory_instance.update(entry_id, updates)
    assert success, "Failed to update memory entry"

    # Retrieve and verify update
    memories = await memory_instance.retrieve("energy reading kWh", limit=1)
    assert len(memories) > 0
    assert memories[0].content['value'] == 160.5

    # Cleanup
    await memory_instance.clear()

@pytest.mark.asyncio
async def test_clear(memory_instance, sample_entry):
    """Test clearing all memories"""
    # Store some entries
    await memory_instance.store(sample_entry)

    # Clear all memories
    success = await memory_instance.clear()
    assert success, "Failed to clear memories"

    # Verify memories are cleared
    memories = await memory_instance.retrieve("energy reading", limit=10)
    assert len(memories) == 0, "Memories were not properly cleared"

@pytest.mark.asyncio
async def test_query_by_metadata(memory_instance, sample_entry):
    """Test querying memories by metadata"""
    # Store the entry
    await memory_instance.store(sample_entry)

    # Query by metadata
    memories = await memory_instance.query_by_metadata({
        "device_id": "meter_001",
        "location": "main_building"
    })

    assert len(memories) > 0, "Failed to retrieve by metadata"
    assert memories[0].metadata['device_id'] == "meter_001"

    # Cleanup
    await memory_instance.clear()

@pytest.mark.asyncio
async def test_query_by_time_range(memory_instance, sample_entry):
    """Test querying memories by time range"""
    # Store the entry
    await memory_instance.store(sample_entry)

    # Query by time range
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now() + timedelta(hours=1)

    memories = await memory_instance.query_by_time_range(start_time, end_time)
    assert len(memories) > 0, "Failed to retrieve by time range"

    # Cleanup
    await memory_instance.clear()

@pytest.mark.asyncio
async def test_store_many(memory_instance):
    """Test storing multiple memory entries"""
    # Create multiple entries
    entries = [
        MemoryEntry(
            timestamp=datetime.now() - timedelta(minutes=i),
            content={"value": i, "unit": "kWh"},
            metadata={"test_id": f"test_{i}"},
            source="test_device",
            tags=["test"]
        )
        for i in range(3)
    ]

    # Store multiple entries
    success = await memory_instance.store_many(entries)
    assert success, "Failed to store multiple entries"

    # Verify storage
    memories = await memory_instance.retrieve("kWh", limit=10)
    assert len(memories) == 3, "Not all entries were stored"

    # Cleanup
    await memory_instance.clear()

@pytest.mark.asyncio
async def test_initialization(memory_instance):
    """Test proper initialization of memory system"""
    assert memory_instance.embedder is not None, "Embedder not initialized"
    assert memory_instance.llm is not None, "LLM not initialized"
    assert hasattr(memory_instance, 'Session'), "Database session not initialized"

if __name__ == '__main__':
    pytest.main([__file__])
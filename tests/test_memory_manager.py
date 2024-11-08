# tests/test_memory_manager.py
import pytest
from datetime import datetime
import os
from dotenv import load_dotenv
from src.core.memory.manager import MemoryManager
from src.core.memory.base import MemoryEntry

# Load environment variables
load_dotenv(override=True)

TEST_CONFIG = {
    'database_config': {
        'long_term_db': 'sqlite:///data/test_manager_long_term.db',
        'entity_db': 'sqlite:///data/test_manager_entity.db'
    },
    'embedder': {
        'model': 'sentence-transformers/all-MiniLM-L6-v2',
        'batch_size': 32,
        'max_seq_length': 128
    },
    'llm': {
        'model': 'mixtral-8x7b-32768',
    },
    'groq_api_key': os.getenv('GROQ_API_KEY')
}

@pytest.fixture(scope="function")
def memory_manager():
    """Create a test memory manager instance"""
    os.makedirs('data', exist_ok=True)
    manager = MemoryManager(TEST_CONFIG)
    yield manager
    # Cleanup after tests
    manager.clear_memories('all')

@pytest.fixture
def sample_entries():
    """Create sample memory entries"""
    return [
        MemoryEntry(
            timestamp=datetime.now(),
            content={
                "type": "energy_reading",
                "value": 150.5,
                "unit": "kWh"
            },
            metadata={
                "device_id": "meter_001",
                "location": "building_a"
            },
            source="test_device",
            tags=["energy", "consumption"]
        ),
        MemoryEntry(
            timestamp=datetime.now(),
            content={
                "type": "maintenance_record",
                "action": "system_upgrade",
                "component": "meter"
            },
            metadata={
                "device_id": "meter_001",
                "priority": "high"
            },
            source="maintenance_log",
            tags=["maintenance", "upgrade"]
        )
    ]

@pytest.mark.asyncio
async def test_store_memories(memory_manager, sample_entries):
    """Test storing memories in different memory systems"""
    # Test long-term storage
    success = await memory_manager.store_memory(sample_entries[0], 'long_term')
    assert success, "Failed to store in long-term memory"

    # Test entity storage
    success = await memory_manager.store_memory(sample_entries[1], 'entity')
    assert success, "Failed to store in entity memory"

@pytest.mark.asyncio
async def test_retrieve_memories(memory_manager, sample_entries):
    """Test retrieving memories from different memory systems"""
    # Store test data
    await memory_manager.store_memory(sample_entries[0], 'long_term')
    await memory_manager.store_memory(sample_entries[1], 'entity')

    # Test retrieval from long-term memory
    long_term_results = await memory_manager.retrieve_memories(
        "energy consumption",
        memory_type='long_term',
        limit=5
    )
    assert len(long_term_results) > 0, "No results from long-term memory"

    # Test retrieval from entity memory
    entity_results = await memory_manager.retrieve_memories(
        "maintenance upgrade",
        memory_type='entity',
        limit=5
    )
    assert len(entity_results) > 0, "No results from entity memory"

@pytest.mark.asyncio
async def test_cross_memory_search(memory_manager, sample_entries):
    """Test searching across different memory types"""
    # Store entries in both memory systems
    for entry in sample_entries:
        await memory_manager.store_memory(entry, 'long_term')
        await memory_manager.store_memory(entry, 'entity')

    # Search with the same query across both systems
    query = "meter"
    long_term_results = await memory_manager.retrieve_memories(query, 'long_term')
    entity_results = await memory_manager.retrieve_memories(query, 'entity')

    assert len(long_term_results) > 0, "No results from long-term memory"
    assert len(entity_results) > 0, "No results from entity memory"

@pytest.mark.asyncio
async def test_memory_updates(memory_manager, sample_entries):
    """Test updating memories in different systems"""
    # Store initial entries
    entry = sample_entries[0]
    await memory_manager.store_memory(entry, 'long_term')

    # Update the entry
    entry_id = str(entry.timestamp.timestamp())
    updates = {
        'content': {
            "type": "energy_reading",
            "value": 160.5,  # Updated value
            "unit": "kWh"
        }
    }

    success = await memory_manager.update_memory(
        entry_id,
        updates,
        memory_type='long_term'
    )
    assert success, "Failed to update memory"

    # Verify update
    results = await memory_manager.retrieve_memories(
        "energy consumption",
        memory_type='long_term'
    )
    assert any(r.content.get('value') == 160.5 for r in results)

@pytest.mark.asyncio
async def test_memory_clearing(memory_manager, sample_entries):
    """Test clearing memories from different systems"""
    # Store entries in both systems
    for entry in sample_entries:
        await memory_manager.store_memory(entry, 'long_term')
        await memory_manager.store_memory(entry, 'entity')

    # Test clearing individual systems
    success = await memory_manager.clear_memories('long_term')
    assert success, "Failed to clear long-term memory"

    long_term_results = await memory_manager.retrieve_memories(
        "any",
        memory_type='long_term'
    )
    assert len(long_term_results) == 0, "Long-term memory not cleared"

    # Test clearing all memories
    success = await memory_manager.clear_memories('all')
    assert success, "Failed to clear all memories"

    entity_results = await memory_manager.retrieve_memories(
        "any",
        memory_type='entity'
    )
    assert len(entity_results) == 0, "Entity memory not cleared"

@pytest.mark.asyncio
async def test_bulk_operations(memory_manager, sample_entries):
    """Test bulk memory operations"""
    # Test bulk storage
    success = await memory_manager.store_many(sample_entries, 'long_term')
    assert success, "Failed to store multiple entries"

    # Verify all entries were stored
    results = await memory_manager.retrieve_memories(
        "meter",
        memory_type='long_term',
        limit=10
    )
    assert len(results) == len(sample_entries), "Not all entries were stored"

if __name__ == '__main__':
    pytest.main([__file__])
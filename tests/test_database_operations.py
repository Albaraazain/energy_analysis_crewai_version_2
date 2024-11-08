# tests/test_database_operations.py
import pytest
from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from src.core.memory.long_term import LongTermMemory, MemoryRecord
from src.core.memory.base import MemoryEntry

# Load environment variables
load_dotenv(override=True)

TEST_CONFIG = {
    'database_config': {
        'long_term_db': 'sqlite:///data/test_db_operations.db'
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
    mem = LongTermMemory(TEST_CONFIG)
    yield mem
    # Cleanup database after each test
    with Session(mem.engine) as session:
        session.query(MemoryRecord).delete()
        session.commit()

@pytest.fixture
def sample_record():
    """Create a sample memory entry"""
    return MemoryEntry(
        timestamp=datetime.now(),
        content={"test": "data"},
        metadata={"test_meta": "value"},
        source="test",
        tags=["test"]
    )

@pytest.mark.asyncio
async def test_database_insert(memory_instance, sample_record):
    """Test database insert operations"""
    success = await memory_instance.store(sample_record)
    assert success, "Failed to insert record"

    with Session(memory_instance.engine) as session:
        record = session.get(MemoryRecord, str(sample_record.timestamp.timestamp()))
        assert record is not None
        assert record.content == sample_record.content
        assert record.source == sample_record.source

@pytest.mark.asyncio
async def test_database_update(memory_instance, sample_record):
    """Test database update operations"""
    # First store the record
    await memory_instance.store(sample_record)

    # Update the record
    record_id = str(sample_record.timestamp.timestamp())
    new_content = {"test": "updated_data"}
    success = await memory_instance.update(record_id, {'content': new_content})
    assert success, "Failed to update record"

    with Session(memory_instance.engine) as session:
        record = session.get(MemoryRecord, record_id)
        assert record.content == new_content

@pytest.mark.asyncio
async def test_database_delete(memory_instance, sample_record):
    """Test database delete operations"""
    await memory_instance.store(sample_record)
    success = await memory_instance.clear()
    assert success, "Failed to delete records"

    with Session(memory_instance.engine) as session:
        count = session.query(MemoryRecord).count()
        assert count == 0, "Records were not deleted"

@pytest.mark.asyncio
async def test_database_query(memory_instance, sample_record):
    """Test database query operations"""
    await memory_instance.store(sample_record)

    with Session(memory_instance.engine) as session:
        # Test different query methods
        record = session.get(MemoryRecord, str(sample_record.timestamp.timestamp()))
        assert record is not None

        records = session.query(MemoryRecord).filter_by(source="test").all()
        assert len(records) == 1

        record = session.query(MemoryRecord).filter(
            MemoryRecord.tags.contains(["test"])
        ).first()
        assert record is not None

@pytest.mark.asyncio
async def test_database_connection(memory_instance):
    """Test database connection and session management"""
    with Session(memory_instance.engine) as session:
        assert session.is_active

        # Test transaction rollback
        try:
            session.add(MemoryRecord(
                id="invalid",  # This should cause an error
                timestamp=None,  # This should cause an error
                content={},
                memory_metadata={},
                source="",
                tags=[]
            ))
            session.commit()
            assert False, "Should have raised an error"
        except:
            session.rollback()
            assert session.is_active, "Session should still be active after rollback"

if __name__ == '__main__':
    pytest.main([__file__])
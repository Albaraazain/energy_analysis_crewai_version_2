# src/core/memory/long_term.py
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .base import BaseMemory, MemoryEntry
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage


Base = declarative_base()

class MemoryRecord(Base):
    """SQLAlchemy model for long-term memory storage"""
    __tablename__ = 'long_term_memories'

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    content = Column(JSON, nullable=False)
    memory_metadata = Column(JSON, nullable=False)
    source = Column(String, nullable=False)
    tags = Column(JSON, nullable=False)
    summary = Column(Text, nullable=True)
    importance_score = Column(Float, nullable=False, default=0.0)
    embedding = Column(JSON, nullable=True)

class LongTermMemory(BaseMemory):
    """Implementation of long-term memory using SQLAlchemy"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the long-term memory system"""
        super().__init__(config)
        db_url = config.get('database_config', {}).get('long_term_db', 'sqlite:///data/long_term_memory.db')
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)


    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry in SQLite database"""
        session = self.Session()
        try:
            # Generate summary and importance score
            summary = await self._generate_summary(entry.content)
            importance_score = await self._calculate_importance(entry.content)

            # Generate embedding for the content
            content_embedding = await self._generate_embedding(entry.content)

            # Create record with UUID
            record = MemoryRecord(
                id=self.generate_id(),  # Use UUID instead of timestamp
                timestamp=entry.timestamp,
                content=entry.content,
                memory_metadata=entry.metadata,
                source=entry.source,
                tags=entry.tags,
                summary=summary,
                importance_score=importance_score,
                embedding=content_embedding
            )

            session.add(record)
            session.commit()
            return True
        except Exception as e:
            print(f"Error storing long-term memory: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()

    async def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve memories based on semantic search"""
        session = self.Session()
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding({"query": query})

            # Get all records and calculate similarity
            records = session.query(MemoryRecord).all()
            similarities = []

            for record in records:
                if record.embedding:  # Check if embedding exists
                    similarity = self._calculate_similarity(query_embedding, record.embedding)
                    similarities.append((record, similarity))

            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_records = similarities[:limit]

            # Convert to MemoryEntry objects
            return [
                MemoryEntry(
                    timestamp=record.timestamp,
                    content=record.content,
                    metadata=record.memory_metadata,
                    source=record.source,
                    tags=record.tags
                )
                for record, _ in top_records
            ]
        except Exception as e:
            print(f"Error retrieving long-term memories: {str(e)}")
            return []
        finally:
            session.close()

    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory entry"""
        session = self.Session()
        try:
            record = session.get(MemoryRecord, entry_id)
            if not record:
                return False

            if 'content' in updates:
                record.content = updates['content']
                record.summary = await self._generate_summary(updates['content'])
                record.importance_score = await self._calculate_importance(updates['content'])
                record.embedding = await self._generate_embedding(updates['content'])

            if 'metadata' in updates:
                record.memory_metadata = updates['metadata']
            if 'tags' in updates:
                record.tags = updates['tags']
            if 'source' in updates:
                record.source = updates['source']

            session.commit()
            return True
        except Exception as e:
            print(f"Error updating memory: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()

    async def clear(self) -> bool:
        """Clear all memories"""
        session = self.Session()
        try:
            session.query(MemoryRecord).delete()
            session.commit()
            return True
        except Exception as e:
            print(f"Error clearing memories: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()

    async def _generate_summary(self, content: Dict[str, Any]) -> str:
        """Generate a summary of the memory content"""
        try:
            content_str = str(content)
            messages = [
                SystemMessage(content="You are a helpful assistant that generates concise summaries."),
                HumanMessage(content=f"Summarize this concisely: {content_str}")
            ]
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return ""

    async def _calculate_importance(self, content: Dict[str, Any]) -> float:
        """Calculate importance score for memory"""
        try:
            content_str = str(content)
            messages = [
                SystemMessage(content="You are a helpful assistant that rates importance on a scale of 0 to 1."),
                HumanMessage(content=f"Rate the importance of this information from 0 to 1: {content_str}")
            ]
            response = await self.llm.agenerate([messages])
            score_str = response.generations[0][0].text

            # Extract numerical value from response
            import re
            numbers = re.findall(r"0?\.[0-9]+|[01]", score_str)
            if numbers:
                return float(numbers[0])
            return 0.5  # Default middle importance if no number found
        except Exception as e:
            print(f"Error calculating importance: {str(e)}")
            return 0.0

    async def query_by_metadata(self, metadata_filter: Dict[str, Any],
                                limit: Optional[int] = None) -> List[MemoryEntry]:
        """Query memories based on metadata"""
        session = self.Session()
        try:
            query = session.query(MemoryRecord)
            records = query.all()

            # Filter records based on metadata
            filtered_records = []
            for record in records:
                if all(record.memory_metadata.get(k) == v
                       for k, v in metadata_filter.items()):
                    filtered_records.append(record)

            # Apply limit if specified
            if limit is not None:
                filtered_records = filtered_records[:limit]

            # Convert to MemoryEntry objects
            return [
                MemoryEntry(
                    timestamp=record.timestamp,
                    content=record.content,
                    metadata=record.memory_metadata,
                    source=record.source,
                    tags=record.tags
                )
                for record in filtered_records
            ]
        except Exception as e:
            print(f"Error querying by metadata: {str(e)}")
            return []
        finally:
            session.close()

    async def query_by_time_range(self, start_time: datetime,
                                  end_time: datetime) -> List[MemoryEntry]:
        """Query memories within a specific time range"""
        session = self.Session()
        try:
            records = (
                session.query(MemoryRecord)
                .filter(MemoryRecord.timestamp >= start_time)
                .filter(MemoryRecord.timestamp <= end_time)
                .all()
            )

            return [
                MemoryEntry(
                    timestamp=record.timestamp,
                    content=record.content,
                    metadata=record.memory_metadata,
                    source=record.source,
                    tags=record.tags
                )
                for record in records
            ]
        except Exception as e:
            print(f"Error querying by time range: {str(e)}")
            return []
        finally:
            session.close()
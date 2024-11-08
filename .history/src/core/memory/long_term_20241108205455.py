# src/core/memory/long_term.py
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .base import BaseMemory, MemoryEntry

Base = declarative_base()

class MemoryRecord(Base):
    """SQLAlchemy model for long-term memory storage"""
    __tablename__ = 'long_term_memories'

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    content = Column(JSON, nullable=False)
    metadata = Column(JSON, nullable=False)
    source = Column(String, nullable=False)
    tags = Column(JSON, nullable=False)
    summary = Column(Text, nullable=True)
    importance_score = Column(Float, nullable=False, default=0.0)

class LongTermMemory(BaseMemory):
    """Implementation of long-term memory using SQLAlchemy"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine = create_engine(config['long_term_db'])
        Base.memory_metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _initialize_embedder(self):
        """Initialize the Groq embedder"""
        from langchain_community.embeddings import GroqEmbeddings

        return GroqEmbeddings(
            model=self.config['embedder']['model'],
            groq_api_key=self.config['groq_api_key']
        )

    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry in SQLite database"""
        try:
            session = self.Session()

            # Generate summary and importance score
            summary = await self._generate_summary(entry.content)
            importance_score = await self._calculate_importance(entry.content)

            # Create record
            record = MemoryRecord(
                id=str(entry.timestamp.timestamp()),
                timestamp=entry.timestamp,
                content=entry.content,
                metadata=entry.memory_metadata,
                source=entry.source,
                tags=entry.tags,
                summary=summary,
                importance_score=importance_score
            )

            session.add(record)
            session.commit()
            session.close()

            return True
        except Exception as e:
            print(f"Error storing long-term memory: {str(e)}")
            session.rollback()
            session.close()
            return False

    async def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve memories based on semantic search"""
        try:
            session = self.Session()

            # Generate query embedding
            query_embedding = await self._generate_embedding({"query": query})

            # Get all records and calculate similarity
            records = session.query(MemoryRecord).all()
            similarities = []

            for record in records:
                content_embedding = await self._generate_embedding(record.content)
                similarity = self._calculate_similarity(query_embedding, content_embedding)
                similarities.append((record, similarity))

            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_records = similarities[:limit]

            # Convert to MemoryEntry objects
            entries = []
            for record, _ in top_records:
                entries.append(MemoryEntry(
                    timestamp=record.timestamp,
                    content=record.content,
                    metadata=record.metadata,
                    source=record.source,
                    tags=record.tags
                ))

            session.close()
            return entries
        except Exception as e:
            print(f"Error retrieving long-term memories: {str(e)}")
            session.close()
            return []

    async def _generate_summary(self, content: Dict[str, Any]) -> str:
        """Generate a summary of the memory content"""
        try:
            content_str = str(content)
            # Use Groq for summarization
            response = await self.llm.agenerate(
                prompts=[f"Summarize this concisely: {content_str}"]
            )
            return response.generations[0][0].text
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return ""

    async def _calculate_importance(self, content: Dict[str, Any]) -> float:
        """Calculate importance score for memory"""
        try:
            content_str = str(content)
            # Use Groq to assess importance
            response = await self.llm.agenerate(
                prompts=[
                    f"Rate the importance of this information from 0 to 1: {content_str}"
                ]
            )
            score_str = response.generations[0][0].text
            return float(score_str)
        except Exception as e:
            print(f"Error calculating importance: {str(e)}")
            return 0.0
# src/core/memory/entity.py

from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .base import BaseMemory, MemoryEntry
import numpy as np
import json
import asyncio

Base = declarative_base()


class EntityRecord(Base):
    """SQLAlchemy model for entity memory storage"""
    __tablename__ = 'entity_memories'

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    entity_type = Column(String, nullable=False)
    content = Column(JSON, nullable=False)
    memory_metadata = Column(JSON, nullable=False)
    source = Column(String, nullable=False)
    tags = Column(JSON, nullable=False)
    attributes = Column(JSON, nullable=False)
    relationships = Column(JSON, nullable=False)


class EntityMemory(BaseMemory):
    """Implementation of entity-based memory using SQLAlchemy"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the entity memory system"""
        super().__init__(config)
        self.engine = create_engine(config.get('entity_db', 'sqlite:///entity_memory.db'))
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.llm = self._initialize_llm()
        self.embedder = self._initialize_embedder()

    def _initialize_embedder(self):
        """Initialize the Groq embedder"""
        try:
            from langchain_groq import ChatGroqEmbeddings
            return ChatGroqEmbeddings(
                model=self.config['embedder']['model'],
                groq_api_key=self.config['groq_api_key']
            )
        except ImportError as e:
            raise ImportError("Failed to import ChatGroqEmbeddings: ensure the correct package is installed.") from e

    def _initialize_llm(self):
        """Initialize LLM for text generation"""
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=self.config['groq_api_key'],
                model_name=self.config['llm']['model']
            )
        except ImportError as e:
            raise ImportError("Failed to import ChatGroq: ensure the correct package is installed.") from e

    async def store(self, entry: MemoryEntry) -> bool:
        """Store an entity memory entry"""
        session = self.Session()
        try:
            entity_info = await self._extract_entity_info(entry.content)
            record = EntityRecord(
                id=str(entry.timestamp.timestamp()),
                timestamp=entry.timestamp,
                entity_type=entity_info['type'],
                content=entry.content,
                memory_metadata=entry.metadata,
                source=entry.source,
                tags=entry.tags,
                attributes=entity_info['attributes'],
                relationships=entity_info['relationships']
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            print(f"Error storing entity memory: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()

    async def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve entity memories based on semantic search"""
        session = self.Session()
        try:
            query_embedding = await self._generate_embedding({"query": query})
            records = session.query(EntityRecord).all()
            similarities = []

            for record in records:
                content_embedding = await self._generate_embedding({
                    "content": record.content,
                    "attributes": record.attributes,
                    "relationships": record.relationships
                })
                similarity = self._calculate_similarity(query_embedding, content_embedding)
                similarities.append((record, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_records = similarities[:limit]

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
            print(f"Error retrieving entity memories: {str(e)}")
            return []
        finally:
            session.close()

    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing entity memory entry"""
        session = self.Session()
        try:
            record = session.query(EntityRecord).get(entry_id)
            if not record:
                return False

            if 'content' in updates:
                record.content = updates['content']
                entity_info = await self._extract_entity_info(updates['content'])
                record.entity_type = entity_info['type']
                record.attributes = entity_info['attributes']
                record.relationships = entity_info['relationships']

            if 'metadata' in updates:
                record.memory_metadata = updates['metadata']
            if 'tags' in updates:
                record.tags = updates['tags']
            if 'source' in updates:
                record.source = updates['source']

            session.commit()
            return True
        except Exception as e:
            print(f"Error updating entity memory: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()

    async def clear(self) -> bool:
        """Clear all entity memories"""
        session = self.Session()
        try:
            session.query(EntityRecord).delete()
            session.commit()
            return True
        except Exception as e:
            print(f"Error clearing entity memories: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()

    async def _extract_entity_info(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entity information from content"""
        try:
            content_str = str(content)
            response = await self.llm.agenerate(
                prompts=[
                    f"""Analyze this content and extract:
                    1. Entity type
                    2. Key attributes
                    3. Relationships to other entities
                    Content: {content_str}
                    Return as JSON with keys: type, attributes, relationships"""
                ]
            )
            entity_info = json.loads(response.generations[0][0].text)
            return {
                'type': entity_info.get('type', 'unknown'),
                'attributes': entity_info.get('attributes', {}),
                'relationships': entity_info.get('relationships', [])
            }
        except Exception as e:
            print(f"Error extracting entity info: {str(e)}")
            return {
                'type': 'unknown',
                'attributes': {},
                'relationships': []
            }

    async def _generate_embedding(self, content: Dict[str, Any]) -> List[float]:
        """Generate embedding for content using Groq"""
        try:
            content_str = str(content)
            embeddings = await self.embedder.aembed_query(content_str)
            return embeddings
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        try:
            a = np.array(embedding1)
            b = np.array(embedding2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

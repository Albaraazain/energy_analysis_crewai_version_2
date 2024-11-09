# src/core/memory/entity.py
from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from .base import BaseMemory, MemoryEntry
import numpy as np
import json
from langchain_core.messages import SystemMessage, HumanMessage

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
    embedding = Column(JSON, nullable=True)

class EntityMemory(BaseMemory):
    """Implementation of entity-based memory using SQLAlchemy"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the entity memory system"""
        super().__init__(config)
        self.engine = create_engine(config.get('database_config', {}).get('entity_db', 'sqlite:///data/entity_memory.db'))
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    async def store(self, entry: MemoryEntry) -> bool:
        """Store an entity memory entry"""
        session = self.Session()
        try:
            # Generate unique ID and extract entity info
            entry_id = self.generate_id()
            entity_info = await self._extract_entity_info(entry.content)
            content_embedding = await self._generate_embedding(entry.content)

            record = EntityRecord(
                id=entry_id,
                timestamp=entry.timestamp,
                entity_type=entity_info['type'],
                content=entry.content,
                memory_metadata=entry.metadata,
                source=entry.source,
                tags=entry.tags,
                attributes=entity_info['attributes'],
                relationships=entity_info['relationships'],
                embedding=content_embedding
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

    async def _extract_entity_info(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entity information from content"""
        try:
            content_str = str(content)
            prompt = f"""Given this content, please extract:
1. A single type field (string)
2. An attributes object (key-value pairs)
3. A relationships array (list of related entities)

Content: {content_str}

Return a valid JSON object with exactly these fields:
{{
    "type": "entity_type_here",
    "attributes": {{"key1": "value1", "key2": "value2"}},
    "relationships": []
}}"""

            messages = [
                SystemMessage(content="You are a helpful assistant that extracts structured entity information."),
                HumanMessage(content=prompt)
            ]

            response = await self.llm.agenerate([messages])
            try:
                entity_info = json.loads(response.generations[0][0].text)
                # Validate and ensure required fields
                return {
                    'type': str(entity_info.get('type', 'unknown')),
                    'attributes': dict(entity_info.get('attributes', {})),
                    'relationships': list(entity_info.get('relationships', []))
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract just the JSON portion
                text = response.generations[0][0].text
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    try:
                        entity_info = json.loads(text[json_start:json_end])
                        return {
                            'type': str(entity_info.get('type', 'unknown')),
                            'attributes': dict(entity_info.get('attributes', {})),
                            'relationships': list(entity_info.get('relationships', []))
                        }
                    except:
                        pass

                return {
                    'type': 'unknown',
                    'attributes': {},
                    'relationships': []
                }
        except Exception as e:
            print(f"Error extracting entity info: {str(e)}")
            return {
                'type': 'unknown',
                'attributes': {},
                'relationships': []
            }

    async def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve entity memories based on semantic search"""
        session = self.Session()
        try:
            query_embedding = await self._generate_embedding({"query": query})
            records = session.query(EntityRecord).all()
            similarities = []

            for record in records:
                if record.embedding:  # Check if embedding exists
                    similarity = self._calculate_similarity(query_embedding, record.embedding)
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
            record = session.get(EntityRecord, entry_id)
            if not record:
                return False

            if 'content' in updates:
                record.content = updates['content']
                entity_info = await self._extract_entity_info(updates['content'])
                record.entity_type = entity_info['type']
                record.attributes = entity_info['attributes']
                record.relationships = entity_info['relationships']
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
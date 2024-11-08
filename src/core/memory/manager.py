# src/core/memory/manager.py
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base import MemoryEntry
from .long_term import LongTermMemory
from .entity import EntityMemory


class MemoryManager:
    """Manager for different types of memory systems"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize memory systems"""
        # Set up configuration with defaults
        self.config = {
            'long_term_db': 'sqlite:///long_term_memory.db',
            'entity_db': 'sqlite:///entity_memory.db',
            'embedder': {
                'model': 'mixture-of-experts',
            },
            'llm': {
                'model': 'mixtral-8x7b-32768',
            },
            **config
        }

        # Initialize memory systems
        self.long_term = LongTermMemory(self.config)
        self.entity = EntityMemory(self.config)

    async def store_memory(self, entry: MemoryEntry, memory_type: str = 'long_term') -> bool:
        """Store memory in specified system"""
        if memory_type == 'long_term':
            return await self.long_term.store(entry)
        elif memory_type == 'entity':
            return await self.entity.store(entry)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

    async def retrieve_memories(self, query: str, memory_type: str = 'long_term',
                                limit: int = 5) -> List[MemoryEntry]:
        """Retrieve memories from specified system"""
        if memory_type == 'long_term':
            return await self.long_term.retrieve(query, limit)
        elif memory_type == 'entity':
            return await self.entity.retrieve(query, limit)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

    async def update_memory(self, entry_id: str, updates: Dict[str, Any],
                            memory_type: str = 'long_term') -> bool:
        """Update memory in specified system"""
        if memory_type == 'long_term':
            return await self.long_term.update(entry_id, updates)
        elif memory_type == 'entity':
            return await self.entity.update(entry_id, updates)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

    async def clear_memories(self, memory_type: str = 'all') -> bool:
        """Clear memories from specified system(s)"""
        try:
            if memory_type in ['all', 'long_term']:
                await self.long_term.clear()
            if memory_type in ['all', 'entity']:
                await self.entity.clear()
            return True
        except Exception as e:
            print(f"Error clearing memories: {str(e)}")
            return False

    async def store_many(self, entries: List[MemoryEntry],
                         memory_type: str = 'long_term') -> bool:
        """Store multiple memories"""
        if memory_type == 'long_term':
            return await self.long_term.store_many(entries)
        elif memory_type == 'entity':
            return await self.entity.store_many(entries)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

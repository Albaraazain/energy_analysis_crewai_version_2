# src/core/memory/short_term.py
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings
from .base import BaseMemory, MemoryEntry
from datetime import datetime, timedelta
from pathlib import Path


class ShortTermMemory(BaseMemory):
    """Implementation of short-term memory using ChromaDB"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        storage_dir = Path(config.get('memory', {}).get('storage_dir', 'data/memory'))
        persist_directory = storage_dir / 'short_term'

        # Initialize the Chroma client with the new configuration
        self.client = chromadb.PersistentClient(path=str(persist_directory))

        self.collection = self._initialize_collection()

    def _initialize_embedder(self):
        """Initialize the Groq embedder"""
        from langchain_groq import ChatGroq

        model_name = self.config.get('embedder', {}).get('model', 'default-model-name')
        api_key = self.config.get('groq_api_key', 'default-api-key')

        return ChatGroq(
            model=model_name,
            temperature=0  # Customize as needed
    )

    def _initialize_collection(self):
        """Initialize or get the memory collection"""
        collection_name = "short_term_memories"
        try:
            return self.client.get_collection(name=collection_name)
        except ValueError:
            return self.client.create_collection(
                name=collection_name,
                metadata={"type": "short_term"}
            )

    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry in ChromaDB"""
        try:
            # Generate embedding
            embedding = await self._generate_embedding(entry.content)

            # Store in ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[entry.json()],
                metadatas=[entry.metadata],
                ids=[str(entry.timestamp.timestamp())]
            )

            return True
        except Exception as e:
            print(f"Error storing memory: {str(e)}")
            return False

    async def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories using similarity search"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding({"query": query})

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )

            # Convert results back to MemoryEntry objects
            entries = []
            for doc in results['documents'][0]:
                entries.append(MemoryEntry.parse_raw(doc))

            return entries
        except Exception as e:
            print(f"Error retrieving memories: {str(e)}")
            return []

    async def _generate_embedding(self, content: Dict[str, Any]) -> List[float]:
        """Generate embedding for content"""
        text = str(content)
        return self.embedder.embed_query(text)

    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory entry"""
        try:
            # Get existing entry
            result = self.collection.get(ids=[entry_id])
            if not result['documents']:
                return False

            # Update entry
            entry = MemoryEntry.parse_raw(result['documents'][0])
            updated_entry = entry.copy(update=updates)

            # Generate new embedding
            embedding = await self._generate_embedding(updated_entry.content)

            # Update in ChromaDB
            self.collection.update(
                embeddings=[embedding],
                documents=[updated_entry.json()],
                metadatas=[updated_entry.metadata],
                ids=[entry_id]
            )

            return True
        except Exception as e:
            print(f"Error updating memory: {str(e)}")
            return False

    async def clear(self) -> bool:
        """Clear all short-term memories"""
        try:
            self.client.delete_collection("short_term_memories")
            self.collection = self._initialize_collection()
            return True
        except Exception as e:
            print(f"Error clearing memories: {str(e)}")
            return False

    async def cleanup_old_memories(self, days: int = 7) -> bool:
        """Clean up memories older than specified days"""
        try:
            cutoff = datetime.now() - timedelta(days=days)

            # Get all memories
            results = self.collection.get()

            # Filter and keep only recent memories
            for idx, timestamp_str in enumerate(results['ids']):
                timestamp = datetime.fromtimestamp(float(timestamp_str))
                if timestamp < cutoff:
                    self.collection.delete(ids=[timestamp_str])

            return True
        except Exception as e:
            print(f"Error cleaning up memories: {str(e)}")
            return False
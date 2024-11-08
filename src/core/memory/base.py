# src/core/memory/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

class MemoryEntry(BaseModel):
    """Model for memory entries"""
    timestamp: datetime
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    source: str
    tags: List[str] = []

class BaseMemory(ABC):
    """Abstract base class for memory implementations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._initialize_llm()
        self.embedder = self._initialize_embedder()

    def _initialize_embedder(self):
        """Initialize the Sentence Transformer embedding model"""
        model_name = self.config.get('embedder', {}).get(
            'model', 'sentence-transformers/all-MiniLM-L6-v2'
        )
        return SentenceTransformer(model_name)

    def _initialize_llm(self):
        """Initialize the LLM"""
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=self.config['groq_api_key'],
            model_name=self.config['llm']['model']
        )

    async def _generate_embedding(self, content: Dict[str, Any]) -> List[float]:
        """Generate embedding for content using HuggingFace model"""
        try:
            content_str = str(content)
            # Generate embedding and convert to list of floats
            embedding = self.embedder.encode(content_str, convert_to_tensor=False).tolist()
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        try:
            # Convert to numpy arrays and ensure they're floating point
            a = np.array(embedding1, dtype=np.float32)
            b = np.array(embedding2, dtype=np.float32)

            # Calculate cosine similarity
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry"""
        pass

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories"""
        pass

    @abstractmethod
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing memory entry"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all memories"""
        pass

    async def store_many(self, entries: List[MemoryEntry]) -> bool:
        """Store multiple memory entries"""
        try:
            results = [await self.store(entry) for entry in entries]
            return all(results)
        except Exception as e:
            print(f"Error storing multiple entries: {str(e)}")
            return False
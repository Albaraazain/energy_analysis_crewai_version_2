# src/core/types.py
from typing import TypedDict, Union, List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel

class ProcessType(str, Enum):
    """Process type enumeration"""
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"

class EnergyData(TypedDict):
    """Energy consumption data structure"""
    timestamp: datetime
    consumption: float
    metadata: Dict[str, Any]

class AnalysisResult(BaseModel):
    """Analysis result model"""
    timestamp: datetime
    metrics: Dict[str, float]
    patterns: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float

class AgentMetadata(BaseModel):
    """Agent metadata model"""
    agent_type: str
    process_time: float
    token_usage: int
    success: bool
    error: Optional[str] = None
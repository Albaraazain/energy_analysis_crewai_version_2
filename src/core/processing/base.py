# src/core/processing/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from pydantic import BaseModel

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class ProcessingResult(BaseModel):
    """Model for processing results"""
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    validation_status: bool
    errors: Optional[List[str]] = None

class DataProcessor(ABC):
    """Abstract base class for data processing components"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = self._setup_validation_rules()

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> ProcessingResult:
        """Process the input data"""
        pass

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate input data"""
        pass

    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Set up data validation rules"""
        return {
            'required_fields': [
                'timestamp', 'consumption'
            ],
            'value_ranges': {
                'consumption': (0, float('inf'))
            },
            'data_types': {
                'consumption': (int, float),
                'timestamp': str
            }
        }

    def _validate_data_structure(self, data: Dict[str, Any]) -> List[str]:
        """Validate data structure and return list of errors"""
        errors = []

        # Check required fields
        for field in self.validation_rules['required_fields']:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check value ranges
        for field, (min_val, max_val) in self.validation_rules['value_ranges'].items():
            if field in data:
                value = data[field]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    errors.append(
                        f"Value for {field} must be between {min_val} and {max_val}"
                    )

        # Check data types
        for field, expected_types in self.validation_rules['data_types'].items():
            if field in data:
                if not isinstance(data[field], expected_types):
                    errors.append(
                        f"Invalid data type for {field}. Expected {expected_types}"
                    )

        return errors
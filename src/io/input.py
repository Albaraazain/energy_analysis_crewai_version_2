# src/io/inputs.py
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import aiofiles
import asyncio

class InputHandler(ABC):
    """Abstract base class for data input handlers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = self._get_supported_formats()
        self.validation_rules = self._setup_validation_rules()

    @abstractmethod
    async def read(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Read data from source"""
        pass

    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate input data"""
        pass

    @abstractmethod
    def _get_supported_formats(self) -> List[str]:
        """Get list of supported formats"""
        pass

    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Set up input validation rules"""
        return {
            'required_fields': ['timestamp', 'consumption'],
            'data_types': {
                'timestamp': (str, datetime),
                'consumption': (int, float)
            },
            'value_ranges': {
                'consumption': (0, float('inf'))
            }
        }

class CSVInputHandler(InputHandler):
    """Handler for CSV input data"""

    def _get_supported_formats(self) -> List[str]:
        return ['.csv']

    async def read(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Read data from CSV file"""
        try:
            # Read file asynchronously
            async with aiofiles.open(source, mode='r') as file:
                content = await file.read()

            # Parse CSV
            df = pd.read_csv(
                pd.StringIO(content),
                parse_dates=['timestamp'] if 'timestamp' in self.validation_rules['required_fields'] else None,
                date_parser=lambda x: pd.to_datetime(x, utc=True)
            )

            # Validate data
            if not self.validate(df.to_dict('records')):
                raise ValueError("Data validation failed")

            return {
                'data': df.to_dict('records'),
                'memory_metadata': {
                    'source': str(source),
                    'format': 'csv',
                    'rows': len(df),
                    'columns': list(df.columns),
                    'timestamp': datetime.now().isoformat()
                }
            }
        except Exception as e:
            raise IOError(f"Error reading CSV file: {str(e)}")

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate CSV data"""
        if not data:
            return False

        # Check required fields
        if not all(field in data[0] for field in self.validation_rules['required_fields']):
            return False

        # Check data types and ranges
        for record in data:
            try:
                for field, types in self.validation_rules['data_types'].items():
                    if not isinstance(record[field], types):
                        return False

                for field, (min_val, max_val) in self.validation_rules['value_ranges'].items():
                    value = record[field]
                    if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                        return False
            except Exception:
                return False

        return True

class JSONInputHandler(InputHandler):
    """Handler for JSON input data"""

    def _get_supported_formats(self) -> List[str]:
        return ['.json']

    async def read(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Read data from JSON file"""
        try:
            # Read file asynchronously
            async with aiofiles.open(source, mode='r') as file:
                content = await file.read()

            # Parse JSON
            data = json.loads(content)

            # Convert to DataFrame for processing
            df = pd.DataFrame(data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Validate data
            if not self.validate(df.to_dict('records')):
                raise ValueError("Data validation failed")

            return {
                'data': df.to_dict('records'),
                'memory_metadata': {
                    'source': str(source),
                    'format': 'json',
                    'rows': len(df),
                    'columns': list(df.columns),
                    'timestamp': datetime.now().isoformat()
                }
            }
        except Exception as e:
            raise IOError(f"Error reading JSON file: {str(e)}")

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate JSON data"""
        return super().validate(data)

class APIInputHandler(InputHandler):
    """Handler for API input data"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_config = config.get('api', {})
        self.session = None

    def _get_supported_formats(self) -> List[str]:
        return ['api']

    async def read(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Read data from API endpoint"""
        try:
            import aiohttp

            if not self.session:
                self.session = aiohttp.ClientSession()

            # Make API request
            async with self.session.get(
                    source,
                    headers=self.api_config.get('headers', {}),
                    params=self.api_config.get('params', {}),
                    timeout=self.api_config.get('timeout', 30)
            ) as response:
                response.raise_for_status()
                data = await response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # Validate data
            if not self.validate(df.to_dict('records')):
                raise ValueError("Data validation failed")

            return {
                'data': df.to_dict('records'),
                'memory_metadata': {
                    'source': str(source),
                    'format': 'api',
                    'rows': len(df),
                    'columns': list(df.columns),
                    'timestamp': datetime.now().isoformat(),
                    'api_metadata': {
                        'status_code': response.status,
                        'headers': dict(response.headers)
                    }
                }
            }
        except Exception as e:
            raise IOError(f"Error reading from API: {str(e)}")
        finally:
            if self.session and source.endswith('/close'):
                await self.session.close()
                self.session = None

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate API data"""
        return super().validate(data)
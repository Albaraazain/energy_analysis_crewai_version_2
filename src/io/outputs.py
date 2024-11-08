# src/io/outputs.py
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import json
import aiofiles
import asyncio
import csv
import io

class OutputHandler(ABC):
    """Abstract base class for data output handlers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_path = Path(config.get('output_path', 'outputs'))
        self.output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def write(self, data: Dict[str, Any], destination: Union[str, Path]) -> bool:
        """Write data to destination"""
        pass

    @abstractmethod
    def format_data(self, data: Dict[str, Any]) -> Any:
        """Format data for output"""
        pass

    @abstractmethod
    def validate_output(self, formatted_data: Any) -> bool:
        """Validate formatted output"""
        pass

class CSVOutputHandler(OutputHandler):
    """Handler for CSV output"""

    async def write(self, data: Dict[str, Any], destination: Union[str, Path]) -> bool:
        """Write data to CSV file"""
        try:
            formatted_data = self.format_data(data)

            if not self.validate_output(formatted_data):
                raise ValueError("Invalid output data format")

            # Write asynchronously
            async with aiofiles.open(destination, mode='w', newline='') as file:
                # Create string buffer
                output = io.StringIO()
                formatted_data.to_csv(output, index=True)
                await file.write(output.getvalue())

            return True
        except Exception as e:
            raise IOError(f"Error writing CSV file: {str(e)}")

    def format_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Format data for CSV output"""
        df = pd.DataFrame(data['data'])

        # Add memory_metadata as additional columns if configured
        if self.config.get('include_metadata', False) and 'memory_metadata' in data:
            for key, value in data['memory_metadata'].items():
                if isinstance(value, (str, int, float)):
                    df[f'metadata_{key}'] = value

        return df

    def validate_output(self, formatted_data: pd.DataFrame) -> bool:
        """Validate CSV output format"""
        return isinstance(formatted_data, pd.DataFrame) and not formatted_data.empty

class JSONOutputHandler(OutputHandler):
    """Handler for JSON output"""

    async def write(self, data: Dict[str, Any], destination: Union[str, Path]) -> bool:
        """Write data to JSON file"""
        try:
            formatted_data = self.format_data(data)

            if not self.validate_output(formatted_data):
                raise ValueError("Invalid output data format")

            # Write asynchronously
            async with aiofiles.open(destination, mode='w') as file:
                await file.write(json.dumps(formatted_data, indent=2))

            return True
        except Exception as e:
            raise IOError(f"Error writing JSON file: {str(e)}")

    def format_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for JSON output"""
        formatted = {
            'data': data['data'],
            'memory_metadata': data.get('memory_metadata', {}),
            'export_timestamp': datetime.now().isoformat()
        }

        # Convert numpy/pandas types to native Python types
        return self._convert_types(formatted)

    def _convert_types(self, obj: Any) -> Any:
        """Convert numpy/pandas types to native Python types"""
        if isinstance(obj, dict):
            return {key: self._convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        return obj

    def validate_output(self, formatted_data: Dict[str, Any]) -> bool:
        """Validate JSON output format"""
        return isinstance(formatted_data, dict) and 'data' in formatted_data

class ExcelOutputHandler(OutputHandler):
    """Handler for Excel output"""

    async def write(self, data: Dict[str, Any], destination: Union[str, Path]) -> bool:
        """Write data to Excel file"""
        try:
            formatted_data = self.format_data(data)

            if not self.validate_output(formatted_data):
                raise ValueError("Invalid output data format")

            # Excel writing must be synchronous due to openpyxl limitations
            await asyncio.to_thread(self._write_excel, formatted_data, destination)
            return True
        except Exception as e:
            raise IOError(f"Error writing Excel file: {str(e)}")

    def _write_excel(self, formatted_data: Dict[str, pd.DataFrame],
                     destination: Union[str, Path]):
        """Write Excel file synchronously"""
        with pd.ExcelWriter(destination, engine='openpyxl') as writer:
            for sheet_name, df in formatted_data.items():
                df.to_excel(writer, sheet_name=sheet_name)

    def format_data(self, data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Format data for Excel output"""
        formatted = {
            'Data': pd.DataFrame(data['data'])
        }

        # Add memory_metadata sheet if configured
        if self.config.get('include_metadata', False) and 'memory_metadata' in data:
            metadata_df = pd.DataFrame([data['memory_metadata']])
            formatted['Metadata'] = metadata_df

        return formatted

    def validate_output(self, formatted_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate Excel output format"""
        return (
                isinstance(formatted_data, dict) and
                all(isinstance(df, pd.DataFrame) for df in formatted_data.values())
        )
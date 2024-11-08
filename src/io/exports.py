# src/io/export.py
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import asyncio
from .outputs import CSVOutputHandler, JSONOutputHandler, ExcelOutputHandler

class ExportManager:
    """Manages data export operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.handlers = {
            'csv': CSVOutputHandler(config),
            'json': JSONOutputHandler(config),
            'excel': ExcelOutputHandler(config)
        }

    async def export_data(self, data: Dict[str, Any],
                          format: str,
                          destination: Optional[Union[str, Path]] = None) -> str:
        """Export data in specified format"""
        if format not in self.handlers:
            raise ValueError(f"Unsupported export format: {format}")

        if destination is None:
            destination = self._generate_default_path(format)

        destination = Path(destination)

        try:
            success = await self.handlers[format].write(data, destination)
            if success:
                return str(destination)
            raise IOError(f"Failed to export data in {format} format")
        except Exception as e:
            raise IOError(f"Error during export: {str(e)}")

    async def export_multiple(self, data: Dict[str, Any],
                              formats: List[str]) -> Dict[str, str]:
        """Export data in multiple formats"""
        results = {}
        tasks = []

        for format in formats:
            if format in self.handlers:
                destination = self._generate_default_path(format)
                task = self.export_data(data, format, destination)
                tasks.append((format, task))

        for format, task in tasks:
            try:
                result = await task
                results[format] = result
            except Exception as e:
                results[format] = str(e)

        return results

    def _generate_default_path(self, format: str) -> Path:
        """Generate default export path"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"energy_data_{timestamp}.{format}"
        return Path(self.config.get('export_path', 'exports')) / filename

class BatchExportManager:
    """Manages batch export operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.export_manager = ExportManager(config)
        self.batch_size = config.get('batch_size', 1000)

    async def export_batch(self, data: List[Dict[str, Any]],
                           format: str) -> List[str]:
        """Export data in batches"""
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_data = {
                'data': batch,
                'memory_metadata': {
                    'batch_number': i // self.batch_size + 1,
                    'batch_size': len(batch),
                    'total_batches': (len(data) + self.batch_size - 1)
                                     // self.batch_size
                }
            }
            try:
                result = await self.export_manager.export_data(
                    batch_data,
                    format,
                    self._generate_batch_path(i // self.batch_size + 1, format)
                )
                results.append(result)
            except Exception as e:
                results.append(str(e))

        return results

    def _generate_batch_path(self, batch_number: int, format: str) -> Path:
        """Generate path for batch export"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"energy_data_batch_{batch_number}_{timestamp}.{format}"
        return Path(self.config.get('export_path', 'exports')) / filename
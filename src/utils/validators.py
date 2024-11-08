# src/utils/validators.py
from typing import Dict, Any, Optional
from datetime import datetime
import json

class DataValidator:
    """Validate input data formats and types"""

    @staticmethod
    def validate_energy_data(data: Dict[str, float]) -> bool:
        """Validate energy consumption data format"""
        try:
            for date_str, value in data.items():
                # Validate date format
                datetime.strptime(date_str, "%Y-%m")
                # Validate value type and range
                if not isinstance(value, (int, float)) or value < 0:
                    return False
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration data"""
        required_fields = ['llm', 'memory', 'process', 'agent']
        return all(field in config for field in required_fields)

# src/utils/logger.py
import logging
from rich.logging import RichHandler
from typing import Optional

class LoggerSetup:
    """Enhanced logging setup with rich formatting"""

    @staticmethod
    def setup_logger(
            name: str,
            level: int = logging.INFO,
            log_file: Optional[str] = None
    ) -> logging.Logger:
        """Set up logger with optional file output"""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Console handler with rich formatting
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True
        )
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


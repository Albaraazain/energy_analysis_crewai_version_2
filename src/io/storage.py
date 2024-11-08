# src/io/storage.py
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import sqlite3
import asyncio
import aiofiles
import json

class StorageHandler(ABC):
    """Abstract base class for data storage handlers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get('storage_path', 'data'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def store(self, data: Dict[str, Any], identifier: str) -> bool:
        """Store data with given identifier"""
        pass

    @abstractmethod
    async def retrieve(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by identifier"""
        pass

    @abstractmethod
    async def delete(self, identifier: str) -> bool:
        """Delete data by identifier"""
        pass

    async def _ensure_storage(self):
        """Ensure storage is ready"""
        pass

class SQLiteStorageHandler(StorageHandler):
    """Handler for SQLite storage"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_path = self.storage_path / 'energy_data.db'
        self.table_name = config.get('table_name', 'energy_readings')
        asyncio.create_task(self._ensure_storage())

    async def _ensure_storage(self):
        """Ensure database and table exist"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        timestamp TIMESTAMP,
                        consumption REAL,
                        memory_metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                await db.commit()
        except Exception as e:
            raise IOError(f"Error initializing database: {str(e)}")

    async def store(self, data: Dict[str, Any], identifier: str) -> bool:
        """Store data in SQLite database"""
        try:
            df = pd.DataFrame(data['data'])
            metadata = json.dumps(data.get('memory_metadata', {}))

            async with aiosqlite.connect(self.db_path) as db:
                await db.executemany(
                    f'''
                    INSERT OR REPLACE INTO {self.table_name} 
                    (id, timestamp, consumption, memory_metadata)
                    VALUES (?, ?, ?, ?)
                    ''',
                    [
                        (
                            f"{identifier}_{idx}",
                            row['timestamp'],
                            row['consumption'],
                            metadata
                        )
                        for idx, row in df.iterrows()
                    ]
                )
                await db.commit()
            return True
        except Exception as e:
            raise IOError(f"Error storing data: {str(e)}")

    async def retrieve(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from SQLite database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                        f'''
                    SELECT timestamp, consumption, memory_metadata
                    FROM {self.table_name}
                    WHERE id LIKE ?
                    ORDER BY timestamp
                    ''',
                        (f"{identifier}%",)
                ) as cursor:
                    rows = await cursor.fetchall()

                if not rows:
                    return None

                df = pd.DataFrame(
                    rows,
                    columns=['timestamp', 'consumption', 'memory_metadata']
                )
                metadata = json.loads(rows[0][2])

                return {
                    'data': df[['timestamp', 'consumption']].to_dict('records'),
                    'memory_metadata': metadata
                }
        except Exception as e:
            raise IOError(f"Error retrieving data: {str(e)}")

    async def delete(self, identifier: str) -> bool:
        """Delete data from SQLite database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f'''
                    DELETE FROM {self.table_name}
                    WHERE id LIKE ?
                    ''',
                    (f"{identifier}%",)
                )
                await db.commit()
            return True
        except Exception as e:
            raise IOError(f"Error deleting data: {str(e)}")
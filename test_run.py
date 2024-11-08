# test_run.py
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Import our components
from src.core.analyzer import EnhancedEnergyAnalyzer
from src.core.memory.manager import MemoryManager
from src.tasks.queue import TaskQueue
from src.tasks.definitions import TaskDefinition, Task, TaskPriority


async def main():
    # Basic configuration
    groq_api_key = os.getenv("GROQ_API_KEY")


    config = {
        "llm": {
            "model": "llama3-groq-70b-8192-tool-use-preview",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq_api_key,
            "temperature": 0.7
        },
        "embedder": {
            "model": "nomic-embed-text-v1_5",
            "api_key": groq_api_key  # Use the same API key
        },
        "memory": {
            "provider": "groq",
            "storage_dir": Path("data/memory")
        },
        "storage_dir": Path("data/memory"),
        "long_term_db": "sqlite:///data/long_term_memory.db",
        "groq_api_key": groq_api_key,  # Add this line
        "process": {
            "type": "hierarchical",
            "verbose": True
        }
    }


    # Generate sample data
    dates = pd.date_range(
        start='2024-01-01',
        end='2024-01-07',  # One week of data
        freq='H'
    )

    # Create realistic consumption patterns
    base_load = 0.5 + np.random.normal(0, 0.1, len(dates))
    daily_pattern = np.sin(np.pi * dates.hour / 12) * 0.3
    seasonal_pattern = np.sin(np.pi * dates.dayofyear / 182.5) * 0.2

    consumption = (base_load + daily_pattern + seasonal_pattern) * 1000
    consumption = np.maximum(consumption, 0)  # Ensure non-negative values

    data = pd.DataFrame({
        'timestamp': dates,
        'consumption': consumption
    })

    try:
        # Initialize components
        print("Initializing components...")
        analyzer = EnhancedEnergyAnalyzer(config)
        memory = MemoryManager(config)
        queue = TaskQueue(config)

        print("Initializing analyzer...")
        await analyzer.initialize(config['llm'])

        # Create and submit analysis task
        print("Creating analysis task...")
        task_def = TaskDefinition(
            name="Energy Analysis",
            priority=TaskPriority.HIGH,
            description="Analyze energy consumption patterns"
        )

        task = Task(task_def)

        print("Submitting task to queue...")
        await queue.submit(task)

        print("Processing task...")
        analysis_task = await queue.process_next()

        print("Running analysis...")
        result = await analyzer.analyze(data.to_dict('records'))

        print("\nAnalysis Results:")
        print("================")
        print(f"Status: {result['status']}")
        print("\nMetrics:")
        print(result.get('data', {}).get('metrics', {}))

        print("\nStoring results in memory...")
        await memory.store_memory(
            content=result['data'],
            source='analysis',
            tags=['analysis', 'energy']
        )

        print("\nRetrieving stored results...")
        stored_results = await memory.query_memory('analysis')

        print(f"\nFound {len(stored_results)} stored results")

        print("\nCompleting task...")
        await queue.complete_task(task.definition.id, result)

        final_status = await queue.get_status(task.definition.id)
        print(f"Final task status: {final_status}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback

        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())

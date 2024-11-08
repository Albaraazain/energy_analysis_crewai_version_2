# tests/test_tasks/test_queue.py
import pytest
from src.tasks.queue import TaskQueue, TaskPriority
from src.tasks.definitions import TaskDefinition, Task

class TestTaskQueue:
    """Test suite for task queue system"""

    @pytest.mark.asyncio
    async def test_priority_ordering(self, sample_config):
        """Test task priority ordering"""
        queue = TaskQueue(sample_config)

        # Create tasks with different priorities
        tasks = [
            Task(TaskDefinition(
                name=f"Task {i}",
                priority=priority,
                description="Test task"
            ))
            for i, priority in enumerate([
                TaskPriority.LOW,
                TaskPriority.CRITICAL,
                TaskPriority.HIGH,
                TaskPriority.MEDIUM
            ])
        ]

        # Submit tasks
        for task in tasks:
            await queue.submit(task)

        # Verify processing order
        first_task = await queue.process_next()
        assert first_task.definition.priority == TaskPriority.CRITICAL

        second_task = await queue.process_next()
        assert second_task.definition.priority == TaskPriority.HIGH

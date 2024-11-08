# src/tasks/queue.py
from typing import Dict, Any, List, Optional, Callable
import asyncio
from datetime import datetime
from .definitions import Task, TaskStatus, TaskPriority
import heapq

class PriorityQueue:
    """Priority queue implementation for tasks"""

    def __init__(self):
        self._queue = []
        self._task_map = {}
        self._lock = asyncio.Lock()

    async def put(self, task: Task) -> None:
        """Add tasks to queue"""
        async with self._lock:
            entry = (
                -task.definition.priority.value,
                task.created_at.timestamp(),
                task
            )
            heapq.heappush(self._queue, entry)
            self._task_map[task.definition.id] = task

    async def get(self) -> Optional[Task]:
        """Get next tasks from queue"""
        async with self._lock:
            while self._queue:
                _, _, task = heapq.heappop(self._queue)
                if task.status == TaskStatus.PENDING:
                    return task
            return None

    async def remove(self, task_id: str) -> bool:
        """Remove tasks from queue"""
        async with self._lock:
            if task_id not in self._task_map:
                return False

            task = self._task_map[task_id]
            self._queue = [
                entry for entry in self._queue
                if entry[2].definition.id != task_id
            ]
            heapq.heapify(self._queue)
            del self._task_map[task_id]
            return True

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get tasks by ID"""
        return self._task_map.get(task_id)

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self._queue) == 0

class TaskQueue:
    """Task queue manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queue = PriorityQueue()
        self.processing_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        self._handlers: Dict[TaskStatus, List[Callable]] = {}

    async def submit(self, task: Task) -> str:
        """Submit tasks to queue"""
        await self.queue.put(task)
        return task.definition.id

    async def process_next(self) -> Optional[Task]:
        """Process next tasks in queue"""
        task = await self.queue.get()
        if task:
            self.processing_tasks[task.definition.id] = task
            return task
        return None

    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """Mark tasks as completed"""
        if task_id in self.processing_tasks:
            task = self.processing_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self.completed_tasks[task_id] = task
            del self.processing_tasks[task_id]
            await self._trigger_handlers(TaskStatus.COMPLETED, task)

    async def fail_task(self, task_id: str, error: str) -> None:
        """Mark tasks as failed"""
        if task_id in self.processing_tasks:
            task = self.processing_tasks[task_id]
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            self.failed_tasks[task_id] = task
            del self.processing_tasks[task_id]
            await self._trigger_handlers(TaskStatus.FAILED, task)

    def add_handler(self, status: TaskStatus, handler: Callable) -> None:
        """Add handler for tasks status"""
        if status not in self._handlers:
            self._handlers[status] = []
        self._handlers[status].append(handler)

    async def _trigger_handlers(self, status: TaskStatus, task: Task) -> None:
        """Trigger handlers for tasks status"""
        if status in self._handlers:
            for handler in self._handlers[status]:
                await handler(task)

    async def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get tasks status"""
        if task_id in self.processing_tasks:
            return TaskStatus.IN_PROGRESS
        elif task_id in self.completed_tasks:
            return TaskStatus.COMPLETED
        elif task_id in self.failed_tasks:
            return TaskStatus.FAILED

        task = await self.queue.get_task(task_id)
        return task.status if task else None

    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        return {
            'pending': len(self.queue._queue),
            'processing': len(self.processing_tasks),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks)
        }
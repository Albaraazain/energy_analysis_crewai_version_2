# src/tasks/definitions.py
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
import uuid

class TaskStatus(Enum):
    """Enumeration of possible tasks statuses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELEGATED = "delegated"

class TaskPriority(Enum):
    """Enumeration of tasks priorities"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskResult(BaseModel):
    """Model for tasks execution results"""
    success: bool
    data: Dict[str, Any]
    errors: Optional[List[str]] = None
    execution_time: float
    metadata: Dict[str, Any]

class TaskDefinition(BaseModel):
    """Model for tasks definitions"""
    id: str = None
    name: str
    description: str
    priority: TaskPriority
    dependencies: List[str] = []
    required_capabilities: List[str] = []
    estimated_duration: float  # in seconds
    timeout: Optional[float] = None
    retries: int = 0
    parameters: Dict[str, Any] = {}
    validation_rules: Dict[str, Any] = {}

    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            self.id = str(uuid.uuid4())

class Task:
    """Task implementation class"""

    def __init__(self, definition: TaskDefinition):
        self.definition = definition
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.assigned_agent: Optional[str] = None
        self.result: Optional[TaskResult] = None
        self.attempts = 0
        self.subtasks: List[Task] = []
        self._callbacks: Dict[str, List[Callable]] = {}

    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute the tasks"""
        try:
            self.started_at = datetime.now()
            self.status = TaskStatus.IN_PROGRESS
            await self._trigger_callbacks('on_start', {'context': context})

            # Validate input parameters
            if not self._validate_parameters(context):
                raise ValueError("Invalid tasks parameters")

            # Execute subtasks if any
            if self.subtasks:
                subtask_results = await self._execute_subtasks(context)
                context['subtask_results'] = subtask_results

            # Execute main tasks logic
            result = await self._execute_task_logic(context)

            self.completed_at = datetime.now()
            self.status = TaskStatus.COMPLETED
            self.result = result

            await self._trigger_callbacks('on_complete', {'result': result})

            return result

        except Exception as e:
            self.status = TaskStatus.FAILED
            error_result = TaskResult(
                success=False,
                data={},
                errors=[str(e)],
                execution_time=(datetime.now() - self.started_at).total_seconds(),
                metadata={'error_type': type(e).__name__}
            )
            self.result = error_result

            await self._trigger_callbacks('on_error', {'error': str(e)})

            if self.attempts < self.definition.retries:
                self.attempts += 1
                return await self.execute(context)

            return error_result

    async def _execute_task_logic(self, context: Dict[str, Any]) -> TaskResult:
        """Execute the main tasks logic"""
        raise NotImplementedError("Task logic must be implemented in subclass")

    async def _execute_subtasks(self, context: Dict[str, Any]) -> List[TaskResult]:
        """Execute subtasks in order"""
        results = []
        for subtask in self.subtasks:
            result = await subtask.execute(context)
            results.append(result)
            if not result.success:
                break
        return results

    def _validate_parameters(self, context: Dict[str, Any]) -> bool:
        """Validate tasks parameters against rules"""
        rules = self.definition.validation_rules
        for param, rule in rules.items():
            if param not in context:
                if rule.get('required', False):
                    return False
                continue

            value = context[param]
            if 'type' in rule and not isinstance(value, rule['type']):
                return False
            if 'range' in rule:
                min_val, max_val = rule['range']
                if not min_val <= value <= max_val:
                    return False
            if 'validator' in rule:
                if not rule['validator'](value):
                    return False

        return True

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add callback for specific event"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    async def _trigger_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger callbacks for an event"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                await callback(self, data)

    def add_subtask(self, subtask: 'Task') -> None:
        """Add a subtask to the tasks"""
        self.subtasks.append(subtask)

    def get_duration(self) -> Optional[float]:
        """Get actual tasks duration"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert tasks to dictionary"""
        return {
            'id': self.definition.id,
            'name': self.definition.name,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'assigned_agent': self.assigned_agent,
            'attempts': self.attempts,
            'duration': self.get_duration(),
            'result': self.result.dict() if self.result else None,
            'subtasks': [subtask.to_dict() for subtask in self.subtasks]
        }
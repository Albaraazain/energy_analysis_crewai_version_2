# src/agents/tools.py
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Model for tool execution results"""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseAgentTool:
    """Base class for agent tools"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.result_history: List[ToolResult] = []

    def store_result(self, result: ToolResult):
        """Store tool execution result"""
        self.result_history.append(result)

    def get_history(self) -> List[ToolResult]:
        """Get tool execution history"""
        return self.result_history

class AnalysisTool(BaseAgentTool):
    """Base class for analysis tools"""

    async def execute(self, data: Dict[str, Any]) -> ToolResult:
        """Execute the analysis tool"""
        try:
            result = await self._process(data)
            tool_result = ToolResult(
                success=True,
                data=result,
                metadata={'timestamp': datetime.now().isoformat()}
            )
        except Exception as e:
            tool_result = ToolResult(
                success=False,
                data={},
                error=str(e),
                metadata={'error_type': type(e).__name__}
            )

        self.store_result(tool_result)
        return tool_result

    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the data - to be implemented by specific tools"""
        raise NotImplementedError
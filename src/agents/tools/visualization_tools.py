# src/agents/tools/visualization_tools.py
from typing import Dict, Any, List
from ..tools import AnalysisTool, ToolResult
import plotly.graph_objects as go
import plotly.express as px

class ChartGeneratorTool(AnalysisTool):
    """Tool for generating various chart types"""

    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        chart_type = data.get('type', 'line')
        df = pd.DataFrame(data['data'])

        if chart_type == 'line':
            return await self._create_line_chart(df, data.get('options', {}))
        elif chart_type == 'bar':
            return await self._create_bar_chart(df, data.get('options', {}))
        elif chart_type == 'scatter':
            return await self._create_scatter_chart(df, data.get('options', {}))
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

class DashboardTool(AnalysisTool):
    """Tool for creating interactive dashboards"""

    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'layout': self._generate_dashboard_layout(data),
            'components': self._generate_dashboard_components(data),
            'interactions': self._define_dashboard_interactions(data)
        }
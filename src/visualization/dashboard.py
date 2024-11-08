# src/visualization/dashboard.py
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base import ChartConfig
from .charts import ConsumptionTrendChart, PatternAnalysisChart
from .advanced_charts import ForecastChart, ComparisonChart, HeatmapChart

class Dashboard:
    """Interactive dashboard for energy analysis"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.charts = {}
        self.layout = self._create_layout()

    def _create_layout(self) -> Dict[str, Any]:
        """Create dashboard layout configuration"""
        return {
            'grid': {
                'rows': 2,
                'columns': 2,
                'pattern': 'independent',
                'row_heights': [0.6, 0.4]
            },
            'spacing': {
                'padding': 20,
                'vertical_spacing': 0.1,
                'horizontal_spacing': 0.05
            },
            'style': {
                'background_color': '#ffffff',
                'font_family': 'Arial, sans-serif'
            }
        }

    def add_chart(self, chart_id: str, chart_type: str,
                  config: ChartConfig) -> None:
        """Add a chart to the dashboard"""
        chart_classes = {
            'consumption_trend': ConsumptionTrendChart,
            'pattern_analysis': PatternAnalysisChart,
            'forecast': ForecastChart,
            'comparison': ComparisonChart,
            'heatmap': HeatmapChart
        }

        if chart_type not in chart_classes:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        self.charts[chart_id] = chart_classes[chart_type](config)

    def create_dashboard(self, data: Dict[str, Any]) -> go.Figure:
        """Create the complete dashboard"""
        fig = make_subplots(
            rows=self.layout['grid']['rows'],
            cols=self.layout['grid']['columns'],
            subplot_titles=tuple(self.charts.keys()),
            specs=[[{'type': 'xy'} for _ in range(self.layout['grid']['columns'])]
                   for _ in range(self.layout['grid']['rows'])],
            vertical_spacing=self.layout['spacing']['vertical_spacing'],
            horizontal_spacing=self.layout['spacing']['horizontal_spacing']
        )

        # Add charts to dashboard
        for idx, (chart_id, chart) in enumerate(self.charts.items()):
            row = idx // self.layout['grid']['columns'] + 1
            col = idx % self.layout['grid']['columns'] + 1

            chart_fig = chart.create_chart(data[chart_id])
            for trace in chart_fig.data:
                fig.add_trace(trace, row=row, col=col)

        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            template='plotly_white',
            margin=dict(t=60, b=30, l=40, r=40),
            font_family=self.layout['style']['font_family'],
            paper_bgcolor=self.layout['style']['background_color']
        )

        return fig

    def update_dashboard(self, updates: Dict[str, Any]) -> None:
        """Update specific charts in the dashboard"""
        for chart_id, update_data in updates.items():
            if chart_id in self.charts:
                self.charts[chart_id].update_chart(update_data)

    def export_dashboard(self, format: str = 'html') -> str:
        """Export dashboard in specified format"""
        if format == 'html':
            return self._export_html()
        elif format == 'json':
            return self._export_json()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_html(self) -> str:
        """Export dashboard as HTML"""
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Energy Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
                .dashboard-container { width: 100%; max-width: 1600px; margin: 0 auto; }
                .chart-container { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="dashboard-container">
        """

        for chart_id, chart in self.charts.items():
            dashboard_html += f"""
                <div class="chart-container" id="{chart_id}">
                    {chart.to_html()}
                </div>
            """

        dashboard_html += """
            </div>
        </body>
        </html>
        """

        return dashboard_html
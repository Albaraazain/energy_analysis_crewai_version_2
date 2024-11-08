# src/visualization/export.py
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import base64
import json
from datetime import datetime
from pathlib import Path

class ExportManager:
    """Manages export functionality for visualizations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.export_path = Path(config.get('export_path', 'exports'))
        self.export_path.mkdir(parents=True, exist_ok=True)

    async def export_visualization(self, figure: go.Figure,
                                   format: str,
                                   filename: Optional[str] = None) -> str:
        """Export visualization in specified format"""
        if filename is None:
            filename = f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if format == 'html':
            return await self._export_html(figure, filename)
        elif format == 'png':
            return await self._export_png(figure, filename)
        elif format == 'svg':
            return await self._export_svg(figure, filename)
        elif format == 'json':
            return await self._export_json(figure, filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def export_dashboard(self, dashboard: Dict[str, go.Figure],
                               format: str) -> str:
        """Export complete dashboard"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == 'html':
            return await self._export_dashboard_html(dashboard, timestamp)
        elif format == 'pdf':
            return await self._export_dashboard_pdf(dashboard, timestamp)
        else:
            raise ValueError(f"Unsupported dashboard export format: {format}")

    async def _export_html(self, figure: go.Figure, filename: str) -> str:
        """Export as HTML file"""
        filepath = self.export_path / f"{filename}.html"

        html_content = self._generate_html_template(
            figure.to_json(),
            self.config.get('template', 'default')
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(filepath)

    def _generate_html_template(self, figure_json: str,
                                template: str) -> str:
        """Generate HTML content with specified template"""
        templates = {
            'default': """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Energy Analysis Visualization</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <style>
                        body { margin: 0; padding: 20px; }
                        #visualization { width: 100%; height: 100%; }
                    </style>
                </head>
                <body>
                    <div id="visualization"></div>
                    <script>
                        var figure = {figure_json};
                        Plotly.newPlot('visualization', figure.data, figure.layout, figure.config);
                    </script>
                </body>
                </html>
            """,
            'interactive': """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Interactive Energy Analysis</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <script src="https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.min.js"></script>
                    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
                    <style>
                        .chart-container { position: relative; }
                        .controls { position: absolute; top: 10px; right: 10px; }
                    </style>
                </head>
                <body class="bg-gray-100">
                    <div class="container mx-auto px-4 py-8">
                        <div class="chart-container bg-white rounded-lg shadow-lg p-6">
                            <div id="visualization"></div>
                            <div class="controls">
                                {controls}
                            </div>
                        </div>
                    </div>
                    <script>
                        var figure = {figure_json};
                        var chart = Plotly.newPlot('visualization', figure.data, figure.layout, figure.config);
                        {interactive_js}
                    </script>
                </body>
                </html>
            """
        }

        return templates.get(template, templates['default']).format(
            figure_json=figure_json,
            controls=self._generate_controls(),
            interactive_js=self._generate_interactive_js()
        )

    def _generate_controls(self) -> str:
        """Generate HTML for interactive controls"""
        return """
            <div class="flex space-x-4">
                <button class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                        onclick="updateTimeRange('1w')">1W</button>
                <button class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                        onclick="updateTimeRange('1m')">1M</button>
                <button class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                        onclick="updateTimeRange('3m')">3M</button>
                <button class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                        onclick="updateTimeRange('1y')">1Y</button>
            </div>
        """

    def _generate_interactive_js(self) -> str:
        """Generate JavaScript for interactive features"""
        return """
            function updateTimeRange(range) {
                var now = new Date();
                var start = new Date();
                
                switch(range) {
                    case '1w': start.setDate(start.getDate() - 7); break;
                    case '1m': start.setMonth(start.getMonth() - 1); break;
                    case '3m': start.setMonth(start.getMonth() - 3); break;
                    case '1y': start.setFullYear(start.getFullYear() - 1); break;
                }
                
                Plotly.relayout('visualization', {
                    'xaxis.range': [start, now]
                });
            }
            
            // Add event listeners
            document.getElementById('visualization').on('plotly_click', function(data) {
                console.log('Selected point:', data.points[0]);
            });
        """
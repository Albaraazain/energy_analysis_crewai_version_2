# src/agents/tools/integration_helper.py
class VisualizationIntegration:
    """Helper class for integrating visualization with external systems"""

    @staticmethod
    def prepare_for_export(fig: go.Figure, format: str) -> bytes:
        """Prepare visualization for export in specified format"""
        if format == 'html':
            return fig.to_html().encode('utf-8')
        elif format == 'png':
            return fig.to_image(format='png')
        elif format == 'svg':
            return fig.to_image(format='svg')
        else:
            raise ValueError(f"Unsupported export format: {format}")

    @staticmethod
    def create_dashboard_template(visualizations: Dict[str, Any]) -> str:
        """Create HTML template for dashboard"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Energy Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                .dashboard-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .chart-container { padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="dashboard-container">
        """

        for name, viz in visualizations.items():
            template += f"""
                <div class="chart-container" id="{name}"></div>
            """

        template += """
            </div>
            <script>
        """

        # Add visualization rendering code
        for name, viz in visualizations.items():
            template += f"""
                Plotly.newPlot('{name}', {viz});
            """

        template += """
            </script>
        </body>
        </html>
        """

        return template
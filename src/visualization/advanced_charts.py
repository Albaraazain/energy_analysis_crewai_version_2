# src/visualization/advanced_charts.py
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from .base import BaseChart, ChartConfig

class ForecastChart(BaseChart):
    """Advanced forecasting visualization with confidence intervals"""

    def create_chart(self, data: Dict[str, Any]) -> go.Figure:
        df = pd.DataFrame(data['forecast_data'])

        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=df.index[:-data['forecast_horizon']],
            y=df['actual'],
            name='Historical',
            mode='lines',
            line=dict(color='#1f77b4', width=2)
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=df.index[-data['forecast_horizon']:],
            y=df['forecast'],
            name='Forecast',
            mode='lines',
            line=dict(color='#2ca02c', width=2)
        ))

        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=df.index[-data['forecast_horizon']:].tolist() +
              df.index[-data['forecast_horizon']:].tolist()[::-1],
            y=df['upper_bound'][-data['forecast_horizon']:].tolist() +
              df['lower_bound'][-data['forecast_horizon']:].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(44, 160, 44, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))

        # Add model performance metrics
        metrics = data.get('metrics', {})
        if metrics:
            annotations = [
                dict(
                    x=0.02,
                    y=0.98,
                    xref='paper',
                    yref='paper',
                    text=(f"MAPE: {metrics.get('mape', 0):.2f}%<br>"
                          f"RMSE: {metrics.get('rmse', 0):.2f}"),
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.3)',
                    borderwidth=1
                )
            ]
            fig.update_layout(annotations=annotations)

        self.figure = self.apply_layout(fig)
        return self.figure

    def _update_data(self, data: Dict[str, Any]) -> None:
        """Update forecast chart data"""
        with self.figure.batch_update():
            # Update historical data
            self.figure.data[0].x = data['dates'][:-data['forecast_horizon']]
            self.figure.data[0].y = data['actual'][:-data['forecast_horizon']]

            # Update forecast
            self.figure.data[1].x = data['dates'][-data['forecast_horizon']:]
            self.figure.data[1].y = data['forecast']

            # Update confidence intervals
            self.figure.data[2].x = (
                    data['dates'][-data['forecast_horizon']:].tolist() +
                    data['dates'][-data['forecast_horizon']:].tolist()[::-1]
            )
            self.figure.data[2].y = (
                    data['upper_bound'][-data['forecast_horizon']:].tolist() +
                    data['lower_bound'][-data['forecast_horizon']:].tolist()[::-1]
            )

class ComparisonChart(BaseChart):
    """Advanced comparison visualization"""

    def create_chart(self, data: Dict[str, Any]) -> go.Figure:
        df = pd.DataFrame(data['comparison_data'])

        fig = go.Figure()

        # Add actual vs expected comparison
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['actual'],
            name='Actual',
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(color='#1f77b4')
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['expected'],
            name='Expected',
            mode='lines',
            line=dict(color='#ff7f0e', dash='dash')
        ))

        # Add deviation areas
        deviation = df['actual'] - df['expected']
        positive_deviation = deviation.copy()
        negative_deviation = deviation.copy()
        positive_deviation[positive_deviation < 0] = np.nan
        negative_deviation[negative_deviation > 0] = np.nan

        fig.add_trace(go.Bar(
            x=df.index,
            y=positive_deviation,
            name='Above Expected',
            marker_color='rgba(44, 160, 44, 0.3)',
            showlegend=True
        ))

        fig.add_trace(go.Bar(
            x=df.index,
            y=negative_deviation,
            name='Below Expected',
            marker_color='rgba(214, 39, 40, 0.3)',
            showlegend=True
        ))

        # Add trend lines if available
        if 'trend' in data:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=data['trend'],
                name='Trend',
                mode='lines',
                line=dict(color='#9467bd', dash='dot'),
                showlegend=True
            ))

        self.figure = self.apply_layout(fig)
        return self.figure

class HeatmapChart(BaseChart):
    """Advanced heatmap visualization"""

    def create_chart(self, data: Dict[str, Any]) -> go.Figure:
        df = pd.DataFrame(data['heatmap_data'])

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale='RdYlBu_r',
            showscale=True,
            hoverongaps=False,
            hovertemplate=(
                'Hour: %{y}<br>'
                'Day: %{x}<br>'
                'Consumption: %{z:.2f} kWh<br>'
                '<extra></extra>'
            )
        ))

        # Update layout for better visualization
        fig.update_layout(
            xaxis=dict(
                tickangle=-45,
                title='Day of Week',
                side='bottom'
            ),
            yaxis=dict(
                title='Hour of Day',
                autorange='reversed'
            )
        )

        self.figure = self.apply_layout(fig)
        return self.figure
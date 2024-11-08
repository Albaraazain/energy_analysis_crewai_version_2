# src/visualization/charts.py
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from .base import BaseChart, ChartConfig

class ConsumptionTrendChart(BaseChart):
    """Visualization for consumption trends"""

    def create_chart(self, data: Dict[str, Any]) -> go.Figure:
        df = pd.DataFrame(data['consumption_data'])

        # Create main trend line
        fig = go.Figure()

        # Add consumption line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['consumption'],
            mode='lines+markers',
            name='Consumption',
            line=dict(color='#1f77b4', width=2),
            hovertemplate=(
                '%{x}<br>'
                'Consumption: %{y:.2f} kWh<br>'
                '<extra></extra>'
            )
        ))

        # Add moving average
        ma = df['consumption'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=ma,
            mode='lines',
            name='7-Day Moving Average',
            line=dict(color='#ff7f0e', dash='dash'),
            hovertemplate=(
                '%{x}<br>'
                'MA: %{y:.2f} kWh<br>'
                '<extra></extra>'
            )
        ))

        # Add predicted values if available
        if 'predictions' in data:
            pred_df = pd.DataFrame(data['predictions'])
            fig.add_trace(go.Scatter(
                x=pred_df.index,
                y=pred_df['predicted'],
                mode='lines',
                name='Predicted',
                line=dict(color='#2ca02c', dash='dot'),
                hovertemplate=(
                    '%{x}<br>'
                    'Predicted: %{y:.2f} kWh<br>'
                    '<extra></extra>'
                )
            ))

        self.figure = self.apply_layout(fig)
        return self.figure

    def _update_data(self, data: Dict[str, Any]) -> None:
        """Update trend chart data"""
        with self.figure.batch_update():
            # Update consumption data
            self.figure.data[0].x = data['dates']
            self.figure.data[0].y = data['consumption']

            # Update moving average
            ma = pd.Series(data['consumption']).rolling(window=7).mean()
            self.figure.data[1].x = data['dates']
            self.figure.data[1].y = ma

            # Update predictions if present
            if len(self.figure.data) > 2 and 'predictions' in data:
                self.figure.data[2].x = data['prediction_dates']
                self.figure.data[2].y = data['predictions']

class PatternAnalysisChart(BaseChart):
    """Visualization for pattern analysis"""

    def create_chart(self, data: Dict[str, Any]) -> go.Figure:
        # Create subplots for different pattern aspects
        fig = go.Figure()

        # Daily patterns
        daily_patterns = self._create_daily_pattern_trace(data)
        fig.add_trace(daily_patterns)

        # Weekly patterns
        weekly_patterns = self._create_weekly_pattern_trace(data)
        fig.add_trace(weekly_patterns)

        # Seasonal patterns
        seasonal_patterns = self._create_seasonal_pattern_trace(data)
        fig.add_trace(seasonal_patterns)

        # Add anomaly markers if present
        if 'anomalies' in data:
            anomaly_markers = self._create_anomaly_markers(data['anomalies'])
            fig.add_trace(anomaly_markers)

        self.figure = self.apply_layout(fig)
        return self.figure

    def _create_daily_pattern_trace(self, data: Dict[str, Any]) -> go.Scatter:
        """Create trace for daily patterns"""
        df = pd.DataFrame(data['daily_patterns'])
        return go.Scatter(
            x=df.index,
            y=df['consumption'],
            name='Daily Pattern',
            mode='lines',
            line=dict(color='#1f77b4'),
            hovertemplate=(
                'Hour: %{x}<br>'
                'Avg Consumption: %{y:.2f} kWh<br>'
                '<extra></extra>'
            )
        )

    def _create_weekly_pattern_trace(self, data: Dict[str, Any]) -> go.Scatter:
        """Create trace for weekly patterns"""
        df = pd.DataFrame(data['weekly_patterns'])
        return go.Scatter(
            x=df.index,
            y=df['consumption'],
            name='Weekly Pattern',
            mode='lines+markers',
            line=dict(color='#ff7f0e'),
            visible='legendonly',
            hovertemplate=(
                'Day: %{x}<br>'
                'Avg Consumption: %{y:.2f} kWh<br>'
                '<extra></extra>'
            )
        )

    def _create_seasonal_pattern_trace(self, data: Dict[str, Any]) -> go.Scatter:
        """Create trace for seasonal patterns"""
        df = pd.DataFrame(data['seasonal_patterns'])
        return go.Scatter(
            x=df.index,
            y=df['consumption'],
            name='Seasonal Pattern',
            mode='lines+markers',
            line=dict(color='#2ca02c'),
            visible='legendonly',
            hovertemplate=(
                'Month: %{x}<br>'
                'Avg Consumption: %{y:.2f} kWh<br>'
                '<extra></extra>'
            )
        )

    def _create_anomaly_markers(self, anomalies: List[Dict[str, Any]]) -> go.Scatter:
        """Create trace for anomaly markers"""
        return go.Scatter(
            x=[a['timestamp'] for a in anomalies],
            y=[a['value'] for a in anomalies],
            mode='markers',
            name='Anomalies',
            marker=dict(
                size=10,
                symbol='circle',
                color='red',
                line=dict(width=2, color='red')
            ),
            hovertemplate=(
                '%{x}<br>'
                'Value: %{y:.2f} kWh<br>'
                'Type: %{customdata[0]}<br>'
                'Score: %{customdata[1]:.2f}<br>'
                '<extra></extra>'
            ),
            customdata=[[a['type'], a['score']] for a in anomalies]
        )
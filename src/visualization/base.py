# src/visualization/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from pydantic import BaseModel

class ChartConfig(BaseModel):
    """Model for chart configuration"""
    title: str
    type: str
    x_label: str
    y_label: str
    theme: str = "plotly_white"
    height: int = 600
    width: int = 800
    interactive: bool = True
    legend_position: str = "top"
    color_scheme: Optional[str] = None
    annotations: Optional[List[Dict[str, Any]]] = None

class BaseChart(ABC):
    """Abstract base class for visualization components"""

    def __init__(self, config: ChartConfig):
        self.config = config
        self.figure: Optional[go.Figure] = None

    @abstractmethod
    def create_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create the visualization"""
        pass

    def apply_layout(self, fig: go.Figure) -> go.Figure:
        """Apply standard layout configuration"""
        fig.update_layout(
            title=self.config.title,
            xaxis_title=self.config.x_label,
            yaxis_title=self.config.y_label,
            template=self.config.theme,
            height=self.config.height,
            width=self.config.width,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        # Add annotations if provided
        if self.config.annotations:
            fig.update_layout(annotations=self.config.annotations)

        return fig

    def to_html(self) -> str:
        """Convert chart to HTML"""
        if not self.figure:
            raise ValueError("Chart has not been created yet")
        return self.figure.to_html(
            include_plotlyjs=True,
            full_html=True
        )

    def to_json(self) -> str:
        """Convert chart to JSON"""
        if not self.figure:
            raise ValueError("Chart has not been created yet")
        return self.figure.to_json()

    def update_chart(self, updates: Dict[str, Any]) -> None:
        """Update existing chart with new data or configuration"""
        if not self.figure:
            raise ValueError("Chart has not been created yet")

        if 'data' in updates:
            self._update_data(updates['data'])
        if 'layout' in updates:
            self._update_layout(updates['layout'])
        if 'config' in updates:
            self._update_config(updates['config'])

    def _update_data(self, data: Dict[str, Any]) -> None:
        """Update chart data"""
        raise NotImplementedError

    def _update_layout(self, layout: Dict[str, Any]) -> None:
        """Update chart layout"""
        self.figure.update_layout(**layout)

    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update chart configuration"""
        self.config = ChartConfig(**{**self.config.dict(), **config})
        self.apply_layout(self.figure)
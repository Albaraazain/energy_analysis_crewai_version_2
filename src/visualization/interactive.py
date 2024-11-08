# src/visualization/interactive.py
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import plotly.graph_objects as go
import json

class InteractionManager:
    """Manages interactive features and event handling"""

    def __init__(self):
        self.callbacks: Dict[str, List[Callable]] = {}
        self.filters: Dict[str, Dict[str, Any]] = {}
        self.selected_data: Dict[str, Any] = {}

    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for specific event type"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger callbacks for an event"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                callback(event_data)

    def add_filter(self, filter_id: str, filter_config: Dict[str, Any]) -> None:
        """Add data filter configuration"""
        self.filters[filter_id] = filter_config

    def apply_filters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply active filters to data"""
        filtered_data = data.copy()
        for filter_id, filter_config in self.filters.items():
            if filter_config.get('active', True):
                filtered_data = self._apply_single_filter(
                    filtered_data, filter_config
                )
        return filtered_data

    def _apply_single_filter(self, data: Dict[str, Any],
                             filter_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply single filter to data"""
        filter_type = filter_config['type']
        if filter_type == 'date_range':
            return self._apply_date_filter(data, filter_config)
        elif filter_type == 'value_range':
            return self._apply_value_filter(data, filter_config)
        elif filter_type == 'category':
            return self._apply_category_filter(data, filter_config)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    def update_selection(self, selection_data: Dict[str, Any]) -> None:
        """Update selected data points"""
        self.selected_data = selection_data
        self.trigger_event('selection_changed', selection_data)

    def get_filter_state(self) -> Dict[str, Any]:
        """Get current state of all filters"""
        return {
            filter_id: {
                'config': filter_config,
                'active': filter_config.get('active', True)
            }
            for filter_id, filter_config in self.filters.items()
        }
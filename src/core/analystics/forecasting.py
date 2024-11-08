# src/core/analytics/forecasting.py
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from .base import BaseAnalyzer, AnalysisResult

class ForecastAnalyzer(BaseAnalyzer):
    """Advanced forecasting component with multiple models"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.forecast_horizon = config.get('forecast_horizon', 30)
        self.models = {
            'exponential_smoothing': self._fit_exponential_smoothing,
            'arima': self._fit_arima,
            'prophet': self._fit_prophet
        }

    async def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        """Perform comprehensive forecasting analysis"""
        try:
            # Split data for validation
            train_data, test_data = self._prepare_data(data)

            # Generate forecasts using multiple models
            forecasts = {}
            for model_name, fit_func in self.models.items():
                forecasts[model_name] = await fit_func(train_data, test_data)

            # Evaluate and select best model
            best_model = self._select_best_model(forecasts)

            # Generate final forecast
            final_forecast = self._generate_final_forecast(
                data, best_model, self.forecast_horizon
            )

            # Calculate confidence intervals
            intervals = self._calculate_confidence_intervals(
                final_forecast, best_model
            )

            # Calculate metrics
            metrics = self._calculate_forecast_metrics(forecasts)

            # Generate insights
            insights = self._generate_forecast_insights(
                final_forecast, intervals, metrics
            )

            return AnalysisResult(
                timestamp=datetime.now(),
                metrics=metrics,
                patterns=[{
                    'type': 'forecast',
                    'values': final_forecast.tolist(),
                    'intervals': intervals,
                    'model': best_model
                }],
                insights=insights,
                confidence=await self._calculate_confidence(metrics, len(data)),
                metadata={
                    'analysis_type': 'forecasting',
                    'forecast_horizon': self.forecast_horizon,
                    'best_model': best_model
                }
            )
        except Exception as e:
            print(f"Error in forecasting analysis: {str(e)}")
            raise

    async def _fit_exponential_smoothing(self, train_data: pd.DataFrame,
                                         test_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit Exponential Smoothing model"""
        try:
            # Fit model
            model = ExponentialSmoothing(
                train_data['consumption'],
                seasonal_periods=12,
                trend='add',
                seasonal='add'
            ).fit()

            # Generate forecast
            forecast = model.forecast(len(test_data))

            # Calculate error metrics
            metrics = self._calculate_error_metrics(
                test_data['consumption'], forecast
            )

            return {
                'model': model,
                'forecast': forecast,
                'metrics': metrics
            }
        except Exception as e:
            print(f"Error fitting exponential smoothing: {str(e)}")
            return None

    async def _fit_arima(self, train_data: pd.DataFrame,
                         test_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit ARIMA model"""
        try:
            # Determine order using auto_arima
            from pmdarima import auto_arima
            auto_model = auto_arima(
                train_data['consumption'],
                seasonal=True,
                m=12,
                suppress_warnings=True
            )

            # Fit ARIMA model with optimal parameters
            model = ARIMA(
                train_data['consumption'],
                order=auto_model.order
            ).fit()

            # Generate forecast
            forecast = model.forecast(len(test_data))

            # Calculate error metrics
            metrics = self._calculate_error_metrics(
                test_data['consumption'], forecast
            )

            return {
                'model': model,
                'forecast': forecast,
                'metrics': metrics
            }
        except Exception as e:
            print(f"Error fitting ARIMA: {str(e)}")
            return None

    async def _fit_prophet(self, train_data: pd.DataFrame,
                           test_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit Prophet model"""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data['consumption']
            })

            # Fit model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(prophet_data)

            # Generate forecast
            future = pd.DataFrame({'ds': test_data.index})
            forecast = model.predict(future)

            # Calculate error metrics
            metrics = self._calculate_error_metrics(
                test_data['consumption'], forecast['yhat']
            )

            return {
                'model': model,
                'forecast': forecast['yhat'],
                'metrics': metrics
            }
        except Exception as e:
            print(f"Error fitting Prophet: {str(e)}")
            return None
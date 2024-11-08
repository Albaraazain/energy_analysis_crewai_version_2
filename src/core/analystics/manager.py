# src/core/analytics/manager.py
from typing import Dict, Any, List, Optional
from .statistical import StatisticalAnalyzer
from .patterns import PatternAnalyzer
from .timeseries import TimeSeriesAnalyzer
from .forecasting import ForecastAnalyzer
from .base import AnalysisResult

class AnalyticsManager:
    """Manages and coordinates all analytics components"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.statistical = StatisticalAnalyzer(config)
        self.patterns = PatternAnalyzer(config)
        self.timeseries = TimeSeriesAnalyzer(config)
        self.forecasting = ForecastAnalyzer(config)

    async def run_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive analysis using all components"""
        try:
            # Run all analyses
            results = await self._run_all_analyses(data)

            # Combine results
            combined_results = self._combine_results(results)

            # Generate final insights
            insights = self._generate_final_insights(results)

            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(results)

            return {
                'results': combined_results,
                'insights': insights,
                'confidence': confidence,
                'memory_metadata': {
                    'components': list(results.keys()),
                    'timestamp': datetime.now().isoformat()
                }
            }
        except Exception as e:
            print(f"Error in analytics manager: {str(e)}")
            raise

    async def _run_all_analyses(self,
                                data: pd.DataFrame) -> Dict[str, AnalysisResult]:
        """Run all analysis components"""
        results = {}

        # Run components concurrently
        analyses = [
            ('statistical', self.statistical.analyze(data)),
            ('patterns', self.patterns.analyze(data)),
            ('timeseries', self.timeseries.analyze(data)),
            ('forecasting', self.forecasting.analyze(data))
        ]

        for name, analysis in analyses:
            try:
                results[name] = await analysis
            except Exception as e:
                print(f"Error in {name} analysis: {str(e)}")
                results[name] = None

        return results

    def _combine_results(self,
                         results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Combine results from all components"""
        combined = {
            'metrics': {},
            'patterns': [],
            'insights': []
        }

        for component, result in results.items():
            if result:
                combined['metrics'][component] = result.metrics
                combined['patterns'].extend(result.patterns)
                combined['insights'].extend(result.insights)

        return combined

    def _generate_final_insights(self,
                                 results: Dict[str, AnalysisResult]) -> List[Dict[str, Any]]:
        """Generate final insights from all analyses"""
        all_insights = []

        # Collect all insights
        for component, result in results.items():
            if result and result.insights:
                for insight in result.insights:
                    insight['source'] = component
                    all_insights.append(insight)

        # Sort by importance/confidence
        all_insights.sort(
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )

        return all_insights

    def _calculate_overall_confidence(self,
                                      results: Dict[str, AnalysisResult]) -> float:
        """Calculate overall confidence score"""
        confidences = [
            result.confidence for result in results.values()
            if result is not None
        ]

        if not confidences:
            return 0.0

        # Weight more recent results higher
        weights = np.linspace(0.5, 1.0, len(confidences))
        return float(np.average(confidences, weights=weights))
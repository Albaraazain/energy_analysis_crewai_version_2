# src/core/analyzer.py
from typing import Dict, Any, Optional
from .memory.manager import MemoryManager
from .types import AnalysisResult

class EnhancedEnergyAnalyzer:
    """Enhanced analyzer with memory integration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.process = self._initialize_process()
        self.initialized = False

    async def analyze(self, data: Dict[str, float]) -> AnalysisResult:
        """Perform analysis with memory context"""
        try:
            # Get relevant historical context
            context = await self._get_analysis_context(data)

            # Perform analysis with context
            analysis_result = await self.process.execute({
                'data': data,
                'context': context
            })

            # Store analysis results in memory
            await self._store_analysis_results(analysis_result)

            return analysis_result
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            raise

    async def _get_analysis_context(self,
                                    data: Dict[str, float]) -> Dict[str, Any]:
        """Get relevant context for analysis"""
        query = self._generate_context_query(data)
        context = await self.memory_manager.get_relevant_context(query)

        return {
            'historical_patterns': self._extract_patterns(context),
            'known_entities': await self._get_relevant_entities(data),
            'previous_recommendations': self._extract_recommendations(context)
        }

    async def _store_analysis_results(self,
                                      results: AnalysisResult):
        """Store analysis results in memory"""
        # Store overall results
        await self.memory_manager.store_memory(
            content=results.dict(),
            source='analysis',
            tags=['analysis_result']
        )

        # Store patterns separately
        if patterns := results.get('patterns'):
            await self.memory_manager.store_memory(
                content={'patterns': patterns},
                source='pattern_analysis',
                tags=['pattern', 'insight']
            )

        # Store entities
        if entities := results.get('entities'):
            for entity in entities:
                await self.memory_manager.entity.store_entity(
                    entity['id'],
                    entity['attributes']
                )

    def _generate_context_query(self, data: Dict[str, float]) -> str:
        """Generate context query from current data"""
        return f"""
        Find relevant information for energy analysis with characteristics:
        - Date range: {min(data.keys())} to {max(data.keys())}
        - Consumption range: {min(data.values()):.2f} to {max(data.values()):.2f} kWh
        """

    async def _get_relevant_entities(self,
                                     data: Dict[str, float]) -> Dict[str, Any]:
        """Get relevant entities for the analysis"""
        entities = {}

        # Get location entity if it exists
        if location := await self.memory_manager.entity.retrieve_entity('location'):
            entities['location'] = location

        # Get relevant device entities
        device_query = "Find devices affecting energy consumption"
        device_results = await self.memory_manager.entity.find_similar_entities(
            device_query
        )
        if device_results:
            entities['devices'] = device_results

        return entities
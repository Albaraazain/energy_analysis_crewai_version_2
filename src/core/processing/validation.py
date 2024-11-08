# src/core/processing/validation.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from pydantic import BaseModel, validator
from .base import DataProcessor, ProcessingResult

class EnergyDataSchema(BaseModel):
    """Schema for energy consumption data validation"""
    timestamp: datetime
    consumption: float

    @validator('consumption')
    def consumption_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Consumption must be positive')
        return v

class DataValidator(DataProcessor):
    """Advanced data validation with customizable rules"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validation_rules = self._setup_validation_rules()
        self.schema = EnergyDataSchema

    def process(self, data: Dict[str, Any]) -> ProcessingResult:
        """Perform comprehensive data validation"""
        try:
            # Convert to DataFrame if necessary
            df = pd.DataFrame(data) if isinstance(data, dict) else data

            # Apply all validation rules
            validation_results = self._apply_validation_rules(df)

            # Check schema compliance
            schema_validation = self._validate_schema(df)

            # Combine results
            all_results = {
                'rule_validation': validation_results,
                'schema_validation': schema_validation
            }

            # Calculate overall validation status
            validation_status = all(
                result.get('passed', False)
                for result in all_results.values()
            )

            return ProcessingResult(
                timestamp=datetime.now(),
                data=all_results,
                metadata=self._generate_validation_metadata(all_results),
                validation_status=validation_status,
                errors=self._collect_validation_errors(all_results)
            )
        except Exception as e:
            return ProcessingResult(
                timestamp=datetime.now(),
                data={},
                metadata={'error': str(e)},
                validation_status=False,
                errors=[str(e)]
            )

    def _apply_validation_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Apply all validation rules to the data"""
        results = {}
        for rule_name, rule_func in self.validation_rules.items():
            results[rule_name] = rule_func(df)
        return results

    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data against schema"""
        validation_errors = []

        try:
            for _, row in df.iterrows():
                try:
                    self.schema(
                        timestamp=row.get('timestamp', row.name),
                        consumption=row['consumption']
                    )
                except Exception as e:
                    validation_errors.append({
                        'row': row.name,
                        'error': str(e)
                    })

            return {
                'passed': len(validation_errors) == 0,
                'errors': validation_errors
            }
        except Exception as e:
            return {
                'passed': False,
                'errors': [str(e)]
            }

    def _generate_validation_metadata(self,
                                      results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate memory_metadata about validation results"""
        return {
            'validation_timestamp': datetime.now().isoformat(),
            'rules_passed': sum(1 for r in results['rule_validation'].values()
                                if r.get('passed', False)),
            'rules_failed': sum(1 for r in results['rule_validation'].values()
                                if not r.get('passed', False)),
            'schema_validation_passed': results['schema_validation']['passed']
        }

    def _collect_validation_errors(self,
                                   results: Dict[str, Any]) -> List[str]:
        """Collect all validation errors"""
        errors = []

        # Collect rule validation errors
        for rule_name, rule_result in results['rule_validation'].items():
            if not rule_result.get('passed', False):
                errors.extend(rule_result.get('errors', []))

        # Collect schema validation errors
        errors.extend([
            str(error) for error in results['schema_validation'].get('errors', [])
        ])

        return errors
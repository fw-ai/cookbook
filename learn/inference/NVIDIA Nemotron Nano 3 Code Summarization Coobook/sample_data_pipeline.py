"""
Data Pipeline Toolkit

A lightweight data processing library for transforming, validating,
and exporting tabular data with support for custom transformations.
"""

import json
import csv
from datetime import datetime
from typing import Callable, Any
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.is_valid


class DataTransformer:
    """
    Chainable data transformer for processing records.
    
    Supports registering custom transformation functions and
    applying them in sequence to data records.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._transforms: list[Callable[[dict], dict]] = []
        self._validators: list[Callable[[dict], ValidationResult]] = []
    
    def add_transform(self, func: Callable[[dict], dict]) -> "DataTransformer":
        """Register a transformation function. Returns self for chaining."""
        self._transforms.append(func)
        return self
    
    def add_validator(self, func: Callable[[dict], ValidationResult]) -> "DataTransformer":
        """Register a validation function. Returns self for chaining."""
        self._validators.append(func)
        return self
    
    def transform(self, record: dict) -> dict:
        """Apply all registered transforms to a record."""
        result = record.copy()
        for func in self._transforms:
            result = func(result)
        return result
    
    def validate(self, record: dict) -> ValidationResult:
        """Run all validators and aggregate results."""
        all_errors = []
        all_warnings = []
        
        for validator in self._validators:
            result = validator(record)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )
    
    def process(self, records: list[dict]) -> tuple[list[dict], list[ValidationResult]]:
        """Transform and validate a batch of records."""
        transformed = []
        validations = []
        
        for record in records:
            t_record = self.transform(record)
            v_result = self.validate(t_record)
            transformed.append(t_record)
            validations.append(v_result)
        
        return transformed, validations


def normalize_string(value: str) -> str:
    """Normalize a string by stripping whitespace and lowercasing."""
    return value.strip().lower() if isinstance(value, str) else value


def parse_date(date_str: str, formats: list[str] = None) -> datetime | None:
    """Parse a date string trying multiple formats."""
    formats = formats or ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def create_field_validator(field: str, required: bool = True, 
                           field_type: type = None) -> Callable[[dict], ValidationResult]:
    """Factory function to create field validators."""
    def validator(record: dict) -> ValidationResult:
        errors = []
        warnings = []
        
        if field not in record:
            if required:
                errors.append(f"Missing required field: {field}")
            return ValidationResult(len(errors) == 0, errors, warnings)
        
        value = record[field]
        
        if required and (value is None or value == ""):
            errors.append(f"Field '{field}' cannot be empty")
        
        if field_type and value is not None and not isinstance(value, field_type):
            errors.append(f"Field '{field}' must be of type {field_type.__name__}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    return validator


def export_to_csv(records: list[dict], filepath: str) -> int:
    """Export records to CSV file. Returns number of records written."""
    if not records:
        return 0
    
    fieldnames = list(records[0].keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    return len(records)


def export_to_json(records: list[dict], filepath: str, indent: int = 2) -> int:
    """Export records to JSON file. Returns number of records written."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=indent, default=str)
    return len(records)


if __name__ == "__main__":
    # Example usage
    sample_data = [
        {"name": "  Alice  ", "email": "alice@example.com", "age": 30},
        {"name": "Bob", "email": "", "age": 25},
        {"name": "Charlie", "email": "charlie@test.com", "age": "invalid"},
    ]
    
    # Create transformer with chained operations
    pipeline = DataTransformer("user_pipeline")
    pipeline.add_transform(lambda r: {**r, "name": normalize_string(r.get("name", ""))})
    pipeline.add_validator(create_field_validator("email", required=True))
    pipeline.add_validator(create_field_validator("age", required=True, field_type=int))
    
    # Process data
    results, validations = pipeline.process(sample_data)
    
    # Report results
    for i, (record, validation) in enumerate(zip(results, validations)):
        status = "✓" if validation.is_valid else "✗"
        print(f"{status} Record {i}: {record['name']}")
        for error in validation.errors:
            print(f"    Error: {error}")

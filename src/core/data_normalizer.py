"""
Core data normalization abstraction for the MeshInsights pipeline.

This module defines the `DataNormalizer` abstract base class used to transform
raw datasets into a standardized, processor-ready format. Normalizers are
expected to be pure (no side effects or external I/O) and focus solely on
data typing, schema alignment, timezone handling, and other deterministic
transformations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import logging
import pandas as pd
import pandera.pandas as pa


class DataNormalizer(ABC):
    """
    Abstract base class for transforming raw source data into a standardized
    schema required by processors in the MeshInsights pipeline.

    Public API:
    - normalize(data, metadata) -> pandas.DataFrame: Orchestrates the
      normalization of a single source dataset.
    - get_schema() -> Optional[pa.DataFrameSchema]: Returns optional Pandera schema
      for the normalized output.
    - transform(data, metadata) -> pandas.DataFrame: Pure transformation logic
      (to be implemented by subclasses).

    Notes:
    - Normalizers must be pure and deterministic (no external I/O, no global
      state modification). They should focus on typing, schema alignment,
      timezone handling, and other data-shaping concerns.
    """

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a normalizer.

        Args:
            name (Optional[str]): Identifier for this normalizer. Defaults to
                the class name.
            config (Optional[Dict[str, Any]]): Optional configuration affecting
                normalization behavior.
        """
        self.name: str = name or self.__class__.__name__
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"normalizer.{self.name}")

    @abstractmethod
    def transform(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Pure transformation logic to convert raw data to normalized format.
        
        Args:
            data (pandas.DataFrame): Raw input for a single source.
            metadata (Optional[Dict[str, Any]]): Optional non-sensitive metadata
                from the originating data source.
                
        Returns:
            pandas.DataFrame: Transformed DataFrame (not yet validated).
        """
        raise NotImplementedError
    
    def get_schema(self) -> Optional[pa.DataFrameSchema]:
        """
        Return optional Pandera schema for the normalized output.
        
        If provided, this schema will be used to validate the normalized output.
        
        Returns:
            Optional[pa.DataFrameSchema]: Schema for validation, or None to skip validation
        """
        return None
    
    def normalize(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Normalize raw data to a standard format suitable for processors.
        
        This orchestrates:
        1. Input validation (basic checks)
        2. Transformation
        3. Output validation (if schema provided)
        
        Args:
            data (pandas.DataFrame): Raw input for a single source.
            metadata (Optional[Dict[str, Any]]): Optional non-sensitive metadata
                from the originating data source.

        Returns:
            pandas.DataFrame: Normalized and optionally validated DataFrame.
        """
        # Basic input validation
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"{self.name}: Expected DataFrame, got {type(data).__name__}")
        
        if data.empty:
            self.logger.warning(f"{self.name}: Input DataFrame is empty")
        
        # Transform
        transformed = self.transform(data, metadata)
        
        # Validate output if schema provided
        schema = self.get_schema()
        if schema is not None:
            try:
                validated = schema.validate(transformed, lazy=True)
                self.logger.debug(f"{self.name}: Schema validation passed")
                return validated
            except pa.errors.SchemaError as e:
                self.logger.warning(f"{self.name}: Schema validation failed: {e}")
                # Return unvalidated data with warning
                return transformed
        
        return transformed
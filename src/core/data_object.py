# src/core/data_object.py

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
import pandas as pd
import pandera.pandas as pa
from pydantic import BaseModel
from types import MappingProxyType


@dataclass
class PipelineDataObject:
    """
    Central state container for the three-phase MeshInsights pipeline.
    
    This class serves as the primary data flow object that accumulates datasets, metadata,
    results, logs, warnings, and errors throughout pipeline execution. It follows the 
    Context + Processor architectural pattern where processors mutate and return the same
    data object instance.

    Attributes:
        pipeline_name: Human-readable identifier for this pipeline instance.
        correlation_id: Unique identifier for this pipeline run, used for tracing and logging.
        start_time: UTC timestamp when this pipeline run was initiated.
        config: Pipeline configuration settings and runtime parameters.
        normalized_data: Mapping of dataset keys to normalized pandas DataFrames.
        artifacts: Combined storage for processor outputs and stage summaries.
        data_sources: Metadata captured during Phase 1 data acquisition.
        normalization_metadata: Metadata captured during Phase 2 normalization.
        errors: Errors collected during pipeline execution.
        warnings: Warnings collected during pipeline execution.
        artifact_schemas: Optional Pandera schemas for DataFrame artifacts.
    """
    # ========== Pipeline Metadata ==========
    pipeline_name: str = ""
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    config: Dict[str, Any] = field(default_factory=dict)

    # ========== Core Data ==========
    normalized_data: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ========== Processing Results / Data ==========
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # ========== Data Acquisition Metadata ==========
    data_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    normalization_metadata: Dict[str, Any] = field(default_factory=dict)

    # ========== Execution Tracking ==========
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # ========== Artifact Validation ==========
    artifact_schemas: Dict[str, pa.DataFrameSchema] = field(default_factory=dict)

    def get_dataset(self, key: str) -> pd.DataFrame:
        """
        Retrieve a normalized dataset by key.
        
        Args:
            key (str): Dataset key (typically matches DataSource name)
            
        Returns:
            pd.DataFrame: The normalized dataset
            
        Raises:
            KeyError: If the dataset key doesn't exist
        """
        return self.normalized_data[key]

    def set_artifact(self, key: str, value: Any) -> "PipelineDataObject":
        """
        Store an artifact produced by processors with optional DataFrame validation.
        
        This is the primary method for processors to store their outputs, including
        DataFrames, analysis results, derived data, and any intermediate results
        that other processors might need.
        
        If the value is a DataFrame and a schema is registered for this key,
        the DataFrame will be validated against the schema.
        
        Args:
            key (str): Namespaced key for the artifact (e.g., "cycles/filtered_data")
            value (Any): The artifact to store (DataFrame, dict, list, etc.)
            
        Returns:
            PipelineDataObject: Self for method chaining
        """
        # Validate DataFrame artifacts if schema is registered
        if isinstance(value, pd.DataFrame) and key in self.artifact_schemas:
            try:
                schema = self.artifact_schemas[key]
                validated_value = schema.validate(value, lazy=True)
                self.artifacts[key] = validated_value
                return self
            except (pa.errors.SchemaError, pa.errors.SchemaErrors) as e:
                warning_msg = f"Artifact '{key}' failed schema validation: {e}"
                self.add_warning(warning_msg)
                # Store anyway but with warning
                
        self.artifacts[key] = value
        return self

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a stored artifact by key.
        
        Used by processors to access outputs from previous processors in the pipeline.
        Returns the default value if the key doesn't exist.
        
        Args:
            key (str): The artifact key to retrieve
            default (Any): Value to return if key doesn't exist (default: None)
            
        Returns:
            Any: The stored artifact or default value
        """
        return self.artifacts.get(key, default)

    def register_artifact_schema(self, key: str, schema: pa.DataFrameSchema) -> "PipelineDataObject":
        """
        Register a Pandera schema for validating DataFrame artifacts.
        
        When a DataFrame is stored with set_artifact() using this key,
        it will be validated against this schema.
        
        Args:
            key (str): The artifact key
            schema (pa.DataFrameSchema): The Pandera schema to apply
            
        Returns:
            PipelineDataObject: Self for method chaining
        """
        self.artifact_schemas[key] = schema
        return self

    def list_artifacts(self, prefix: Optional[str] = None) -> List[str]:
        """
        List all artifact keys, optionally filtered by prefix.
        
        Args:
            prefix (Optional[str]): Filter to keys starting with this prefix.
                                  None returns all keys.
        
        Returns:
            List[str]: List of artifact keys matching the prefix filter
        """
        keys = list(self.artifacts.keys())
        return [k for k in keys if prefix is None or k.startswith(prefix)]

    def add_error(self, message: str) -> "PipelineDataObject":
        """
        Record an error with timestamp.
        
        Args:
            message (str): Error description
            
        Returns:
            PipelineDataObject: Self for method chaining
        """
        timestamped = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        self.errors.append(timestamped)
        return self

    def add_warning(self, message: str) -> "PipelineDataObject":
        """
        Record a non-fatal warning with timestamp.
        
        Used for data quality issues, minor configuration problems, and other
        concerns that don't stop pipeline execution but should be noted.
        
        Args:
            message (str): Warning description
            
        Returns:
            PipelineDataObject: Self for method chaining
        """
        timestamped = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        self.warnings.append(timestamped)
        return self

    def freeze_normalized_data(self) -> None:
        """
        Make `normalized_data` read-only:
        - set each DataFrame column's underlying NumPy array to non-writable
        - replace the mapping with a read-only MappingProxyType
        Params:
            None
        Returns:
            None
        """
        # Freeze column arrays (lightweight; no deep copy)
        for key, df in list(self.normalized_data.items()):
            if not isinstance(df, pd.DataFrame):
                continue
            for col in df.columns:
                try:
                    arr = df[col].values
                    arr.flags.writeable = False
                except Exception:
                    # best-effort; do not fail pipeline if a buffer can't be frozen
                    continue

        # Make mapping itself read-only so keys/values cannot be replaced
        try:
            self.normalized_data = MappingProxyType(dict(self.normalized_data))
        except Exception:
            # fallback: keep original dict if MappingProxyType fails
            pass

    # ========== Serialization Helpers ==========
    def to_dto(self) -> "PipelineRunDTO":
        """
        Create a lightweight DTO suitable for cross-boundary serialization.
        
        Extracts all metadata, logs, and indices while excluding heavy payloads
        like normalized DataFrames and processor artifacts. Useful for API responses,
        database storage, and inter-service communication.
        
        Returns:
            PipelineRunDTO: Serializable summary of pipeline run state
        """
        return PipelineRunDTO(
            pipeline_name=self.pipeline_name,
            correlation_id=self.correlation_id,
            start_time=self.start_time,
            config=self.config,
            data_sources=self.data_sources,
            normalization_metadata=self.normalization_metadata,
            errors=self.errors,
            warnings=self.warnings,
            artifact_index=self.list_artifacts(),
            dataset_index=list(self.normalized_data.keys()),
        )


class PipelineRunDTO(BaseModel):
    """
    Lightweight Data Transfer Object for cross-boundary serialization and validation.
    
    This DTO provides a serializable summary of pipeline run state while excluding
    heavy payloads like DataFrames and artifacts.
    """

    model_config = {"arbitrary_types_allowed": True}

    pipeline_name: str
    correlation_id: str
    start_time: datetime
    config: Dict[str, Any]
    data_sources: Dict[str, Dict[str, Any]]
    normalization_metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    artifact_index: List[str]
    dataset_index: List[str]
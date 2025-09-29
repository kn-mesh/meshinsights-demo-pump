from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
import time
import pandas as pd
import pandera.pandas as pa

from src.core.data_object import PipelineDataObject


class Processor(ABC):
    """
    Abstract base class for Phase 3 processors in the MeshInsights pipeline.

    Processors consume the normalized/merged dataset from `PipelineDataObject`,
    apply domain logic, and write results back to the data object.

    Contract and constraints:
    - Processors must be stateless with respect to internal instance fields.
    - Processors must NOT mutate `data_object.normalized_data` (treated as read-only).
    - All processor outputs must be written to `data_object.artifacts`
      via `data_object.set_artifact(...)`.
    - DataFrame artifacts can be validated by registering schemas.
    """

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a processor.

        Args:
            name (Optional[str]): Processor name. Defaults to the class name.
            config (Optional[Dict[str, Any]]): Configuration for this processor.
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(f"processor.{self.name}")

    @abstractmethod
    def process(self, data_object: PipelineDataObject) -> PipelineDataObject:
        """
        Transform the data object and return the updated data object.

        Implementations should:
        1. Read required data from `data_object.normalized_data` (read-only) and/or `data_object.artifacts`
        2. Perform processing logic
        3. Write results via `data_object.set_artifact(...)`
        4. Return the modified `data_object`

        Args:
            data_object (PipelineDataObject): Current pipeline data object.

        Returns:
            PipelineDataObject: Updated pipeline data object.
        """
        pass

    def register_output_schemas(self, data_object: PipelineDataObject) -> None:
        """
        Register Pandera schemas for DataFrame outputs this processor will create.
        
        Override this method to register schemas for any DataFrame artifacts
        this processor will create. This enables automatic validation when
        the processor calls set_artifact().
        
        Args:
            data_object (PipelineDataObject): Pipeline data object to register schemas with.
        """
        pass

    def validate_prerequisites(self, data_object: PipelineDataObject) -> None:
        """
        Validate that required data exists in the data object.

        Override to add processor-specific validation (e.g., required columns).

        Raises:
            ValueError: If prerequisites are not met.
        """
        # Ensure there is at least one normalized dataset
        if not data_object.normalized_data:
            raise ValueError(f"{self.name}: No normalized_data datasets available for processing")

    def validate_output(self, data_object: PipelineDataObject) -> None:
        """
        Validate processor output (optional hook).

        Override to enforce invariants on the mutated data object.

        Raises:
            ValueError: If output validation fails.
        """
        pass

    def __call__(self, data_object: PipelineDataObject) -> PipelineDataObject:
        """
        Execute the processor with logging, timing, and error handling.

        Args:
            data_object (PipelineDataObject): Current pipeline data object.

        Returns:
            PipelineDataObject: Updated data object (or original if configured to
            continue on error).
        """
        start_time = time.time()
        self.logger.info(f"Starting {self.name}")

        try:
            # Register output schemas before processing
            self.register_output_schemas(data_object)
            
            # Validate prerequisites
            self.validate_prerequisites(data_object)

            # Process
            result = self.process(data_object)

            # Validate output
            self.validate_output(result)

            # Log execution
            execution_time = time.time() - start_time
            self.logger.info(f"Completed {self.name} in {execution_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}", exc_info=True)
            data_object.add_error(f"{self.name}: {str(e)}")

            if self.config.get("stop_on_error", True):
                raise

            data_object.add_warning(f"Processor {self.name} failed: {str(e)}")
            return data_object
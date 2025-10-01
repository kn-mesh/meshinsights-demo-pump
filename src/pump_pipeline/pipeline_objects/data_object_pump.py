from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import pandera.pandas as pa

from src.core.data_object import PipelineDataObject


class PumpPipelineDataObject(PipelineDataObject):
    """Pump-specific `PipelineDataObject` with canonical artifacts.

    Public API focuses on intent-revealing getters/setters for pump analytics
    artifacts produced by processors in the CPU stage. Artifacts are validated
    with lightweight Pandera schemas when present.

    Artifacts
    - ``differential_pressure``: DataFrame with columns ``timestamp_utc`` (datetime, tz-aware) and ``dP_kPa`` (float)
    - ``efficiency``: DataFrame with columns ``timestamp_utc`` (datetime, tz-aware) and ``Eff`` (float)
    """

    ARTIFACT_DIFFERENTIAL_PRESSURE: str = "differential_pressure"
    ARTIFACT_EFFICIENCY: str = "efficiency"

    def __init__(self, *, pipeline_name: str = "", config: Optional[Dict[str, Any]] = None) -> None:
        """Create a pump-specific data object and register artifact schemas.

        Args:
            pipeline_name (str): Human-readable pipeline instance name.
            config (Optional[Dict[str, Any]]): Pipeline configuration and metadata.
        """
        super().__init__(pipeline_name=pipeline_name, config=config or {})
        self._register_canonical_artifact_schemas()

    # ===== Public API (simple, intent-revealing) =====
    def set_differential_pressure(self, data: pd.DataFrame) -> "PumpPipelineDataObject":
        """Store the differential pressure time-series artifact.

        Expects a DataFrame with columns ``timestamp_utc`` and ``dP_kPa``.

        Args:
            data (pd.DataFrame): Differential pressure time series.

        Returns:
            PumpPipelineDataObject: Self for fluent chaining.
        """
        self.set_artifact(self.ARTIFACT_DIFFERENTIAL_PRESSURE, data)
        return self

    def get_differential_pressure(self) -> Optional[pd.DataFrame]:
        """Return the stored differential pressure artifact, if present.

        Returns:
            Optional[pd.DataFrame]: DataFrame with ``timestamp_utc`` and ``dP_kPa`` columns or ``None``.
        """
        return self.get_artifact(self.ARTIFACT_DIFFERENTIAL_PRESSURE, None)

    def set_efficiency(self, data: pd.DataFrame) -> "PumpPipelineDataObject":
        """Store the efficiency time-series artifact.

        Expects a DataFrame with columns ``timestamp_utc`` and ``Eff``.

        Args:
            data (pd.DataFrame): Efficiency time series.

        Returns:
            PumpPipelineDataObject: Self for fluent chaining.
        """
        self.set_artifact(self.ARTIFACT_EFFICIENCY, data)
        return self

    def get_efficiency(self) -> Optional[pd.DataFrame]:
        """Return the stored efficiency artifact, if present.

        Returns:
            Optional[pd.DataFrame]: DataFrame with ``timestamp_utc`` and ``Eff`` columns or ``None``.
        """
        return self.get_artifact(self.ARTIFACT_EFFICIENCY, None)

    # ===== Internal helpers =====
    def _register_canonical_artifact_schemas(self) -> None:
        """Register Pandera schemas for canonical pump artifacts.

        Schemas provide soft validation for downstream processors and adapters.
        Validation warnings are captured on the data object; execution continues.
        """
        dp_schema = pa.DataFrameSchema(
            {
                "timestamp_utc": pa.Column(pa.DateTime, nullable=False),
                "dP_kPa": pa.Column(pa.Float, nullable=True),
            }
        )
        eff_schema = pa.DataFrameSchema(
            {
                "timestamp_utc": pa.Column(pa.DateTime, nullable=False),
                "Eff": pa.Column(pa.Float, nullable=True),
            }
        )
        self.register_artifact_schema(self.ARTIFACT_DIFFERENTIAL_PRESSURE, dp_schema)
        self.register_artifact_schema(self.ARTIFACT_EFFICIENCY, eff_schema)
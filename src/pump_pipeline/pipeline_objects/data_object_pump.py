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
    - ``head_spread_ratio``: DataFrame with columns ``week`` (str), ``recipe`` (str), and ``head_spread_ratio`` (float)
    - ``head_trend_slope``: DataFrame summarizing trend metrics per recipe with
      columns ``as_of_week`` (str), ``baseline_range`` (str), ``recipe`` (str),
      and ``head_trend_slope_pct`` (float)
    - ``intermittent_event_rate``: DataFrame with columns ``week`` (str),
      ``recipe`` (str), and ``intermittent_event_rate`` (float)
    """

    ARTIFACT_DIFFERENTIAL_PRESSURE: str = "differential_pressure"
    ARTIFACT_EFFICIENCY: str = "efficiency"
    ARTIFACT_HEAD_SPREAD_RATIO: str = "head_spread_ratio"
    ARTIFACT_HEAD_TREND_SLOPE: str = "head_trend_slope"
    ARTIFACT_INTERMITTENT_EVENT_RATE: str = "intermittent_event_rate"

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

    def set_head_spread_ratio(self, data: pd.DataFrame) -> "PumpPipelineDataObject":
        """Store the head-spread ratio time-series artifact.

        Expects a DataFrame with columns ``week``, ``recipe``, and
        ``head_spread_ratio``.

        Args:
            data (pd.DataFrame): Head spread ratio features by recipe/week.

        Returns:
            PumpPipelineDataObject: Self for fluent chaining.
        """
        self.set_artifact(self.ARTIFACT_HEAD_SPREAD_RATIO, data)
        return self

    def get_head_spread_ratio(self) -> Optional[pd.DataFrame]:
        """Return the stored head-spread ratio artifact, if present.

        Returns:
            Optional[pd.DataFrame]: DataFrame with columns ``week``,
            ``recipe``, and ``head_spread_ratio`` or ``None``.
        """
        return self.get_artifact(self.ARTIFACT_HEAD_SPREAD_RATIO, None)

    def set_head_trend_slope(self, data: pd.DataFrame) -> "PumpPipelineDataObject":
        """Store the head trend slope artifact aggregated per recipe.

        Expects a DataFrame with columns ``as_of_week``, ``baseline_range``,
        ``recipe``, and ``head_trend_slope_pct``.

        Args:
            data (pd.DataFrame): Head trend slope summary per recipe.

        Returns:
            PumpPipelineDataObject: Self for fluent chaining.
        """
        self.set_artifact(self.ARTIFACT_HEAD_TREND_SLOPE, data)
        return self

    def get_head_trend_slope(self) -> Optional[pd.DataFrame]:
        """Return the stored head trend slope artifact, if present.

        Returns:
            Optional[pd.DataFrame]: DataFrame with ``as_of_week``,
            ``baseline_range``, ``recipe``, and ``head_trend_slope_pct`` columns
            or ``None``.
        """
        return self.get_artifact(self.ARTIFACT_HEAD_TREND_SLOPE, None)

    def set_intermittent_event_rate(self, data: pd.DataFrame) -> "PumpPipelineDataObject":
        """Store the intermittent event rate artifact.

        Expects a DataFrame with columns ``week``, ``recipe``, and
        ``intermittent_event_rate`` plus diagnostic fields.

        Args:
            data (pd.DataFrame): Intermittent event rate metrics by recipe/week.

        Returns:
            PumpPipelineDataObject: Self for fluent chaining.
        """
        self.set_artifact(self.ARTIFACT_INTERMITTENT_EVENT_RATE, data)
        return self

    def get_intermittent_event_rate(self) -> Optional[pd.DataFrame]:
        """Return the stored intermittent event rate artifact, if present.

        Returns:
            Optional[pd.DataFrame]: DataFrame with ``week``, ``recipe``, and
            ``intermittent_event_rate`` columns or ``None``.
        """
        return self.get_artifact(self.ARTIFACT_INTERMITTENT_EVENT_RATE, None)

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
                # optional metadata for plotting
                "batch_id": pa.Column(pa.String, nullable=True),
                "recipe": pa.Column(pa.String, nullable=True),
            }
        )
        eff_schema = pa.DataFrameSchema(
            {
                "timestamp_utc": pa.Column(pa.DateTime, nullable=False),
                "Eff": pa.Column(pa.Float, nullable=True),
                # optional metadata for plotting
                "batch_id": pa.Column(pa.String, nullable=True),
                "recipe": pa.Column(pa.String, nullable=True),
            }
        )
        hsr_schema = pa.DataFrameSchema(
            {
                "week": pa.Column(pa.String, nullable=False),
                "recipe": pa.Column(pa.String, nullable=True),
                "head_spread_ratio": pa.Column(pa.Float, nullable=True),
                "head_spread_kpa": pa.Column(pa.Float, nullable=True),
                "baseline_spread_kpa": pa.Column(pa.Float, nullable=True),
                "sample_count": pa.Column(pa.Int, nullable=True),
            }
        )
        hts_schema = pa.DataFrameSchema(
            {
                "as_of_week": pa.Column(pa.String, nullable=False),
                "baseline_range": pa.Column(pa.String, nullable=True),
                "recipe": pa.Column(pa.String, nullable=True),
                "head_trend_slope_pct": pa.Column(pa.Float, nullable=True),
                "baseline_head_kpa": pa.Column(pa.Float, nullable=True),
                "latest_head_kpa": pa.Column(pa.Float, nullable=True),
                "ols_slope_pct_90": pa.Column(pa.Float, nullable=True),
                "weeks_observed": pa.Column(pa.Int, nullable=True),
                "baseline_sample_count": pa.Column(pa.Int, nullable=True),
                "total_samples": pa.Column(pa.Int, nullable=True),
            }
        )
        ier_schema = pa.DataFrameSchema(
            {
                "week": pa.Column(pa.String, nullable=False),
                "recipe": pa.Column(pa.String, nullable=True),
                "intermittent_event_rate": pa.Column(pa.Float, nullable=True),
                "event_count": pa.Column(pa.Int, nullable=True),
                "hours_in_week": pa.Column(pa.Float, nullable=True),
                "sample_count": pa.Column(pa.Int, nullable=True),
            }
        )
        self.register_artifact_schema(self.ARTIFACT_DIFFERENTIAL_PRESSURE, dp_schema)
        self.register_artifact_schema(self.ARTIFACT_EFFICIENCY, eff_schema)
        self.register_artifact_schema(self.ARTIFACT_HEAD_SPREAD_RATIO, hsr_schema)
        self.register_artifact_schema(self.ARTIFACT_HEAD_TREND_SLOPE, hts_schema)
        self.register_artifact_schema(self.ARTIFACT_INTERMITTENT_EVENT_RATE, ier_schema)

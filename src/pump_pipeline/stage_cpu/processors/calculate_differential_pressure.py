from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd

from src.core.processor import Processor
from src.pump_pipeline.pipeline_objects.data_object_pump import PumpPipelineDataObject


class CalculateDifferentialPressureProcessor(Processor):
    """Processor that extracts/derives the differential pressure time-series.

    The processor expects a normalized dataset named ``pump_telemetry`` to be
    present on the data object. It writes a DataFrame artifact keyed by
    ``PumpPipelineDataObject.ARTIFACT_DIFFERENTIAL_PRESSURE`` containing the
    columns ``timestamp_utc`` and ``dP_kPa``.
    """

    def register_output_schemas(self, data_object: PumpPipelineDataObject) -> None:
        # Canonical schemas are registered by the PumpPipelineDataObject on
        # construction, but keep the hook in case a processor-specific schema
        # is desired in the future. No-op here.
        return None

    def validate_prerequisites(self, data_object: PumpPipelineDataObject) -> None:
        # Ensure normalized dataset exists and has required columns
        super().validate_prerequisites(data_object)
        if "pump_telemetry" not in data_object.normalized_data:
            raise ValueError(f"{self.name}: normalized dataset 'pump_telemetry' not found")

        df = data_object.normalized_data["pump_telemetry"]
        if "timestamp_utc" not in df.columns:
            raise ValueError(f"{self.name}: 'timestamp_utc' missing from pump_telemetry")

        # Either dP_kPa is present or we can compute it from Pd_kPa & Ps_kPa
        if not any(col in df.columns for col in ("dP_kPa", "Pd_kPa", "Ps_kPa")):
            raise ValueError(
                f"{self.name}: neither 'dP_kPa' nor 'Pd_kPa'/'Ps_kPa' present in pump_telemetry"
            )

    def process(self, data_object: PumpPipelineDataObject) -> PumpPipelineDataObject:
        df = data_object.get_dataset("pump_telemetry").copy()

        # Restrict to active batches only (pump running)
        if "batch_id" in df.columns:
            active_mask = df["batch_id"].notna() & (df["batch_id"] != "")
            df = df.loc[active_mask]

        # Ensure timestamp is datetime-like
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

        # Derive dP_kPa if missing on active rows
        if "dP_kPa" not in df.columns:
            if "Pd_kPa" in df.columns and "Ps_kPa" in df.columns:
                df["dP_kPa"] = df["Pd_kPa"] - df["Ps_kPa"]
            else:
                data_object.add_warning(f"{self.name}: unable to derive dP_kPa; leaving NaNs")
                df["dP_kPa"] = pd.NA

        # include batch and recipe if present so downstream plotting can color by them
        cols = ["timestamp_utc", "dP_kPa"]
        for optional in ("batch_id", "recipe"):
            if optional in df.columns:
                cols.append(optional)

        out = df[cols].copy()
        out = out.sort_values("timestamp_utc").reset_index(drop=True)

        # Store artifact via data object API (will validate against registered schema)
        data_object.set_artifact(PumpPipelineDataObject.ARTIFACT_DIFFERENTIAL_PRESSURE, out)
        return data_object


__all__ = ["CalculateDifferentialPressureProcessor"]



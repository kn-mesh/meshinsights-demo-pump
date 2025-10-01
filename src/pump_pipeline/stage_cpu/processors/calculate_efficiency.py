from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from src.core.processor import Processor
from src.pump_pipeline.pipeline_objects.data_object_pump import PumpPipelineDataObject


class CalculateEfficiencyProcessor(Processor):
    """Processor that computes the efficiency proxy time-series.

    Efficiency is defined as Eff = (Q_m3h * dP_kPa) / I_A per the simulator
    specification. The processor reads the normalized ``pump_telemetry`` dataset
    and writes a DataFrame artifact keyed by
    ``PumpPipelineDataObject.ARTIFACT_EFFICIENCY`` with columns ``timestamp_utc``
    and ``Eff``.
    """

    ZERO_CURRENT_TOLERANCE: float = 1e-9

    def register_output_schemas(self, data_object: PumpPipelineDataObject) -> None:
        # schemas already registered by PumpPipelineDataObject
        return None

    def validate_prerequisites(self, data_object: PumpPipelineDataObject) -> None:
        super().validate_prerequisites(data_object)
        if "pump_telemetry" not in data_object.normalized_data:
            raise ValueError(f"{self.name}: normalized dataset 'pump_telemetry' not found")

        df = data_object.normalized_data["pump_telemetry"]
        required = {"timestamp_utc", "Q_m3h"}
        if not required.issubset(df.columns):
            raise ValueError(f"{self.name}: missing required columns: {required - set(df.columns)}")

        # Need either Eff already or dP_kPa + I_A to compute
        if "Eff" not in df.columns and not ("dP_kPa" in df.columns and "I_A" in df.columns):
            raise ValueError(
                f"{self.name}: neither 'Eff' present nor both 'dP_kPa' and 'I_A' available to compute"
            )

    def process(self, data_object: PumpPipelineDataObject) -> PumpPipelineDataObject:
        df = data_object.get_dataset("pump_telemetry").copy()

        # Restrict to active batches only (pump running)
        if "batch_id" in df.columns:
            active_mask = df["batch_id"].notna() & (df["batch_id"] != "")
            df = df.loc[active_mask]

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

        # If Eff is missing, compute it as (Q_m3h * dP_kPa) / I_A
        if "Eff" not in df.columns:
            # avoid divide-by-zero and invalid values
            numerator = df["Q_m3h"].to_numpy(dtype=float) * df["dP_kPa"].to_numpy(dtype=float)
            denom = df["I_A"].to_numpy(dtype=float)

            with np.errstate(divide="ignore", invalid="ignore"):
                eff = numerator / denom

            # replace inf/-inf with NaN and mask very small currents
            eff = pd.Series(eff).replace([np.inf, -np.inf], np.nan)
            eff.loc[df["I_A"].abs() <= self.ZERO_CURRENT_TOLERANCE] = pd.NA
            df["Eff"] = eff

        # include batch and recipe if present so downstream plotting can color by them
        cols = ["timestamp_utc", "Eff"]
        for optional in ("batch_id", "recipe"):
            if optional in df.columns:
                cols.append(optional)

        out = df[cols].copy()
        out = out.sort_values("timestamp_utc").reset_index(drop=True)

        data_object.set_artifact(PumpPipelineDataObject.ARTIFACT_EFFICIENCY, out)
        return data_object


__all__ = ["CalculateEfficiencyProcessor"]



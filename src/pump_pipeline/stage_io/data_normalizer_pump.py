"""Pump-specific normalizer for MeshInsights Stage 1 output.

Transforms the raw simulator slice retrieved by :class:`PumpDataRetriever`
into the canonical schema consumed by downstream processors.

Normalized dataset columns:

* timestamp_utc (datetime, tz-aware UTC)
* pump_id, batch_id, recipe (string dtype; missing -> ``<NA>``)
* Ps_kPa, Pd_kPa, Q_m3h, I_A, dP_kPa, Eff (float, rounded to 4 decimals)
"""

from __future__ import annotations

from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd

from src.core.data_normalizer import DataNormalizer


class PumpDataNormalizer(DataNormalizer):
    """Normalize raw pump telemetry into the standard MeshInsights schema."""

    REQUIRED_COLUMNS: List[str] = [
        "timestamp_utc",
        "pump_id",
        "batch_id",
        "recipe",
        "Ps_kPa",
        "Pd_kPa",
        "Q_m3h",
        "I_A",
    ]

    STRING_COLUMNS: List[str] = ["pump_id", "batch_id", "recipe"]
    NUMERIC_COLUMNS: List[str] = ["Ps_kPa", "Pd_kPa", "Q_m3h", "I_A"]
    ROUND_COLUMNS: List[str] = [
        "Ps_kPa",
        "Pd_kPa",
        "Q_m3h",
        "I_A",
        "dP_kPa",
        "Eff",
    ]
    OUTPUT_COLUMNS: List[str] = [
        "timestamp_utc",
        "pump_id",
        "batch_id",
        "recipe",
        "Ps_kPa",
        "Pd_kPa",
        "Q_m3h",
        "I_A",
        "dP_kPa",
        "Eff",
    ]

    ZERO_CURRENT_TOLERANCE: float = 1e-9

    def transform(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing:
            raise ValueError(f"PumpDataNormalizer missing required columns: {missing}")

        df = data.copy()

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        invalid_ts = df["timestamp_utc"].isna()
        if invalid_ts.any():
            self.logger.warning("Dropping %s rows with invalid timestamps", int(invalid_ts.sum()))
            df = df.loc[~invalid_ts]

        for column in self.STRING_COLUMNS:
            df[column] = df[column].astype("string").str.strip()

        for column in self.NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df["dP_kPa"] = df["Pd_kPa"] - df["Ps_kPa"]

        with np.errstate(divide="ignore", invalid="ignore"):
            eff = (df["Q_m3h"] * df["dP_kPa"]) / df["I_A"]
        eff = eff.replace([np.inf, -np.inf], np.nan)
        eff.loc[df["I_A"].abs() <= self.ZERO_CURRENT_TOLERANCE] = np.nan
        df["Eff"] = eff

        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        for column in self.ROUND_COLUMNS:
            df[column] = df[column].round(4)

        return df[self.OUTPUT_COLUMNS]

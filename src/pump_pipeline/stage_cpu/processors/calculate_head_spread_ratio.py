from __future__ import annotations

from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from src.core.processor import Processor
from src.pump_pipeline.pipeline_objects.data_object_pump import PumpPipelineDataObject


class CalculateHeadSpreadRatioProcessor(Processor):
    """Processor that computes the head-spread ratio (HNI) per recipe/week.

    The processor consumes the ``pump_telemetry`` dataset and produces a
    DataFrame artifact keyed by
    ``PumpPipelineDataObject.ARTIFACT_HEAD_SPREAD_RATIO``. Output includes the
    week start timestamp, recipe label, the head-spread ratio relative to the
    baseline (first 14 days per recipe), and helper columns for diagnostics.
    """

    BASELINE_WINDOW_DAYS: int = 14
    DENOMINATOR_EPS: float = 1e-6

    def register_output_schemas(self, data_object: PumpPipelineDataObject) -> None:
        # Schemas are registered by PumpPipelineDataObject; no-op.
        return None

    def validate_prerequisites(self, data_object: PumpPipelineDataObject) -> None:
        super().validate_prerequisites(data_object)
        if "pump_telemetry" not in data_object.normalized_data:
            raise ValueError(f"{self.name}: normalized dataset 'pump_telemetry' not found")

        df = data_object.normalized_data["pump_telemetry"]
        required_columns = {"timestamp_utc", "recipe"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"{self.name}: missing required columns: {missing}")

        if "dP_kPa" not in df.columns and not ({"Pd_kPa", "Ps_kPa"} <= set(df.columns)):
            raise ValueError(
                f"{self.name}: need 'dP_kPa' or both 'Pd_kPa' and 'Ps_kPa' to compute head spread"
            )

    def process(self, data_object: PumpPipelineDataObject) -> PumpPipelineDataObject:
        df = data_object.get_dataset("pump_telemetry").copy()

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"])

        # Derive differential pressure if absent.
        if "dP_kPa" not in df.columns:
            df["dP_kPa"] = df["Pd_kPa"] - df["Ps_kPa"]

        # Focus on rows with a non-empty recipe label.
        recipe_mask = df["recipe"].notna() & (df["recipe"].astype(str).str.strip() != "")
        df = df.loc[recipe_mask].copy()
        df = df.dropna(subset=["dP_kPa"])

        if df.empty:
            empty = pd.DataFrame(
                columns=[
                    "timestamp_utc",
                    "recipe",
                    "head_spread_ratio",
                    "head_spread_kpa",
                    "baseline_spread_kpa",
                    "sample_count",
                ]
            )
            data_object.set_artifact(PumpPipelineDataObject.ARTIFACT_HEAD_SPREAD_RATIO, empty)
            return data_object

        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        records: List[Dict[str, Any]] = []
        baseline_window = pd.Timedelta(days=self.BASELINE_WINDOW_DAYS)

        for recipe, group in df.groupby("recipe", sort=True):
            group = group.sort_values("timestamp_utc")
            first_ts = group["timestamp_utc"].min()
            cutoff = first_ts + baseline_window

            baseline_series = group.loc[group["timestamp_utc"] < cutoff, "dP_kPa"].dropna()
            if baseline_series.empty:
                baseline_spread = np.nan
            else:
                q95 = baseline_series.quantile(0.95)
                q05 = baseline_series.quantile(0.05)
                baseline_spread = float(q95 - q05)

            weekly_groups = group.groupby(pd.Grouper(key="timestamp_utc", freq="W-MON", label="left"))
            weekly_spread = weekly_groups["dP_kPa"].apply(self._spread_quantiles)
            weekly_counts = weekly_groups["dP_kPa"].count()

            for week_start, spread in weekly_spread.items():
                if pd.isna(spread):
                    ratio = np.nan
                else:
                    denom = baseline_spread
                    if denom is None or pd.isna(denom) or abs(denom) <= self.DENOMINATOR_EPS:
                        ratio = np.nan
                    else:
                        ratio = float(spread) / float(denom)

                timestamp = self._ensure_utc_timestamp(week_start)
                records.append(
                    {
                        "timestamp_utc": timestamp,
                        "recipe": recipe,
                        "head_spread_ratio": ratio,
                        "head_spread_kpa": float(spread) if not pd.isna(spread) else np.nan,
                        "baseline_spread_kpa": baseline_spread,
                        "sample_count": int(weekly_counts.get(week_start, 0)),
                    }
                )

        result = pd.DataFrame.from_records(records)
        if not result.empty:
            result = result.sort_values(["recipe", "timestamp_utc"]).reset_index(drop=True)

        data_object.set_artifact(PumpPipelineDataObject.ARTIFACT_HEAD_SPREAD_RATIO, result)
        return data_object

    @staticmethod
    def _spread_quantiles(series: pd.Series) -> float:
        """Return the 95th-5th percentile spread for the provided values."""
        if series.empty:
            return np.nan
        q95 = series.quantile(0.95)
        q05 = series.quantile(0.05)
        if pd.isna(q95) or pd.isna(q05):
            return np.nan
        return float(q95 - q05)

    @staticmethod
    def _ensure_utc_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
        """Normalize a pandas Timestamp to be timezone-aware UTC."""
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")


__all__ = ["CalculateHeadSpreadRatioProcessor"]

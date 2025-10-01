from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.core.processor import Processor
from src.pump_pipeline.pipeline_objects.data_object_pump import PumpPipelineDataObject


class CalculateIntermittentEventRateProcessor(Processor):
    """Compute the intermittent event rate (IER) per recipe and calendar week.

    Implements the simplified cavitation KPI described in ``PumpAlgorithms.md``:

    - Identify short flow dips (two or more consecutive 5-minute samples where
      ``Q_m3h`` is below ``0.95`` of the week's median).
    - Confirm the dip as an intermittent cavitation event when any sample in the
      dip has ``I_A`` above ``1.08`` of the week's median.
    - Collapse overlapping dips into a single event.
    - Express the event rate as ``events / hours_in_week``.

    The processor writes a DataFrame artifact keyed by
    ``PumpPipelineDataObject.ARTIFACT_INTERMITTENT_EVENT_RATE``. The output
    includes a human-readable ``week`` range string (e.g., "1/1/25 - 1/7/25")
    plus helper columns for downstream interpretation.
    """

    FLOW_DROP_FACTOR: float = 0.95
    CURRENT_SPIKE_FACTOR: float = 1.08
    SAMPLE_MINUTES: int = 5
    WEEK_FREQ: str = "W-MON"

    def register_output_schemas(self, data_object: PumpPipelineDataObject) -> None:
        # Schemas are registered at data object construction time.
        return None

    def validate_prerequisites(self, data_object: PumpPipelineDataObject) -> None:
        super().validate_prerequisites(data_object)
        if "pump_telemetry" not in data_object.normalized_data:
            raise ValueError(f"{self.name}: normalized dataset 'pump_telemetry' not found")

        df = data_object.normalized_data["pump_telemetry"]
        required_columns = {"timestamp_utc", "recipe", "Q_m3h", "I_A"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"{self.name}: missing required columns: {missing}")

    def process(self, data_object: PumpPipelineDataObject) -> PumpPipelineDataObject:
        df = data_object.get_dataset("pump_telemetry").copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"])

        # Focus on active rows that have a recipe label.
        recipe_mask = df["recipe"].notna() & (df["recipe"].astype(str).str.strip() != "")
        df = df.loc[recipe_mask].copy()
        if df.empty:
            data_object.set_intermittent_event_rate(self._empty_output())
            return data_object

        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        records: List[Dict[str, Any]] = []
        for recipe, recipe_df in df.groupby("recipe", sort=True):
            recipe_df = recipe_df.sort_values("timestamp_utc")
            weekly_groups = recipe_df.groupby(
                pd.Grouper(key="timestamp_utc", freq=self.WEEK_FREQ, label="left")
            )

            for week_start, week_df in weekly_groups:
                if week_df.empty:
                    continue

                q_median = week_df["Q_m3h"].median(skipna=True)
                i_median = week_df["I_A"].median(skipna=True)
                sample_count = int(len(week_df))
                hours_in_week = (sample_count * self.SAMPLE_MINUTES) / 60.0

                if np.isnan(q_median) or np.isnan(i_median):
                    event_count = 0
                    ier = np.nan
                else:
                    flow_threshold = self.FLOW_DROP_FACTOR * float(q_median)
                    current_threshold = self.CURRENT_SPIKE_FACTOR * float(i_median)
                    event_count = self._count_weekly_events(week_df, flow_threshold, current_threshold)
                    ier = event_count / hours_in_week if hours_in_week > 0 else np.nan

                week_ts = self._ensure_utc_timestamp(week_start)
                week_label = self._format_week_range(week_ts)
                records.append(
                    {
                        "_week_start": week_ts,
                        "week": week_label,
                        "recipe": recipe,
                        "intermittent_event_rate": float(ier) if not pd.isna(ier) else np.nan,
                        "event_count": int(event_count),
                        "hours_in_week": float(hours_in_week),
                        "sample_count": sample_count,
                    }
                )

        output = pd.DataFrame.from_records(records)

        if not output.empty:
            output = output.sort_values(["recipe", "_week_start"]).reset_index(drop=True)
            output = output.drop(columns=["_week_start"], errors="ignore")

        data_object.set_intermittent_event_rate(output)
        return data_object

    def _count_weekly_events(
        self,
        week_df: pd.DataFrame,
        flow_threshold: float,
        current_threshold: float,
    ) -> int:
        """Count intermittent events within a recipe/week segment."""
        ordered = week_df.sort_values("timestamp_utc")
        flow_values = ordered["Q_m3h"].to_numpy(dtype=float, copy=False)
        current_values = ordered["I_A"].to_numpy(dtype=float, copy=False)

        run_length = 0
        high_current_in_run = False
        event_count = 0

        for flow, current in zip(flow_values, current_values):
            is_low_flow = flow <= flow_threshold if not np.isnan(flow) else False
            if is_low_flow:
                run_length += 1
                if not np.isnan(current) and current >= current_threshold:
                    high_current_in_run = True
            else:
                if run_length >= 2 and high_current_in_run:
                    event_count += 1
                run_length = 0
                high_current_in_run = False

        if run_length >= 2 and high_current_in_run:
            event_count += 1

        return event_count

    @staticmethod
    def _ensure_utc_timestamp(timestamp: pd.Timestamp) -> pd.Timestamp:
        """Normalize a pandas Timestamp to be timezone-aware UTC."""
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    @staticmethod
    def _format_week_range(week_start_utc: pd.Timestamp) -> str:
        """Return a human-readable week range like "M/D/YY - M/D/YY" in UTC."""
        start = CalculateIntermittentEventRateProcessor._ensure_utc_timestamp(week_start_utc)
        end = start + pd.Timedelta(days=6)
        def fmt(ts: pd.Timestamp) -> str:
            yy = ts.year % 100
            return f"{ts.month}/{ts.day}/{yy:02d}"
        return f"{fmt(start)} - {fmt(end)}"

    @staticmethod
    def _empty_output() -> pd.DataFrame:
        """Return an empty DataFrame with the expected schema."""
        return pd.DataFrame(
            columns=[
                "week",
                "recipe",
                "intermittent_event_rate",
                "event_count",
                "hours_in_week",
                "sample_count",
            ]
        )


__all__ = ["CalculateIntermittentEventRateProcessor"]

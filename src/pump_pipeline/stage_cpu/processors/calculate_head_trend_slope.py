from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.core.processor import Processor
from src.pump_pipeline.pipeline_objects.data_object_pump import PumpPipelineDataObject


@dataclass(frozen=True)
class _RecipeTrendResult:
    """Container for per-recipe head trend statistics."""

    recipe: str
    week_start: pd.Timestamp  # internal sort key
    as_of_week: str
    baseline_range: str
    head_trend_slope_pct: float
    baseline_head_kpa: float
    latest_head_kpa: float
    ols_slope_pct_90: float
    weeks_observed: int
    baseline_sample_count: int
    total_samples: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "_week_start": self.week_start,
            "recipe": self.recipe,
            "as_of_week": self.as_of_week,
            "baseline_range": self.baseline_range,
            "head_trend_slope_pct": self.head_trend_slope_pct,
            "baseline_head_kpa": self.baseline_head_kpa,
            "latest_head_kpa": self.latest_head_kpa,
            "ols_slope_pct_90": self.ols_slope_pct_90,
            "weeks_observed": self.weeks_observed,
            "baseline_sample_count": self.baseline_sample_count,
            "total_samples": self.total_samples,
        }


class CalculateHeadTrendSlopeProcessor(Processor):
    """Compute the head trend slope KPI per recipe.

    The processor consumes the normalized ``pump_telemetry`` dataset, aggregates
    median differential pressure by calendar week per recipe, and compares the
    latest week to the first 14-day baseline. Outputs a summary artifact with
    head trend slope percentages and supporting context.
    """

    BASELINE_WINDOW_DAYS: int = 14
    BASELINE_EPSILON_KPA: float = 1.0  # clamp denominator when baseline ~0

    def register_output_schemas(self, data_object: PumpPipelineDataObject) -> None:
        # Schemas registered on PumpPipelineDataObject construction
        return None

    def validate_prerequisites(self, data_object: PumpPipelineDataObject) -> None:
        super().validate_prerequisites(data_object)
        if "pump_telemetry" not in data_object.normalized_data:
            raise ValueError(f"{self.name}: normalized dataset 'pump_telemetry' not found")

        df = data_object.normalized_data["pump_telemetry"]
        required = {"timestamp_utc", "recipe", "dP_kPa"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"{self.name}: missing required columns: {missing}")

    def process(self, data_object: PumpPipelineDataObject) -> PumpPipelineDataObject:
        df = data_object.get_dataset("pump_telemetry").copy()

        if df.empty:
            data_object.set_head_trend_slope(self._empty_result())
            return data_object

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"])

        # Filter to active rows only (recipe present). Idle rows have empty recipe.
        if "recipe" in df.columns:
            df["recipe"] = df["recipe"].astype("string").str.strip()
            df = df[df["recipe"].notna() & (df["recipe"] != "")]

        df = df.dropna(subset=["dP_kPa"])
        if df.empty:
            data_object.add_warning(f"{self.name}: no dP_kPa samples after filtering active rows")
            data_object.set_head_trend_slope(self._empty_result())
            return data_object

        results = []
        for recipe, group in df.groupby("recipe"):
            result = self._compute_recipe_trend(recipe, group.sort_values("timestamp_utc"))
            if result is not None:
                results.append(result.as_dict())

        if not results:
            data_object.add_warning(f"{self.name}: insufficient data to compute head trend slope")
            data_object.set_head_trend_slope(self._empty_result())
            return data_object

        out_df = pd.DataFrame(results)
        out_df = out_df.sort_values(["recipe", "_week_start"]).reset_index(drop=True)
        out_df = out_df.drop(columns=["_week_start"], errors="ignore")
        data_object.set_head_trend_slope(out_df)
        return data_object

    def _compute_recipe_trend(self, recipe: str, df: pd.DataFrame) -> _RecipeTrendResult | None:
        if df.empty:
            return None

        baseline_end = df["timestamp_utc"].min() + pd.Timedelta(days=self.BASELINE_WINDOW_DAYS)
        baseline = df[df["timestamp_utc"] < baseline_end]
        if baseline.empty:
            return None

        baseline_head = float(baseline["dP_kPa"].median(skipna=True))
        baseline_samples = int(baseline["dP_kPa"].notna().sum())

        # Compute week start (Monday) without converting to PeriodArray which drops timezone
        weekly = (
            df.assign(
                week_start=(df["timestamp_utc"] - pd.to_timedelta(df["timestamp_utc"].dt.weekday, unit="d")).dt.normalize()
            )
            .groupby("week_start", as_index=False)
            .agg(
                median_head_kpa=("dP_kPa", "median"),
                sample_count=("dP_kPa", "count"),
            )
            .sort_values("week_start")
        )

        if weekly.empty:
            return None

        # Compute slope via OLS on weekly medians when at least two weeks exist.
        slope_pct_90 = np.nan
        if len(weekly) >= 2 and not np.isnan(baseline_head):
            x_days = (weekly["week_start"] - weekly["week_start"].min()) / pd.Timedelta(days=1)
            x = x_days.to_numpy(dtype=float)
            y = weekly["median_head_kpa"].to_numpy(dtype=float)
            if np.any(np.isfinite(x)) and np.any(np.isfinite(y)) and np.ptp(x) > 0:
                slope, _intercept = np.polyfit(x, y, 1)
                denom = self._baseline_denominator(baseline_head)
                slope_pct_90 = float((slope * 90.0) / denom * 100.0)

        latest_row = weekly.iloc[-1]
        latest_head = float(latest_row["median_head_kpa"])
        pct_change = np.nan
        if not np.isnan(baseline_head):
            denom = self._baseline_denominator(baseline_head)
            pct_change = float(((latest_head - baseline_head) / denom) * 100.0)

        head_trend_pct = pct_change
        if np.isnan(head_trend_pct) and not np.isnan(slope_pct_90):
            head_trend_pct = slope_pct_90

        if np.isnan(head_trend_pct) and np.isnan(slope_pct_90):
            return None

        week_start = latest_row["week_start"]
        if week_start.tzinfo is None:
            week_start = week_start.tz_localize("UTC")

        as_of_week = self._format_week_range(week_start)
        baseline_start = df["timestamp_utc"].min().tz_convert("UTC")
        baseline_end = baseline_start + pd.Timedelta(days=self.BASELINE_WINDOW_DAYS - 1)
        baseline_range = self._format_week_range(baseline_start) if self.BASELINE_WINDOW_DAYS == 7 else f"{baseline_start.month}/{baseline_start.day}/{baseline_start.year % 100:02d} - {baseline_end.month}/{baseline_end.day}/{baseline_end.year % 100:02d}"

        return _RecipeTrendResult(
            recipe=str(recipe),
            week_start=week_start,
            as_of_week=as_of_week,
            baseline_range=baseline_range,
            head_trend_slope_pct=head_trend_pct,
            baseline_head_kpa=baseline_head,
            latest_head_kpa=latest_head,
            ols_slope_pct_90=slope_pct_90,
            weeks_observed=int(len(weekly)),
            baseline_sample_count=baseline_samples,
            total_samples=int(df["dP_kPa"].notna().sum()),
        )

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        columns = [
            "as_of_week",
            "baseline_range",
            "recipe",
            "head_trend_slope_pct",
            "baseline_head_kpa",
            "latest_head_kpa",
            "ols_slope_pct_90",
            "weeks_observed",
            "baseline_sample_count",
            "total_samples",
        ]
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _format_week_range(week_start_utc: pd.Timestamp) -> str:
        """Return a human-readable week range like "M/D/YY - M/D/YY" in UTC."""
        start = week_start_utc.tz_convert("UTC") if week_start_utc.tzinfo is not None else week_start_utc.tz_localize("UTC")
        end = start + pd.Timedelta(days=6)
        def fmt(ts: pd.Timestamp) -> str:
            yy = ts.year % 100
            return f"{ts.month}/{ts.day}/{yy:02d}"
        return f"{fmt(start)} - {fmt(end)}"

    def _baseline_denominator(self, baseline_head: float) -> float:
        if np.isnan(baseline_head) or not np.isfinite(baseline_head):
            return self.BASELINE_EPSILON_KPA
        return max(abs(baseline_head), self.BASELINE_EPSILON_KPA)


__all__ = ["CalculateHeadTrendSlopeProcessor"]

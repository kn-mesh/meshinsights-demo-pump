"""Synthetic telemetry generator for centrifugal pump health scenarios.

This module builds 5-minute telemetry covering multiple months for three pump
health narratives captured in ``Docs/PumpSimulatorSpec.md``:

* ``CAVITATION`` – suction degradation with widening pressure variance.
* ``IMPELLER_WEAR`` – gradual hydraulic efficiency loss.
* ``HEALTHY_FP`` – benign process-driven shifts that resemble alerts.

Executing the module (``python -m src.pump_data.pump_simulator``) generates two
CSV files inside :mod:`src.pump_data`:

``pump_timeseries.csv``
    All simulated 5-minute telemetry with derived KPIs per pump.

``pump_labels.csv``
    Per-pump class windows to simplify downstream analytics / model training.

The implementation sticks to NumPy and pandas so that simulations remain
reproducible, configurable, and fast without extra dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SIM_OUTPUT_DIR = Path(__file__).resolve().parent

PUMP_CLASSES = {
    "cavitation": "CAVITATION",
    "impeller_wear": "IMPELLER_WEAR",
    "healthy_fp": "HEALTHY_FP",
}


@dataclass(slots=True)
class SimulationConfig:
    """High level configuration for the telemetry simulation."""

    start_timestamp: pd.Timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    days: int = 90
    freq_minutes: int = 5
    pump_counts: Dict[str, int] = field(
        default_factory=lambda: {"cavitation": 1, "impeller_wear": 1, "healthy_fp": 1}
    )
    random_seed: int | None = 42

    def __post_init__(self) -> None:
        unsupported = set(self.pump_counts) - set(PUMP_CLASSES)
        if unsupported:
            raise ValueError(f"Unsupported pump types supplied: {unsupported}")


@dataclass(slots=True)
class PumpSimulationResult:
    """Bundle containing a pump's telemetry and associated class labels."""

    data: pd.DataFrame
    labels: List[Dict[str, object]]


def create_time_index(config: SimulationConfig) -> pd.DatetimeIndex:
    """Return the simulation timestamps with uniform spacing."""

    periods = int((config.days * 24 * 60) / config.freq_minutes)
    return pd.date_range(
        start=config.start_timestamp,
        periods=periods,
        freq=f"{config.freq_minutes}min",
        tz="UTC",
    )


def dP_expected_from_flow(flow_m3h: np.ndarray) -> np.ndarray:
    """OEM differential pressure curve (quadratic) for the pump."""

    a, b, c = 320.0, 0.60, 0.0010
    return a - b * flow_m3h - c * np.square(flow_m3h)


def ensure_non_negative(values: np.ndarray) -> np.ndarray:
    """Guard helper to avoid negative physical quantities."""

    return np.maximum(values, 0.0)


def recompute_power_signals(df: pd.DataFrame) -> None:
    """Recalculate discharge pressure and efficiency after adjustments."""

    df["Pd_kPa"] = df["Ps_kPa"] + df["dP_kPa"]
    active = df["is_active"].to_numpy()
    eff_numerator = df["Q_m3h"] * df["dP_kPa"]

    with np.errstate(divide="ignore", invalid="ignore"):
        df.loc[active, "Eff"] = eff_numerator.to_numpy()[active] / df.loc[active, "I_A"]
    df.loc[~active, "Eff"] = 0.0


def sample_batch_schedule(
    rng: np.random.Generator, num_batches: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return start times, durations, and recipes for a single day."""

    recipes = np.array([65, 80, 95])
    while True:
        durations = rng.integers(120, 361, size=num_batches)
        if durations.sum() <= 1200:  # keep batches within the workday
            break

    starts = np.zeros(num_batches)
    cursor = float(rng.uniform(10, 90))
    for idx, duration in enumerate(durations):
        latest_start = 1440 - float(np.sum(durations[idx:])) - 20.0
        if cursor > latest_start:
            cursor = max(0.0, latest_start - rng.uniform(0, 30))
        starts[idx] = cursor
        cursor += duration + rng.uniform(45, 120)

    return starts, durations, rng.choice(recipes, size=num_batches)


def initialise_day_batches(
    df: pd.DataFrame,
    day_index: int,
    pump_id: str,
    batch_counter: int,
    rng: np.random.Generator,
) -> int:
    """Populate the active batch metadata for the supplied day."""

    num_batches = int(rng.integers(2, 5))
    starts, durations, recipes = sample_batch_schedule(rng, num_batches)

    day_start = df.index.min() + pd.Timedelta(days=day_index)
    for start_minute, duration, recipe in zip(starts, durations, recipes):
        start_ts = day_start + pd.Timedelta(minutes=float(start_minute))
        end_ts = start_ts + pd.Timedelta(minutes=int(duration))
        mask = (df.index >= start_ts) & (df.index < end_ts)
        if not mask.any():
            continue

        batch_id = f"{pump_id}_B{batch_counter:04d}"
        df.loc[mask, "is_active"] = True
        df.loc[mask, "batch_id"] = batch_id
        df.loc[mask, "recipe"] = f"R{int(recipe)}"
        df.loc[mask, "Q_set_m3h"] = float(recipe)
        batch_counter += 1

    return batch_counter


def build_baseline_profile(
    pump_id: str, config: SimulationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Create the baseline (healthy) operating profile for a pump."""

    index = create_time_index(config)
    df = pd.DataFrame(index=index)
    df.index.name = "timestamp_utc"
    df["pump_id"] = pump_id
    df["is_active"] = False
    df["batch_id"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["recipe"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["Q_set_m3h"] = 0.0

    batch_counter = 1
    for day_idx in range(config.days):
        batch_counter = initialise_day_batches(df, day_idx, pump_id, batch_counter, rng)

    is_active = df["is_active"].to_numpy()
    active_idx = np.where(is_active)[0]
    idle_idx = np.where(~is_active)[0]
    q_set = df["Q_set_m3h"].to_numpy()

    q_values = np.zeros(df.shape[0])
    if active_idx.size:
        q_values[active_idx] = ensure_non_negative(
            rng.normal(loc=q_set[active_idx], scale=0.03 * q_set[active_idx])
        )
    if idle_idx.size:
        q_values[idle_idx] = ensure_non_negative(
            rng.normal(loc=0.3, scale=0.2, size=idle_idx.size)
        )
    df["Q_m3h"] = q_values

    ps_values = np.zeros(df.shape[0])
    if active_idx.size:
        ps_values[active_idx] = rng.normal(loc=-0.5, scale=2.0, size=active_idx.size)
    if idle_idx.size:
        ps_values[idle_idx] = rng.normal(loc=0.0, scale=0.5, size=idle_idx.size)
    df["Ps_kPa"] = ps_values

    df["dP_expected"] = dP_expected_from_flow(df["Q_m3h"].to_numpy())
    df.loc[~df["is_active"], "dP_expected"] = 0.0

    sigma = np.zeros(df.shape[0])
    if active_idx.size:
        sigma[active_idx] = np.maximum(0.015 * df.loc[df["is_active"], "dP_expected"].to_numpy(), 1.0)
    if idle_idx.size:
        sigma[idle_idx] = 0.5
    df["sigma_dP"] = sigma

    df["dP_kPa"] = df["dP_expected"] + rng.normal(0.0, sigma)

    eff_targets = np.ones(df.shape[0])
    if active_idx.size:
        eff_targets[active_idx] = np.clip(
            rng.normal(loc=1.0, scale=0.03, size=active_idx.size),
            0.7,
            1.3,
        )

    i_values = np.zeros(df.shape[0]) + 0.5
    if active_idx.size:
        i_values[active_idx] = (
            df.loc[df["is_active"], "Q_m3h"].to_numpy()
            * df.loc[df["is_active"], "dP_kPa"].to_numpy()
        ) / eff_targets[active_idx]
    if idle_idx.size:
        i_values[idle_idx] = ensure_non_negative(
            rng.normal(loc=2.0, scale=1.0, size=idle_idx.size)
        )
    df["I_A"] = i_values

    df["Eff"] = 0.0
    recompute_power_signals(df)

    return df


def apply_cavitation(
    df: pd.DataFrame, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Overlay cavitation effects onto a baseline profile."""

    cavitation_start_day = int(rng.integers(18, 26))
    cavitation_start = df.index.min() + pd.Timedelta(days=cavitation_start_day)
    active_mask = df["is_active"].to_numpy()
    cavitation_mask = active_mask & (df.index >= cavitation_start)

    if cavitation_mask.any():
        elapsed_days = (
            (df.index[cavitation_mask] - cavitation_start) / np.timedelta64(1, "D")
        ).astype(float)
        elapsed_days = np.maximum(elapsed_days, 0.0)
        progress = elapsed_days / max(elapsed_days.max(), 1.0)
        elapsed_series = pd.Series(elapsed_days, index=df.index[cavitation_mask])
        progress_series = pd.Series(progress, index=df.index[cavitation_mask])

        baseline_mask = active_mask & (df.index < cavitation_start)
        ps_baseline = float(
            df.loc[baseline_mask, "Ps_kPa"].median()
        ) if baseline_mask.any() else -1.5
        baseline_abs = max(abs(ps_baseline), 3.0)
        suction_drift_rate = rng.uniform(0.2, 0.5)
        target_drop = baseline_abs * rng.uniform(0.08, 0.15)
        ps_drop = suction_drift_rate * elapsed_series.to_numpy()
        max_drop_profile = target_drop * progress_series.to_numpy()
        ps_drop = np.minimum(ps_drop, max_drop_profile)
        df.loc[cavitation_mask, "Ps_kPa"] = (
            df.loc[cavitation_mask, "Ps_kPa"].to_numpy() - ps_drop
        )
        df.loc[cavitation_mask, "Ps_kPa"] = np.clip(
            df.loc[cavitation_mask, "Ps_kPa"].to_numpy(),
            -45.0,
            5.0,
        )

        variance_factor = rng.uniform(1.5, 2.0)
        df.loc[cavitation_mask, "sigma_dP"] = np.maximum(
            df.loc[cavitation_mask, "sigma_dP"].to_numpy() * variance_factor,
            1.0,
        )

        dp_decline = rng.uniform(0.08, 0.15)

        chronic_flow_loss = rng.uniform(0.02, 0.05)
        df.loc[cavitation_mask, "Q_m3h"] = ensure_non_negative(
            df.loc[cavitation_mask, "Q_m3h"].to_numpy()
            * (1.0 - chronic_flow_loss * progress_series.to_numpy())
        )

        current_increase = rng.uniform(0.10, 0.18)
        df.loc[cavitation_mask, "I_A"] = (
            df.loc[cavitation_mask, "I_A"].to_numpy()
            * (1.0 + current_increase * progress_series.to_numpy())
        )

        cavitation_indices = np.where(cavitation_mask)[0]
        num_events = int(min(rng.integers(30, 45), cavitation_indices.size))
        if num_events > 0:
            selected = rng.choice(cavitation_indices, size=num_events, replace=False)
            for start_idx in selected:
                duration_steps = int(rng.integers(1, 4))
                reduction = rng.uniform(0.05, 0.12)
                spike = rng.uniform(0.08, 0.15)
                event_idx = np.arange(start_idx, min(start_idx + duration_steps, df.shape[0]))
                event_idx = event_idx[df.iloc[event_idx]["is_active"].to_numpy()]
                if event_idx.size == 0:
                    continue
                df.iloc[event_idx, df.columns.get_loc("Q_m3h")] *= 1.0 - reduction
                df.iloc[event_idx, df.columns.get_loc("I_A")] *= 1.0 + spike

        active_positions = np.where(active_mask)[0]
        if active_positions.size:
            df.iloc[active_positions, df.columns.get_loc("dP_expected")] = dP_expected_from_flow(
                df.iloc[active_positions]["Q_m3h"].to_numpy()
            )
        df.loc[~df["is_active"], "dP_expected"] = 0.0

        df.loc[cavitation_mask, "dP_expected"] *= 1.0 - dp_decline * progress_series

        df.loc[cavitation_mask, "dP_kPa"] = df.loc[cavitation_mask, "dP_expected"] + rng.normal(
            0.0, df.loc[cavitation_mask, "sigma_dP"].to_numpy()
        )

    return df, cavitation_start


def apply_impeller_wear(
    df: pd.DataFrame, config: SimulationConfig, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Overlay impeller wear effects (slow efficiency decline)."""

    active_mask = df["is_active"].to_numpy()
    baseline_sigma = df.loc[active_mask, "sigma_dP"].to_numpy(copy=True)
    progress = (df.index - df.index.min()) / np.timedelta64(1, "D")
    progress = np.clip(progress / config.days, 0.0, 1.0)

    flow_drop = rng.uniform(0.02, 0.10)
    df.loc[active_mask, "Q_m3h"] *= 1.0 - flow_drop * progress[active_mask]

    if active_mask.any():
        df.loc[active_mask, "dP_expected"] = dP_expected_from_flow(
            df.loc[active_mask, "Q_m3h"].to_numpy()
        )
        decline_pct = rng.uniform(0.05, 0.20)
        df.loc[active_mask, "dP_expected"] *= 1.0 - decline_pct * progress[active_mask]
        df.loc[active_mask, "sigma_dP"] = baseline_sigma
        df.loc[active_mask, "dP_kPa"] = df.loc[active_mask, "dP_expected"] + rng.normal(
            0.0, baseline_sigma
        )

        current_increase = rng.uniform(0.05, 0.10)
        df.loc[active_mask, "I_A"] *= 1.0 + current_increase * progress[active_mask]

    df.loc[~df["is_active"], "dP_expected"] = 0.0

    return df, df.index.min()


def apply_healthy_epochs(
    df: pd.DataFrame, config: SimulationConfig, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Apply benign process / recipe shifts across multiple epochs."""

    day = 0
    active_mask = df["is_active"].to_numpy()
    while day < config.days:
        span_days = int(min(rng.integers(7, 15), config.days - day))
        start = df.index.min() + pd.Timedelta(days=day)
        end = start + pd.Timedelta(days=span_days)
        mask = active_mask & (df.index >= start) & (df.index < end)
        if mask.any():
            q_shift = rng.uniform(0.05, 0.10)
            if rng.random() < 0.5:
                q_shift *= -1
            dp_shift = rng.uniform(0.03, 0.06)
            if rng.random() < 0.5:
                dp_shift *= -1

            df.loc[mask, "Q_m3h"] *= 1.0 + q_shift
            df.loc[mask, "Q_set_m3h"] *= 1.0 + q_shift

            df.loc[mask, "dP_expected"] = dP_expected_from_flow(
                df.loc[mask, "Q_m3h"].to_numpy()
            )
            df.loc[mask, "dP_expected"] *= 1.0 + dp_shift

            sigma = np.maximum(0.015 * df.loc[mask, "dP_expected"], 1.0)
            df.loc[mask, "sigma_dP"] = sigma
            df.loc[mask, "dP_kPa"] = df.loc[mask, "dP_expected"] + rng.normal(
                0.0, sigma.to_numpy()
            )

            ps_shift = rng.normal(loc=0.0, scale=0.8)
            df.loc[mask, "Ps_kPa"] += ps_shift

            eff_targets = np.clip(
                rng.normal(loc=1.0, scale=0.04, size=int(mask.sum())),
                0.85,
                1.2,
            )
            df.loc[mask, "I_A"] = (
                df.loc[mask, "Q_m3h"] * df.loc[mask, "dP_kPa"]
            ) / eff_targets

        day += span_days

    df.loc[~df["is_active"], "dP_expected"] = 0.0
    return df, df.index.min()


def simulate_single_pump(
    pump_type: str, index: int, config: SimulationConfig, rng: np.random.Generator
) -> PumpSimulationResult:
    """Simulate one pump for the requested class."""

    pump_id = f"{PUMP_CLASSES[pump_type]}_{index + 1:02d}"
    df = build_baseline_profile(pump_id, config, rng)

    if pump_type == "cavitation":
        df, class_start = apply_cavitation(df, rng)
        class_label = "CAVITATION"
    elif pump_type == "impeller_wear":
        df, class_start = apply_impeller_wear(df, config, rng)
        class_label = "IMPELLER_WEAR"
    elif pump_type == "healthy_fp":
        df, class_start = apply_healthy_epochs(df, config, rng)
        class_label = "HEALTHY_FP"
    else:
        raise ValueError(f"Unsupported pump type: {pump_type}")

    recompute_power_signals(df)

    final_df = df.reset_index()
    final_df["batch_id"] = final_df["batch_id"].fillna("")
    final_df["recipe"] = final_df["recipe"].fillna("")

    telemetry = final_df[
        [
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
    ].copy()

    labels = [
        {
            "pump_id": pump_id,
            "class": class_label,
            "class_start_utc": class_start.tz_convert("UTC") if class_start.tzinfo else class_start,
            "class_end_utc": df.index.max(),
        }
    ]

    return PumpSimulationResult(data=telemetry, labels=labels)


def simulate_pumps(config: SimulationConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the simulation for the configured pump fleet."""

    rng = np.random.default_rng(config.random_seed)
    data_frames: List[pd.DataFrame] = []
    label_rows: List[Dict[str, object]] = []

    for pump_type, count in config.pump_counts.items():
        for idx in range(count):
            result = simulate_single_pump(pump_type, idx, config, rng)
            data_frames.append(result.data)
            label_rows.extend(result.labels)

    timeseries = pd.concat(data_frames, ignore_index=True).sort_values("timestamp_utc")
    label_df = pd.DataFrame(label_rows)
    return timeseries, label_df


def write_outputs(timeseries: pd.DataFrame, labels: pd.DataFrame) -> Tuple[Path, Path]:
    """Persist the simulation outputs to CSV files."""

    timeseries_path = SIM_OUTPUT_DIR / "pump_timeseries.csv"
    labels_path = SIM_OUTPUT_DIR / "pump_labels.csv"
    timeseries.to_csv(timeseries_path, index=False)
    labels.to_csv(labels_path, index=False)
    return timeseries_path, labels_path


def parse_args() -> SimulationConfig:
    """Parse CLI arguments into a :class:`SimulationConfig`."""

    parser = argparse.ArgumentParser(description="Simulate centrifugal pump telemetry datasets.")
    parser.add_argument("--days", type=int, default=90, help="Number of days to simulate (default: 90)")
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01T00:00:00Z",
        help="UTC start timestamp for the simulation window.",
    )
    parser.add_argument("--freq", type=int, default=5, help="Data cadence in minutes (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--cavitation", type=int, default=1, help="Number of cavitation pumps to simulate")
    parser.add_argument(
        "--impeller-wear", type=int, default=1, help="Number of impeller-wear pumps to simulate"
    )
    parser.add_argument(
        "--healthy", type=int, default=1, help="Number of healthy/false-positive pumps to simulate"
    )

    args = parser.parse_args()
    pump_counts = {
        "cavitation": max(0, args.cavitation),
        "impeller_wear": max(0, args.impeller_wear),
        "healthy_fp": max(0, args.healthy),
    }

    return SimulationConfig(
        start_timestamp=pd.Timestamp(args.start, tz="UTC"),
        days=args.days,
        freq_minutes=args.freq,
        pump_counts=pump_counts,
        random_seed=args.seed,
    )


def main() -> None:
    """Entry point for CLI usage."""

    config = parse_args()
    timeseries, labels = simulate_pumps(config)
    ts_path, label_path = write_outputs(timeseries, labels)
    print(f"Wrote telemetry to {ts_path.relative_to(SIM_OUTPUT_DIR.parent)}")
    print(f"Wrote labels to {label_path.relative_to(SIM_OUTPUT_DIR.parent)}")


if __name__ == "__main__":
    main()

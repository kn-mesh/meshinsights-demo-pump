"""Pump-specific data retriever built on the core :class:`DataSource` API.

The pump simulator ships a canonical CSV under ``src/pump_data/pump_timeseries.csv``
containing 5-minute telemetry for each simulated asset. This module exposes a
small DataSource implementation that wires that CSV through the MeshInsights
plugin layer (``local_file`` connector) and returns the raw, per-device slice
expected by the Stage 1 I/O pipeline.

Responsibilities handled here:

* Resolve and validate the simulator CSV path.
* Configure the ``local_file`` plugin to parse timestamps eagerly and keep
  identifier columns as strings (preserve leading zeros).
* Filter the loaded DataFrame down to the requested ``pump_id`` and optional
  time window while keeping the dataset in raw form for the normalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

from src.core.data_source import DataSource
from src.core.plugins.data_access.data_access_plugin_core import PluginManager
from src.core.plugins.data_access.implementations.local_file_plugin import LocalFileConnector


def _resolve_default_dataset() -> Path:
    """Return the absolute path to the packaged pump time-series CSV."""

    return Path(__file__).resolve().parents[2] / "pump_data" / "pump_timeseries.csv"


@dataclass(frozen=True)
class PumpDataWindow:
    """Optional inclusive time window used to trim raw records."""

    start_utc: Optional[pd.Timestamp] = None
    end_utc: Optional[pd.Timestamp] = None

    @staticmethod
    def coerce(value: Optional[datetime | str | pd.Timestamp]) -> Optional[pd.Timestamp]:
        """Convert incoming timestamp hints to timezone-aware UTC pandas timestamps."""

        if value is None:
            return None

        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp

    @classmethod
    def from_bounds(
        cls,
        start: Optional[datetime | str | pd.Timestamp] = None,
        end: Optional[datetime | str | pd.Timestamp] = None,
    ) -> "PumpDataWindow":
        return cls(start_utc=cls.coerce(start), end_utc=cls.coerce(end))


class PumpDataRetriever(DataSource):
    """Concrete DataSource that loads the simulator CSV for a single pump."""

    DEFAULT_PANDAS_KWARGS: Dict[str, Any] = {
        "parse_dates": ["timestamp_utc"],
        "dtype": {
            "pump_id": "string",
            "batch_id": "string",
            "recipe": "string",
        },
    }

    def __init__(
        self,
        pump_id: str,
        *,
        file_path: Optional[str | Path] = None,
        start_utc: Optional[datetime | str | pd.Timestamp] = None,
        end_utc: Optional[datetime | str | pd.Timestamp] = None,
    ) -> None:
        if not pump_id:
            raise ValueError("PumpDataRetriever requires a non-empty pump_id")

        # Ensure local_file plugin is registered (idempotent guard)
        if "local_file" not in PluginManager.list_plugins():
            PluginManager.register_plugin("local_file", LocalFileConnector)

        dataset_path = Path(file_path) if file_path is not None else _resolve_default_dataset()
        dataset_path = dataset_path.expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Pump dataset not found at {dataset_path}")

        self.pump_id: str = str(pump_id)
        self.window = PumpDataWindow.from_bounds(start=start_utc, end=end_utc)

        config = {
            "pump_id": self.pump_id,
            "file_path": str(dataset_path),
        }
        if self.window.start_utc is not None:
            config["start_utc"] = self.window.start_utc.isoformat()
        if self.window.end_utc is not None:
            config["end_utc"] = self.window.end_utc.isoformat()

        plugin_kwargs = {
            "file_path": str(dataset_path),
            "file_format": "csv",
            "pandas_read_kwargs": {**self.DEFAULT_PANDAS_KWARGS},
        }

        super().__init__(
            name=f"PumpDataRetriever[{self.pump_id}]",
            plugin_name="local_file",
            config=config,
            plugin_kwargs=plugin_kwargs,
        )

    # ===== DataSource interface =====
    def build_query(self) -> str:
        """Return an empty query to load the full CSV through the plugin."""

        return ""

    def fetch(self) -> pd.DataFrame:
        """
        Load the simulator CSV and trim it to the configured pump and window.

        Returns:
            pandas.DataFrame: Raw slice corresponding to ``pump_id`` within the
            optional time bounds. Columns remain unaltered to keep normalization
            concerns out of the retriever.
        """

        # Ensure connector is ready; local file connector is idempotent here.
        self.plugin.connect()

        raw_df = self._execute(self.build_query())
        if raw_df.empty:
            self.logger.warning("PumpDataRetriever returned an empty frame before filtering")
            return raw_df

        df = raw_df.copy()

        # Guarantee timestamp typing; CSV parse_dates should already cover this.
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

        df["pump_id"] = df["pump_id"].astype("string")

        mask = df["pump_id"] == self.pump_id
        if mask.any():
            df = df.loc[mask]
        else:
            self.logger.warning("No records found for pump_id %s", self.pump_id)
            df = df.iloc[0:0]

        if df.empty:
            return df.reset_index(drop=True)

        if self.window.start_utc is not None:
            df = df.loc[df["timestamp_utc"] >= self.window.start_utc]

        if self.window.end_utc is not None:
            df = df.loc[df["timestamp_utc"] <= self.window.end_utc]

        df = df.reset_index(drop=True)
        self.config["retrieved_rows"] = len(df)
        return df

# uv run python -m src.pump_pipeline.stage_io.data_retriever_pump
if __name__ == "__main__":

    import logging
    from datetime import date

    logging.basicConfig(level=logging.WARNING)

    SAMPLE_PUMP_ID = "089250"
    dr = PumpDataRetriever(pump_id=SAMPLE_PUMP_ID)
    df = dr.fetch()
    print(f"Retrieved rows for pump {SAMPLE_PUMP_ID}: {len(df)}")
    if not df.empty:
        print(df.head(10).to_string(index=False))

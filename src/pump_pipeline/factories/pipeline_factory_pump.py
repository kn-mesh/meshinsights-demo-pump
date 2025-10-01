from __future__ import annotations

"""
Pump pipeline wiring using the generic two-stage `Pipeline`.

This module composes the pump demo solution (CSV data source, normalizer,
domain data object) with the core batch pipeline via injected factories.

- Factory functions to create components per `DeviceUnit`
- DataObject factory to produce `PumpPipelineDataObject`
- Processor chain is intentionally stubbed (no processors implemented yet)
- Convenience helpers to build devices and run single/batch analyses
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.pipeline import (
    Pipeline,
    PipelineConfig,
    DeviceUnit,
    BatchResults,
)
from src.core.data_source import DataSource
from src.core.data_normalizer import DataNormalizer
from src.core.processor import Processor

from src.pump_pipeline.pipeline_objects.data_object_pump import PumpPipelineDataObject
from src.pump_pipeline.stage_io.data_retriever_pump import PumpDataRetriever
from src.pump_pipeline.stage_io.data_normalizer_pump import PumpDataNormalizer
from src.pump_pipeline.stage_cpu.processors.calculate_differential_pressure import CalculateDifferentialPressureProcessor
from src.pump_pipeline.stage_cpu.processors.calculate_efficiency import CalculateEfficiencyProcessor
from src.pump_pipeline.stage_cpu.processors.calculate_head_spread_ratio import CalculateHeadSpreadRatioProcessor
from src.pump_pipeline.stage_cpu.processors.calculate_head_trend_slope import CalculateHeadTrendSlopeProcessor
from src.pump_pipeline.stage_cpu.processors.calculate_intermittent_event_rate import CalculateIntermittentEventRateProcessor



# ==================== Module defaults (to avoid drift) ==================== #

DEFAULT_PUMP_BATCH_SIZE: int = 10
DEFAULT_PUMP_IO_MAX_WORKERS: int = 4
DEFAULT_PUMP_CPU_MAX_WORKERS: int = 2


# ==================== Factory functions (per-device) ==================== #

def create_pump_data_sources(device: DeviceUnit) -> Dict[str, DataSource]:
    """Create pump data sources for a specific device.

    Parameters
    ----------
    device : DeviceUnit
        Device descriptor. `device.device_id` is treated as the pump id.
        Optional parameters accepted:
        - `file_path`: Override CSV path
        - `start_utc`, `end_utc`: Inclusive time bounds (datetime | str | pd.Timestamp)

    Returns
    -------
    Dict[str, DataSource]
        Mapping of dataset name to DataSource. Uses key `"pump_telemetry"`.
    """
    params = device.parameters or {}
    pump_id = str(params.get("pump_id", device.device_id))

    return {
        "pump_telemetry": PumpDataRetriever(
            pump_id=pump_id,
            file_path=params.get("file_path"),
            start_utc=params.get("start_utc"),
            end_utc=params.get("end_utc"),
        )
    }


def create_pump_normalizers(device: DeviceUnit) -> Dict[str, DataNormalizer]:
    """Create pump normalizers for a specific device.

    Returns
    -------
    Dict[str, DataNormalizer]
        Mapping for dataset key `"pump_telemetry"` to `PumpDataNormalizer`.
    """
    return {"pump_telemetry": PumpDataNormalizer()}


def create_pump_processors(device: DeviceUnit) -> List[Processor]:
    """Create the ordered list of pump processors for a specific device.

    Processors execute in sequence on the per-device ``PumpPipelineDataObject``
    and populate canonical artifacts such as differential pressure, efficiency,
    cavitation head metrics, and intermittent event rates.

    Returns
    -------
    List[Processor]
        Ordered processor chain for CPU stage analytics.
    """
    return [
        CalculateDifferentialPressureProcessor(),
        CalculateEfficiencyProcessor(),
        CalculateHeadSpreadRatioProcessor(),
        CalculateHeadTrendSlopeProcessor(),
        CalculateIntermittentEventRateProcessor(),
    ]

# ==================== DataObject factory ==================== #

def create_pump_data_object(
    device_id: str, pipeline_name: str, metadata: Dict[str, Any]
) -> PumpPipelineDataObject:
    """Create a pump-specific `PipelineDataObject`.

    Parameters
    ----------
    device_id : str
        Device identifier (pump id).
    pipeline_name : str
        Logical pipeline name.
    metadata : Dict[str, Any]
        Additional metadata to include in the configuration.

    Returns
    -------
    PumpPipelineDataObject
        Domain-specific data object with pump semantics and artifact schemas.
    """
    return PumpPipelineDataObject(
        pipeline_name=f"{pipeline_name}_{device_id}",
        config={"device_id": device_id, "pipeline_type": "pump", **(metadata or {})},
    )


# ==================== Public helpers to build and run ==================== #

def create_pump_batch_pipeline(
    *,
    pipeline_name: str = "pump_demo",
    stop_on_error: bool = True,
    log_level: str = "INFO",
    batch_size: int = DEFAULT_PUMP_BATCH_SIZE,
    freeze_normalized_data: bool = True,
    io_max_workers: int = DEFAULT_PUMP_IO_MAX_WORKERS,
    cpu_max_workers: int = DEFAULT_PUMP_CPU_MAX_WORKERS,
) -> Pipeline[PumpPipelineDataObject]:
    """Create a pump batch pipeline configured for the simulator dataset.

    Parameters
    ----------
    pipeline_name : str
        Logical pipeline name.
    stop_on_error : bool
        Abort on first error if True; otherwise collect errors and continue.
    log_level : str
        Root logging level.
    batch_size : int
        Devices per batch.
    freeze_normalized_data : bool
        Whether to freeze normalized data to prevent modification.
    io_max_workers : int
        Maximum number of concurrent workers for I/O-bound tasks.
    cpu_max_workers : int
        Maximum number of concurrent workers for CPU-bound per-device processing.

    Returns
    -------
    Pipeline[PumpPipelineDataObject]
        Configured pipeline that accepts a list of `DeviceUnit`.
    """
    config = PipelineConfig(
        pipeline_name=pipeline_name,
        stop_on_error=stop_on_error,
        log_level=log_level,
        batch_size=batch_size,
        freeze_normalized_data=freeze_normalized_data,
        io_max_workers=io_max_workers,
        cpu_max_workers=cpu_max_workers,
    )
    return Pipeline[PumpPipelineDataObject](
        data_source_factory=create_pump_data_sources,
        normalizer_factory=create_pump_normalizers,
        processor_factory=create_pump_processors,
        data_object_factory=create_pump_data_object,
        results_adapter=None,
        config=config,
    )


def make_pump_device(
    *,
    device_id: str,
    start_utc: Optional[datetime] = None,
    end_utc: Optional[datetime] = None,
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DeviceUnit:
    """Helper to build a `DeviceUnit` for the pump simulator.

    Parameters
    ----------
    device_id : str
        Pump identifier (matches `pump_id` in the CSV dataset).
    start_utc, end_utc : Optional[datetime]
        Inclusive bounds for trimming the time series.
    file_path : Optional[str]
        Optional override path to the CSV dataset.
    metadata : Optional[Dict[str, Any]]
        Additional metadata to attach to the device.
    """
    return DeviceUnit(
        device_id=device_id,
        parameters={
            "pump_id": device_id,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "file_path": file_path,
        },
        metadata=metadata or {},
    )


def run_pump_batch(
    devices: List[DeviceUnit],
    *,
    pipeline_name: str = "pump_demo",
    stop_on_error: bool = True,
    log_level: str = "WARNING",
    batch_size: int = DEFAULT_PUMP_BATCH_SIZE,
    io_max_workers: int = DEFAULT_PUMP_IO_MAX_WORKERS,
    cpu_max_workers: int = DEFAULT_PUMP_CPU_MAX_WORKERS,
) -> BatchResults[PumpPipelineDataObject]:
    """Run pump analysis for all provided devices and return aggregated results.

    Returns
    -------
    BatchResults[PumpPipelineDataObject]
        One aggregated `BatchResults` covering all internal batches.
    """
    pipeline = create_pump_batch_pipeline(
        pipeline_name=pipeline_name,
        stop_on_error=stop_on_error,
        log_level=log_level,
        batch_size=batch_size,
        io_max_workers=io_max_workers,
        cpu_max_workers=cpu_max_workers,
    )
    return pipeline.run_producer_consumer(devices)

# local run
# uv run python -m src.pump_pipeline.factories.pipeline_factory_pump
if __name__ == "__main__":
    import logging
    from datetime import date as _date
    import json

    DEVICE_ID_1 = "089250" # CAVITATION
    DEVICE_ID_2 = "418496" # IMPELLER_WEAR
    DEVICE_ID_3 = "923684" # HEALTHY_FP
    START_DATE = _date(2025, 1, 1)
    END_DATE = _date(2025, 3, 31)

    logging.basicConfig(level=logging.WARNING)

    devices = [
        make_pump_device(device_id=DEVICE_ID_1, start_utc=datetime.combine(START_DATE, datetime.min.time()), end_utc=datetime.combine(END_DATE, datetime.max.time())),
        make_pump_device(device_id=DEVICE_ID_2, start_utc=datetime.combine(START_DATE, datetime.min.time()), end_utc=datetime.combine(END_DATE, datetime.max.time())),
        make_pump_device(device_id=DEVICE_ID_3, start_utc=datetime.combine(START_DATE, datetime.min.time()), end_utc=datetime.combine(END_DATE, datetime.max.time())),
    ]

    batch_results = run_pump_batch(
        devices,
        batch_size=DEFAULT_PUMP_BATCH_SIZE,
        io_max_workers=DEFAULT_PUMP_IO_MAX_WORKERS,
        cpu_max_workers=DEFAULT_PUMP_CPU_MAX_WORKERS,
    )

    print("\n" + "=" * 50)
    print("PUMP DEMO - NORMALIZED DATA PREVIEW")
    print("Devices processed:", batch_results.total_devices)
    print("Successful:", batch_results.successful_devices)
    print("Failed:", batch_results.failed_devices)
    print("=" * 50)

    for device_id, dobj in batch_results.device_results.items():
        normalized_df = dobj.get_dataset("pump_telemetry")
        differential_pressure_df = dobj.get_artifact("differential_pressure")
        efficiency_df = dobj.get_artifact("efficiency")
        head_spread_ratio_df = dobj.get_artifact("head_spread_ratio")
        head_trend_slope_df = dobj.get_artifact("head_trend_slope")
        intermittent_event_rate_df = dobj.get_artifact("intermittent_event_rate")

        print("\n" + "=" * 50)
        print(f"\nDevice: {device_id}")
        if device_id == DEVICE_ID_1:
            print("\nCAVITATION")
        elif device_id == DEVICE_ID_2:
            print("\nIMPELLER_WEAR")
        elif device_id == DEVICE_ID_3:
            print("\nHEALTHY_FP")
        # sort by batch_id
        normalized_df = normalized_df.sort_values("batch_id")
        print("\nNormalized DataFrame:")
        print(normalized_df)
        print("\nDifferential Pressure DataFrame:")
        print(differential_pressure_df)
        print("\nEfficiency DataFrame:")
        print(efficiency_df)
        print("\nHead Spread Ratio DataFrame:")
        print(head_spread_ratio_df)
        print("\nHead Trend Slope DataFrame:")
        print(head_trend_slope_df)
        print("\nIntermittent Event Rate DataFrame:")
        print(intermittent_event_rate_df)
        print("=" * 50)

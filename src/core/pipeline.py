# src/core/pipeline.py
from __future__ import annotations

from typing import (
    List,
    Optional,
    Dict,
    Any,
    Protocol,
    Generic,
    TypeVar,
    Tuple,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from uuid import uuid4
import time
from queue import Queue
from threading import Thread, Event, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from src.core.data_object import PipelineDataObject
from src.core.data_source import DataSource
from src.core.data_normalizer import DataNormalizer
from src.core.processor import Processor
from src.core.plugins.data_access.data_access_plugin_core import PluginManager


# ==================== Protocols / Factories ====================

class DataSourceFactory(Protocol):
    """
    Protocol for factory functions that create data sources per device.

    Args:
        device (DeviceUnit): The device to create sources for.

    Returns:
        Dict[str, DataSource]: Mapping of dataset name to `DataSource`.
    """
    def __call__(self, device: "DeviceUnit") -> Dict[str, DataSource]:
        ...


class DataNormalizerFactory(Protocol):
    """
    Protocol for factory functions that create data normalizers per device.

    Args:
        device (DeviceUnit): The device to create normalizers for.

    Returns:
        Dict[str, DataNormalizer]: Mapping of dataset name to `DataNormalizer`.
    """
    def __call__(self, device: "DeviceUnit") -> Dict[str, DataNormalizer]:
        ...


class ProcessorFactory(Protocol):
    """
    Protocol for factory functions that create an ordered list of processors per device.

    Args:
        device (DeviceUnit): The device to create processors for.

    Returns:
        List[Processor]: Ordered list of `Processor` instances.
    """
    def __call__(self, device: "DeviceUnit") -> List[Processor]:
        ...


TDO = TypeVar("TDO", bound=PipelineDataObject)


class DataObjectFactory(Protocol, Generic[TDO]):
    """
    Protocol for creating the per-device PipelineDataObject (or subclass).

    Implementations may inject domain context and shape-specific defaults.

    Args:
        device_id (str): Identifier of the device.
        pipeline_name (str): Logical pipeline name.
        metadata (Dict[str, Any]): Additional metadata to attach to the object.

    Returns:
        TDO: A PipelineDataObject (or subclass) instance for this device.
    """
    def __call__(self, device_id: str, pipeline_name: str, metadata: Dict[str, Any]) -> TDO:
        ...


class ResultsAdapter(Protocol, Generic[TDO]):
    """
    Adapter for exporting/summarizing batch results without baking shapes
    into the pipeline. Keep this tiny and composable.

    Methods:
        summarize(results): Summary as a JSON-serializable dict.
        to_dataframe(results): Flat view as a pandas DataFrame.
    """
    def summarize(self, results: "BatchResults[TDO]") -> Dict[str, Any]:
        ...

    def to_dataframe(self, results: "BatchResults[TDO]") -> pd.DataFrame:
        ...


# ==================== Configuration & Core Data Types ====================

@dataclass
class PipelineConfig:
    """
    Configuration options for pipeline execution.

    Attributes:
        pipeline_name (str): Human-readable name for the pipeline.
        stop_on_error (bool): If True, abort on first error; otherwise continue
            and collect errors in the data object.
        log_level (str): Root logging level for pipeline execution.
        batch_size (int): Number of devices to process in each batch.
        freeze_normalized_data (bool): Make normalized_data read-only for processors.
        io_max_workers (int): Max concurrent device I/O workers (default 1 = serial).
        cpu_max_workers (int): Max concurrent device CPU workers (default 1 = serial).
    """
    pipeline_name: str = "default_pipeline"
    stop_on_error: bool = True
    log_level: str = "INFO"
    batch_size: int = 25
    freeze_normalized_data: bool = True
    io_max_workers: int = 1
    cpu_max_workers: int = 1


@dataclass
class DeviceUnit:
    """
    Represents a single device/unit to be analyzed.

    Attributes:
        device_id: Unique identifier for this device.
        parameters: Device-specific parameters (e.g., location, date range).
        metadata: Optional metadata about the device.
    """
    device_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IOBatch:
    """
    Results from the I/O stage for a batch of devices.
    Contains normalized data ready for CPU processing.

    Attributes:
        batch_id: Unique id for this I/O batch.
        timestamp: UTC timestamp at creation.
        device_data: Mapping device_id -> dataset_name -> normalized DataFrame.
        device_metadata: Mapping device_id -> dataset_name -> metadata dict.
        errors: Mapping device_id -> list of I/O error strings.
        warnings: Mapping device_id -> list of I/O warning strings.
    """
    batch_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    device_data: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    device_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: Dict[str, List[str]] = field(default_factory=dict)
    warnings: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class BatchResults(Generic[TDO]):
    """
    Final aggregated results from processing a batch of devices.

    Attributes:
        batch_id: Unique id for this batch processing round.
        start_time: UTC time when batch started.
        end_time: UTC time when batch ended.
        total_devices: Number of devices attempted.
        successful_devices: Number of devices without CPU-stage fatal errors.
        failed_devices: Number of devices that failed CPU-stage.
        device_results: Mapping device_id -> data object (domain-specific or base).
        io_timing: Timing details for I/O stage.
        cpu_timing: Timing details for CPU stage.
    """
    batch_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    total_devices: int = 0
    successful_devices: int = 0
    failed_devices: int = 0
    device_results: Dict[str, TDO] = field(default_factory=dict)
    io_timing: Dict[str, float] = field(default_factory=dict)
    cpu_timing: Dict[str, float] = field(default_factory=dict)


# ==================== Pipeline ====================

class Pipeline(Generic[TDO]):
    """
    Two-stage batch processing pipeline.

    Responsibilities:
    - Stage 1 (I/O): Data acquisition + normalization per batch.
    - Stage 2 (CPU): Processing normalized data per device.
    - Batching, timing, and error aggregation.

    Extension seams (composition over inheritance):
    - Data source / normalizer / processor factories (per device).
    - Data object factory (per device) for domain-specific data objects.
    - Optional results adapter for exporting batch results.
    """

    def __init__(
        self,
        data_source_factory: Optional[DataSourceFactory] = None,
        normalizer_factory: Optional[DataNormalizerFactory] = None,
        processor_factory: Optional[ProcessorFactory] = None,
        data_object_factory: Optional[DataObjectFactory[TDO]] = None,
        results_adapter: Optional[ResultsAdapter[TDO]] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the pipeline with factory functions for batch processing.

        Parameters
        ----------
        data_source_factory : Optional[DataSourceFactory]
            Creates DataSource instances per device.
        normalizer_factory : Optional[DataNormalizerFactory]
            Creates DataNormalizer instances per device.
        processor_factory : Optional[ProcessorFactory]
            Creates ordered Processor list per device.
        data_object_factory : Optional[DataObjectFactory[TDO]]
            Creates per-device PipelineDataObject (or subclass). If omitted,
            a default base `PipelineDataObject` is created.
        results_adapter : Optional[ResultsAdapter[TDO]]
            Optional adapter for exporting/summarizing results.
        config : Optional[PipelineConfig]
            Pipeline configuration options.
        """
        # Factory-based configuration
        self.data_source_factory = data_source_factory
        self.normalizer_factory = normalizer_factory
        self.processor_factory = processor_factory
        self.data_object_factory: DataObjectFactory[TDO] = data_object_factory or self._default_data_object_factory()  # type: ignore[assignment]
        self.results_adapter = results_adapter

        # Shared configuration
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger("pipeline")

        # Set logging level for root logger (best-effort)
        logging.basicConfig(level=getattr(logging, self.config.log_level, logging.INFO))

        # Validate configuration via dedicated validator
        self._validate_configuration()

    @staticmethod
    def _default_data_object_factory() -> DataObjectFactory[PipelineDataObject]:
        """
        Default factory that constructs a basic PipelineDataObject.

        Returns
        -------
        DataObjectFactory[PipelineDataObject]
        """
        def _factory(device_id: str, pipeline_name: str, metadata: Dict[str, Any]) -> PipelineDataObject:
            return PipelineDataObject(
                pipeline_name=f"{pipeline_name}_{device_id}",
                correlation_id=str(uuid4()),
                config={"device_id": device_id, **(metadata or {})},
            )
        return _factory

    def _validate_configuration(self) -> None:
        """
        Validate factory-based configuration for the pipeline.

        Ensures that a data source factory is provided, and documents expectations
        for normalizer and processor strategies.

        Raises
        ------
        ValueError
            If required configuration is missing.
        """
        if self.data_source_factory is None:
            raise ValueError("data_source_factory must be provided")

        if self.processor_factory is None:
            self.logger.info(
                "No processor_factory provided; the pipeline will perform acquisition/normalization only"
            )

    @staticmethod
    def _chunk_devices(devices: List[DeviceUnit], size: int) -> List[List[DeviceUnit]]:
        """
        Split the given list of devices into contiguous chunks of at most `size`.

        Parameters
        ----------
        devices : List[DeviceUnit]
            Full device list to split.
        size : int
            Maximum number of devices per chunk; must be > 0.

        Returns
        -------
        List[List[DeviceUnit]]
            Contiguous sublists of devices with length up to `size`.
        """
        if size <= 0:
            raise ValueError("Chunk size must be > 0")
        return [devices[i:i + size] for i in range(0, len(devices), size)]

    # ==================== STAGE 1: I/O Operations (Acquisition + Normalization) ====================
    def run_producer_consumer(self, devices: List[DeviceUnit]) -> BatchResults[TDO]:
        """
        Run the pipeline using a producer–consumer model to overlap Stage 1 (I/O) with Stage 2 (CPU).

        The producer performs I/O serially per batch and enqueues the normalized results (IOBatch).
        The consumer dequeues each I/O batch and runs CPU processing serially for that batch.
        While the consumer is processing batch N, the producer is already fetching batch N+1.

        Parameters
        ----------
        devices : List[DeviceUnit]
            All devices to process.

        Returns
        -------
        BatchResults[TDO]
            Aggregated results and timing across all batches.
        """
        if not devices:
            return BatchResults()

        all_results: BatchResults[TDO] = BatchResults(total_devices=len(devices))
        batch_size = self.config.batch_size
        batches = self._chunk_devices(devices, batch_size)

        # We pass (batch_index, devices_sublist, io_batch, io_duration_seconds) through the queue.
        q: Queue[Tuple[int, List[DeviceUnit], IOBatch, float]] = Queue(maxsize=2)  # small buffer = double buffering
        stop_event = Event()
        lock = Lock()
        sentinel_index = -1  # sentinel to indicate completion
        thread_errors: List[BaseException] = []

        # Start real wall timer before launching threads
        overall_t0 = time.time()

        def _record_exception(exc: BaseException) -> None:
            with lock:
                thread_errors.append(exc)
            stop_event.set()

        def producer() -> None:
            try:
                for idx, batch in enumerate(batches, start=1):
                    if stop_event.is_set():
                        break
                    start = (idx - 1) * batch_size + 1
                    end = start + len(batch) - 1
                    self.logger.info(f"\n[Producer] Building I/O batch {idx}: devices {start}-{end}")

                    t0 = time.time()
                    io_batch = self.stage1_io_operations(batch)
                    io_duration = time.time() - t0
                    self.logger.info(f"[Producer] Batch {idx} I/O duration: {io_duration:.2f}s")

                    q.put((idx, batch, io_batch, io_duration))
            except BaseException as e:
                self.logger.error(f"[Producer] Unhandled exception: {e}")
                _record_exception(e)
            finally:
                # Signal consumer we're done (even on error)
                q.put((sentinel_index, [], IOBatch(), 0.0))

        def consumer() -> None:
            try:
                while True:
                    idx, batch, io_batch, io_duration = q.get()
                    if idx == sentinel_index:
                        break
                    self.logger.info(f"[Consumer] Received I/O batch {idx} with {len(io_batch.device_data)} devices")

                    t1 = time.time()
                    device_results = self.stage2_cpu_operations(io_batch, batch)
                    cpu_duration = time.time() - t1
                    self.logger.info(f"[Consumer] Batch {idx} CPU duration: {cpu_duration:.2f}s")

                    # Aggregate results in a single place under a lock.
                    succ = sum(
                        1
                        for do in device_results.values()
                        if not do.errors or not any("CPU Stage failure" in e for e in do.errors)
                    )
                    with lock:
                        all_results.device_results.update(device_results)
                        all_results.successful_devices += succ
                        all_results.failed_devices += (len(batch) - succ)
                        all_results.io_timing[f"batch_{idx}"] = io_duration
                        all_results.cpu_timing[f"batch_{idx}"] = cpu_duration

                    self.logger.info(
                        f"[Consumer] Batch {idx} complete "
                        f"(I/O: {io_duration:.2f}s, CPU: {cpu_duration:.2f}s, "
                        f"success {succ}/{len(batch)})"
                    )

                    if stop_event.is_set():
                        break
            except BaseException as e:
                self.logger.error(f"[Consumer] Unhandled exception: {e}")
                _record_exception(e)

        # Start threads
        prod = Thread(target=producer, name="pipeline-producer", daemon=True)
        cons = Thread(target=consumer, name="pipeline-consumer", daemon=True)
        all_results.start_time = datetime.now(timezone.utc)

        prod.start()
        cons.start()
        prod.join()
        cons.join()

        # Compute real wall time after both threads finish
        overall_wall = time.time() - overall_t0
        self.logger.info(f"Actual wall time (overlapped): {overall_wall:.2f}s")

        # After threads are done, propagate any fatal error if configured
        if thread_errors and self.config.stop_on_error:
            raise RuntimeError("Producer/Consumer failed") from thread_errors[0]

        # Compute totals
        all_results.io_timing["total"] = sum(
            v for k, v in all_results.io_timing.items() if k.startswith("batch_")
        )
        all_results.cpu_timing["total"] = sum(
            v for k, v in all_results.cpu_timing.items() if k.startswith("batch_")
        )
        all_results.end_time = datetime.now(timezone.utc)

        # Cleanup connections (best-effort)
        try:
            PluginManager.cleanup()
        except Exception:
            pass

        # Summary log
        total_time = all_results.io_timing.get("total", 0.0) + all_results.cpu_timing.get("total", 0.0)
        self.logger.info("=" * 60)
        self.logger.info("Producer-Consumer Run Complete")
        self.logger.info(f"  Total devices: {all_results.total_devices}")
        self.logger.info(f"  Successful: {all_results.successful_devices}")
        self.logger.info(f"  Failed: {all_results.failed_devices}")
        self.logger.info(f"  I/O total: {all_results.io_timing.get('total', 0.0):.2f}s")
        self.logger.info(f"  CPU total: {all_results.cpu_timing.get('total', 0.0):.2f}s")
        self.logger.info(f"  Total wall-ish (sum of stage times): {total_time:.2f}s")
        self.logger.info(f"Actual wall time (overlapped): {overall_wall:.2f}s")
        self.logger.info("=" * 60)

        return all_results

    def stage1_io_operations(self, devices: List[DeviceUnit]) -> IOBatch:
        """
        Perform I/O-bound work for a batch of devices: fetch + normalize.

        Parameters
        ----------
        devices : List[DeviceUnit]
            Devices to fetch and normalize data for.

        Returns
        -------
        IOBatch
            Normalized data, metadata, and I/O errors/warnings collected per device.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STAGE 1: I/O Operations for {len(devices)} devices")
        self.logger.info("=" * 60)

        io_batch = IOBatch()

        # Determine concurrency. Default (1) preserves current serial behavior.
        workers = max(1, min(self.config.io_max_workers, len(devices)))

        # === Serial path (backward-compatible default) ===
        if workers == 1:
            for device in devices:
                device_id = device.device_id
                self.logger.info(f"Processing I/O for device: {device_id}")

                try:
                    # Get data sources and normalizers for this device
                    data_sources = self.data_source_factory(device)  # type: ignore[operator]
                    normalizers = self.normalizer_factory(device) if self.normalizer_factory else {}

                    # Fetch and normalize each dataset
                    normalized_datasets: Dict[str, pd.DataFrame] = {}
                    device_metadata: Dict[str, Any] = {}

                    for dataset_name, source in data_sources.items():
                        try:
                            # Fetch raw data
                            self.logger.debug(f"  Fetching {dataset_name} for {device_id}")
                            raw_data = source.fetch()

                            # Store source metadata
                            device_metadata[dataset_name] = source.get_metadata()

                            # Normalize if normalizer exists
                            if dataset_name in normalizers:
                                normalizer = normalizers[dataset_name]
                                normalized_data = normalizer.normalize(
                                    raw_data,
                                    device_metadata[dataset_name],
                                )
                                normalized_datasets[dataset_name] = normalized_data
                                self.logger.debug(
                                    f"  Normalized {dataset_name}: "
                                    f"{len(raw_data)} -> {len(normalized_data)} rows"
                                )
                            else:
                                # No normalizer, use raw data
                                normalized_datasets[dataset_name] = raw_data
                                self.logger.debug(f"  No normalizer for {dataset_name}, using raw data")

                        except Exception as e:
                            error_msg = f"Failed to fetch/normalize {dataset_name}: {str(e)}"
                            self.logger.error(f"  {error_msg}")

                            if device_id not in io_batch.errors:
                                io_batch.errors[device_id] = []
                            io_batch.errors[device_id].append(error_msg)

                            if self.config.stop_on_error:
                                raise

                    # Store successful results
                    if normalized_datasets:
                        io_batch.device_data[device_id] = normalized_datasets
                        io_batch.device_metadata[device_id] = device_metadata
                        self.logger.info(f"✓ I/O complete for {device_id}: {len(normalized_datasets)} datasets")

                except Exception as e:
                    self.logger.error(f"Failed I/O for device {device_id}: {e}")
                    if device_id not in io_batch.errors:
                        io_batch.errors[device_id] = []
                    io_batch.errors[device_id].append(str(e))

                    if self.config.stop_on_error:
                        raise

            self.logger.info(
                f"Stage 1 complete: {len(io_batch.device_data)}/{len(devices)} devices successful"
            )

            return io_batch

        # === Concurrent path (per-device worker; datasets remain serial within a device) ===
        self.logger.info(f"I/O concurrency enabled: workers={workers}")

        def device_task(device: DeviceUnit) -> Tuple[str, Dict[str, pd.DataFrame], Dict[str, Any], List[str], List[str]]:
            """
            Perform fetch + normalize for a single device.

            Returns a tuple:
                (device_id, normalized_datasets, device_metadata, errors, warnings)
            """
            device_id = device.device_id
            self.logger.info(f"Processing I/O for device: {device_id}")

            errors: List[str] = []
            warnings: List[str] = []
            normalized_datasets: Dict[str, pd.DataFrame] = {}
            device_metadata: Dict[str, Any] = {}

            # Prepare per-device factories
            data_sources = self.data_source_factory(device)  # type: ignore[operator]
            normalizers = self.normalizer_factory(device) if self.normalizer_factory else {}

            for dataset_name, source in data_sources.items():
                try:
                    self.logger.debug(f"  Fetching {dataset_name} for {device_id}")
                    raw_data = source.fetch()

                    # Store source metadata
                    device_metadata[dataset_name] = source.get_metadata()

                    if dataset_name in normalizers:
                        normalizer = normalizers[dataset_name]
                        normalized = normalizer.normalize(raw_data, device_metadata[dataset_name])
                        normalized_datasets[dataset_name] = normalized
                        self.logger.debug(
                            f"  Normalized {dataset_name}: {len(raw_data)} -> {len(normalized)} rows"
                        )
                    else:
                        normalized_datasets[dataset_name] = raw_data
                        self.logger.debug(f"  No normalizer for {dataset_name}, using raw data")
                except Exception as e:
                    msg = f"Failed to fetch/normalize {dataset_name}: {str(e)}"
                    self.logger.error(f"  {msg}")
                    errors.append(msg)
                    if self.config.stop_on_error:
                        # Propagate to terminate Stage 1 early if configured
                        raise

            return (device_id, normalized_datasets, device_metadata, errors, warnings)

        # Submit all devices and collect as they complete
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for device in devices:
                futures.append(executor.submit(device_task, device))

            for fut in as_completed(futures):
                try:
                    device_id, normalized_datasets, device_metadata, errors, warnings = fut.result()

                    # Attach results
                    if normalized_datasets:
                        io_batch.device_data[device_id] = normalized_datasets
                        io_batch.device_metadata[device_id] = device_metadata
                        self.logger.info(f"✓ I/O complete for {device_id}: {len(normalized_datasets)} datasets")

                    if errors:
                        io_batch.errors.setdefault(device_id, []).extend(errors)
                    if warnings:
                        io_batch.warnings.setdefault(device_id, []).extend(warnings)

                except Exception as e:
                    # We don't know which device without mapping; include repr of future exception
                    self.logger.error(f"Failed I/O in concurrent worker: {e}")
                    # Best-effort attribution is not available; store under a synthetic key
                    synth_id = f"worker_error_{len(io_batch.errors) + 1}"
                    io_batch.errors.setdefault(synth_id, []).append(str(e))
                    if self.config.stop_on_error:
                        # Cancel pending futures and re-raise
                        for pending in futures:
                            pending.cancel()
                        raise

        self.logger.info(
            f"Stage 1 complete: {len(io_batch.device_data)}/{len(devices)} devices successful"
        )

        return io_batch

    # ==================== STAGE 2: CPU Operations (Processing) ====================

    def stage2_cpu_operations(self, io_batch: IOBatch, devices: List[DeviceUnit]) -> Dict[str, TDO]:
        """
        Run CPU-bound processing on normalized data (per device).

        Parameters
        ----------
        io_batch : IOBatch
            Batch of normalized data from Stage 1.
        devices : List[DeviceUnit]
            Original device units for processor creation.

        Returns
        -------
        Dict[str, TDO]
            Mapping device_id -> data object with processing results.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STAGE 2: CPU Operations for {len(io_batch.device_data)} devices")
        self.logger.info("=" * 60)

        device_results: Dict[str, TDO] = {}

        # Create device lookup for processor factory
        device_lookup = {d.device_id: d for d in devices}

        # Process each device independently. Default is serial; optional per-device concurrency via threads.
        workers = max(1, min(self.config.cpu_max_workers, len(io_batch.device_data)))

        def device_task(device_id: str) -> Tuple[str, TDO]:
            """
            Process a single device's normalized data through its processors.

            Returns
            -------
            Tuple[str, TDO]
                (device_id, resulting data object)
            """
            try:
                datasets = io_batch.device_data[device_id]
                # Create DataObject (domain-specific if provided)
                data_object: TDO = self.data_object_factory(
                    device_id=device_id,
                    pipeline_name=self.config.pipeline_name,
                    metadata={"pipeline_type": self.config.pipeline_name},
                )

                # Add normalized datasets
                data_object.normalized_data.update(datasets)

                # Add metadata from I/O stage
                if device_id in io_batch.device_metadata:
                    data_object.data_sources = io_batch.device_metadata[device_id]

                # Carry forward I/O stage errors/warnings
                if device_id in io_batch.errors:
                    for error in io_batch.errors[device_id]:
                        data_object.add_error(f"I/O Stage: {error}")
                if device_id in io_batch.warnings:
                    for warning in io_batch.warnings[device_id]:
                        data_object.add_warning(f"I/O Stage: {warning}")

                # Freeze normalized data (read-only for processors)
                if self.config.freeze_normalized_data:
                    data_object.freeze_normalized_data()

                # Get processors for this device
                processors = (
                    self.processor_factory(device_lookup.get(device_id, DeviceUnit(device_id=device_id)))  # type: ignore[operator]
                    if self.processor_factory else []
                )

                # Run processors sequentially (per-device)
                for i, processor in enumerate(processors, 1):
                    processor.config.setdefault("stop_on_error", self.config.stop_on_error)
                    try:
                        self.logger.debug(f"  [{i}/{len(processors)}] Running {processor.name} for {device_id}")
                        data_object = processor(data_object)  # type: ignore[assignment]
                    except Exception as e:
                        self.logger.error(f"  Processor {processor.name} failed for {device_id}: {e}")
                        if self.config.stop_on_error:
                            raise
                        self.logger.warning(f"  Continuing after {processor.name} failure...")

                self.logger.info(f"✓ CPU processing complete for {device_id}")
                return device_id, data_object

            except Exception as e:
                self.logger.error(f"Failed CPU processing for device {device_id}: {e}")
                error_object: TDO = self.data_object_factory(  # type: ignore[assignment]
                    device_id=device_id,
                    pipeline_name=f"{self.config.pipeline_name}_error",
                    metadata={},
                )
                error_object.add_error(f"CPU Stage failure: {str(e)}")
                if self.config.stop_on_error:
                    # Propagate to caller to honor stop_on_error semantics
                    raise
                return device_id, error_object

        if workers == 1:
            # Backward-compatible serial path
            for device_id in io_batch.device_data.keys():
                self.logger.info(f"Processing device: {device_id}")
                dev_id, obj = device_task(device_id)
                device_results[dev_id] = obj
        else:
            self.logger.info(f"CPU concurrency enabled: workers={workers}")
            futures = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for device_id in io_batch.device_data.keys():
                    self.logger.info(f"Processing device: {device_id}")
                    futures.append(executor.submit(device_task, device_id))

                for fut in as_completed(futures):
                    try:
                        dev_id, obj = fut.result()
                        device_results[dev_id] = obj
                    except Exception as e:
                        self.logger.error(f"Failed CPU in concurrent worker: {e}")
                        if self.config.stop_on_error:
                            for pending in futures:
                                pending.cancel()
                            raise

        self.logger.info(f"Stage 2 complete: {len(device_results)} devices processed")

        return device_results

    # ==================== Batch Processing ====================

    def process_batch(self, devices: List[DeviceUnit]) -> BatchResults[TDO]:
        """
        Process a batch of devices through both stages.

        Parameters
        ----------
        devices : List[DeviceUnit]
            Devices to process.

        Returns
        -------
        BatchResults[TDO]
            Aggregated device results and timing information for this batch.
        """
        results: BatchResults[TDO] = BatchResults(total_devices=len(devices))

        # Stage 1: I/O Operations (Acquisition + Normalization)
        io_start = time.time()
        io_batch = self.stage1_io_operations(devices)
        io_time = time.time() - io_start
        results.io_timing["total"] = io_time
        results.io_timing["per_device"] = io_time / len(devices) if devices else 0

        # Stage 2: CPU Operations (Processing)
        cpu_start = time.time()
        device_results = self.stage2_cpu_operations(io_batch, devices)
        cpu_time = time.time() - cpu_start
        results.cpu_timing["total"] = cpu_time
        results.cpu_timing["per_device"] = cpu_time / len(device_results) if device_results else 0

        # Aggregate results
        results.device_results = device_results
        results.successful_devices = sum(
            1
            for do in device_results.values()
            if not do.errors or not any("CPU Stage failure" in e for e in do.errors)
        )
        results.failed_devices = results.total_devices - results.successful_devices
        results.end_time = datetime.now(timezone.utc)

        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("Batch Processing Complete")
        self.logger.info(f"  Total devices: {results.total_devices}")
        self.logger.info(f"  Successful: {results.successful_devices}")
        self.logger.info(f"  Failed: {results.failed_devices}")
        self.logger.info(f"  I/O time: {results.io_timing['total']:.2f}s")
        self.logger.info(f"  CPU time: {results.cpu_timing['total']:.2f}s")
        self.logger.info(f"  Total time: {io_time + cpu_time:.2f}s")
        self.logger.info("=" * 60)

        return results

    def run_batch(self, devices: List[DeviceUnit]) -> BatchResults[TDO]:
        """
        Run the pipeline for all devices, processing in configured batch sizes.

        Parameters
        ----------
        devices : List[DeviceUnit]
            All devices to process.

        Returns
        -------
        BatchResults[TDO]
            Aggregated BatchResults over all batches.
        """
        if not devices:
            return BatchResults()

        all_results: BatchResults[TDO] = BatchResults(total_devices=len(devices))

        # Process devices in batches
        batch_size = self.config.batch_size
        for i in range(0, len(devices), batch_size):
            batch = devices[i : i + batch_size]
            self.logger.info(f"\nProcessing batch {i // batch_size + 1}: devices {i + 1}-{i + len(batch)}")

            batch_results = self.process_batch(batch)

            # Aggregate batch results
            all_results.device_results.update(batch_results.device_results)
            all_results.successful_devices += batch_results.successful_devices
            all_results.failed_devices += batch_results.failed_devices

            # Aggregate timing
            all_results.io_timing[f"batch_{i // batch_size + 1}"] = batch_results.io_timing["total"]
            all_results.cpu_timing[f"batch_{i // batch_size + 1}"] = batch_results.cpu_timing["total"]

        # Calculate total timing
        all_results.io_timing["total"] = sum(
            v for k, v in all_results.io_timing.items() if k.startswith("batch_")
        )
        all_results.cpu_timing["total"] = sum(
            v for k, v in all_results.cpu_timing.items() if k.startswith("batch_")
        )
        all_results.end_time = datetime.now(timezone.utc)

        # Cleanup connections (best-effort)
        try:
            PluginManager.cleanup()
        except Exception:
            pass

        return all_results

    # ==================== Optional Export Helpers (Adapter-based) ====================

    def export_summary(self, results: BatchResults[TDO], adapter: Optional[ResultsAdapter[TDO]] = None) -> Dict[str, Any]:
        """
        Export a summary dict using a provided or configured ResultsAdapter.

        Parameters
        ----------
        results : BatchResults[TDO]
            Results to summarize.
        adapter : Optional[ResultsAdapter[TDO]]
            Optional adapter; if not provided, use self.results_adapter.

        Returns
        -------
        Dict[str, Any]
            JSON-serializable summary representation.

        Raises
        ------
        ValueError
            If no adapter is available.
        """
        chosen = adapter or self.results_adapter
        if not chosen:
            raise ValueError("No ResultsAdapter provided to export_summary().")
        return chosen.summarize(results)

    def export_dataframe(
        self, results: BatchResults[TDO], adapter: Optional[ResultsAdapter[TDO]] = None
    ) -> pd.DataFrame:
        """
        Export a flat DataFrame using a provided or configured ResultsAdapter.

        Parameters
        ----------
        results : BatchResults[TDO]
            Results to flatten.
        adapter : Optional[ResultsAdapter[TDO]]
            Optional adapter; if not provided, use self.results_adapter.

        Returns
        -------
        pd.DataFrame
            Tabular representation of results.

        Raises
        ------
        ValueError
            If no adapter is available.
        """
        chosen = adapter or self.results_adapter
        if not chosen:
            raise ValueError("No ResultsAdapter provided to export_dataframe().")
        return chosen.to_dataframe(results)
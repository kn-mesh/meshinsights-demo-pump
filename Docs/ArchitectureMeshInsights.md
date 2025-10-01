# MeshInsights Pipeline Architecture

## Overview

This document describes the **Two-Stage Batch Processing** architecture with **Context (Data Object) + Processor** pattern for building scalable, composable data analytics pipelines in the MeshInsights platform. This architecture enables processing of multiple devices/units efficiently while maintaining per-device isolation and flexibility.

## Core Architecture Components

### Two-Stage Batch Pipeline Architecture

The pipeline is structured in two distinct stages for batch processing:

1. **Stage 1 (I/O Operations)**: Data acquisition + normalization for batches of devices
   - Fetches data from external sources (I/O bound)
   - Normalizes data to standard format (lightweight transformations)
   - Prepares data for CPU-intensive processing
   
2. **Stage 2 (CPU Operations)**: Processing normalized data per device
   - Executes business logic processors (CPU bound)
   - Maintains per-device isolation with individual PipelineDataObjects
   - Produces artifacts and results for each device

This separation enables:
- Efficient resource utilization (I/O vs CPU)
- Clear separation of concerns
- Easy parallelization within each stage
- Batch processing while maintaining device isolation

## Batch Processing Model

### Key Concepts

1. **DeviceUnit**: Represents a single unit of analysis (e.g., IoT device)
   - Contains device ID and device-specific parameters
   - Enables heterogeneous device processing in same batch

2. **Factory Functions**: Create components dynamically per device
   - `data_source_factory`: Creates DataSources for a device
   - `normalizer_factory`: Creates DataNormalizers for a device  
   - `processor_factory`: Creates Processors for a device

3. **Per-Device Isolation**: Each device gets its own PipelineDataObject
   - Independent error handling per device
   - Device-specific artifacts and results
   - Continues processing other devices on failure (configurable)

4. **Batch Results**: Aggregated output from batch processing
   - Individual device results (PipelineDataObjects)
   - Timing metrics for I/O and CPU stages
   - Success/failure statistics

### Processing Flow

```
Batch of Devices [1-25]
    ↓
[Stage 1: I/O Operations]
  - Fetch data for each device
  - Normalize datasets
  - Collect in IOBatch
    ↓
[Stage 2: CPU Operations]  
  - Process each device independently
  - Create PipelineDataObject per device
  - Run processors sequentially
    ↓
[Batch Results]
  - Device results dictionary
  - Timing and statistics
```

## Core vs Solution-Specific

- **Core**: Reusable framework pieces that should rarely change. These live under `src/core/` and define the primitives to build any pipeline.
  - `PipelineDataObject` (state container per device)
  - `Processor` (processing step contract)
  - `DataSource` (data acquisition contract)
  - `DataNormalizer` (normalization contract)
  - `Pipeline` + `PipelineConfig` (two-stage batch orchestration)
  - `DeviceUnit`, `IOBatch`, `BatchResults` (batch processing primitives)
  - Database Plugin System (`DBConnector`, `PluginManager`) with implementations in `src/core/plugins/data_access/`
  - Logging, error handling, execution tracking conventions

- **Solution-Specific**: Domain implementations that compose the core into a working solution. The current HVAC power analysis lives in `src/phd_pipeline/`.
  - Domain data sources (e.g., `PowerDataRetrieval` for ADX telemetry)
  - Domain normalizers (e.g., `PowerDataNormalizer`)
  - Domain processors (cycle filtering, variance classification, baseline thresholds)
  - Domain data object and results adapter (`HVACPipelineDataObject`, `HVACResultsAdapter`)
  - Factory helpers for device-specific pipeline assembly (`pipeline_factory_power.py`)

## Directory Structure

```
src/
├── core/
│   ├── data_object.py                  # PipelineDataObject definition
│   ├── processor.py                    # Base Processor class
│   ├── data_source.py                  # Base DataSource class
│   ├── data_normalizer.py              # Base DataNormalizer class
│   ├── pipeline.py                     # Two-stage batch pipeline orchestrator
│   ├── plugins/
│   │   └── data_access/
│   │       ├── data_access_plugin_core.py  # DBConnector + PluginManager
│   │       └── implementations/
│   │           ├── adx_plugin.py           # Azure Data Explorer connector
│   │           ├── local_mysql_plugin.py   # MySQL connector
│   │           ├── local_file_plugin.py    # Local CSV/Parquet connector
│   │           └── phd_sitesage_api.py     # SiteSage API connector
│   └── utils/
│       └── convert_data_for_ai.py
└── phd_pipeline/
    ├── data_object_phd.py               # HVAC-specific data object + schemas
    ├── data_sources/                    # Domain-specific data sources (PowerDataRetrieval)
    ├── normalizers/                     # Data normalization logic (PowerDataNormalizer)
    ├── processors/                      # Business logic processors
    ├── factories/                       # Pipeline factory functions + helpers
    ├── algorithms/                      # Shared analytics helpers
    └── utils/                           # Constants, lookup tables, helpers
```

## Core Components

### 1. Pipeline (Two-Stage Batch Orchestrator)

The `Pipeline` class orchestrates batch processing through two stages:

**Configuration** (`PipelineConfig`):
- `pipeline_name`: Human-readable name
- `stop_on_error`: Whether to abort on first error
- `log_level`: Logging verbosity
- `batch_size`: Number of devices per batch
- `freeze_normalized_data`: Make data read-only for processors.
- `io_max_workers`: Max concurrent device I/O workers in Stage 1 (default 1 = serial).
- `cpu_max_workers`: Max concurrent device CPU workers in Stage 2 (default 1 = serial).

- **Batch orchestration**: The `Pipeline` now includes explicit batch-level orchestration methods (`process_batch` and `run_batch`) that manage device batching, timing (I/O vs CPU), aggregation of per-device `BatchResults`, and cleanup of external connections. These methods record detailed `io_timing` and `cpu_timing`, maintain per-batch and overall statistics (`successful_devices`/`failed_devices`), and provide hooks for results export via `ResultsAdapter`.

- **Output schema validation**: The pipeline and `PipelineDataObject` support registering and enforcing output schemas for processor artifacts. Processors can register Pandera schemas with `PipelineDataObject.register_artifact_schema(key, schema)` and write DataFrame artifacts with `PipelineDataObject.set_artifact(key, df)`. When an artifact has a registered schema it will be validated on set; validation failures are recorded as warnings (not hard failures) so processors produce consistent, discoverable artifact shapes for downstream adapters and visualizations.

**Key Methods**:
- `stage1_io_operations(devices)`: Fetch and normalize data for device batch
- `stage2_cpu_operations(io_batch, devices)`: Process normalized data per device
- `process_batch(devices)`: Run both stages for a batch
- `run_batch(devices)`: Process all devices in configured batch sizes

#### Producer–Consumer Batch Runner

We extended the pipeline execution model with a **producer–consumer queue** between Stage 1 (I/O-bound fetch) and Stage 2 (CPU-bound normalization + processing).

- **Stage 1 (Producer):** Each batch of devices is fetched and normalized; results are pushed into a bounded queue for consumption.
- **Stage 2 (Consumer):** While I/O for the next batch is still running, the CPU stage consumes the previous batch from the queue and runs processors.
- **Double Buffering:** The default queue size is 2 which enables overlapping fetch and processing (double buffering) so both I/O and CPU stay busy.
- **Configuration:**
  - `PipelineConfig.batch_size` controls devices per batch (for example, `10` in `src/phd_pipeline/factories/pipeline_factory_power.py`).
  - `PipelineConfig.io_max_workers` enables per-device I/O concurrency within Stage 1.
  - `PipelineConfig.cpu_max_workers` enables per-device CPU concurrency within Stage 2.
  - The same queue-based runner can be reused for other pipelines.
- **Entry Point:** Use `pipeline.run_producer_consumer(devices)` instead of `pipeline.run_batch(devices)` to take advantage of the queue-based, overlapped execution model.

This reduces idle time compared to sequential runs and provides smoother throughput when device fetch latency is significant.

**Factory-Based Configuration**:
```python
pipeline = Pipeline(
    data_source_factory=lambda device: {...},  # Create sources per device
    normalizer_factory=lambda device: {...},   # Create normalizers per device
    processor_factory=lambda device: [...],    # Create processors per device
    config=PipelineConfig(
        batch_size=25,
        io_max_workers=8,     # concurrent device I/O (Stage 1)
        cpu_max_workers=4     # concurrent device CPU (Stage 2)
    )
)

# Overlapped execution across batches (I/O and CPU in parallel)
results = pipeline.run_producer_consumer(devices)
```

### 2. DeviceUnit

Represents a single device/unit to be analyzed:
```python
DeviceUnit(
    device_id="DEVICE_001",
    parameters={
        "location_id": "BLDG_001",
        "control_id": "DEVICE_001",
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 1, 31)
    },
    metadata={"pipeline": "phd_hvac_power"}
)
```

### 3. IOBatch

Container for Stage 1 results:
- `device_data`: Dict of device_id → {dataset_name: normalized_dataframe}
- `device_metadata`: Source metadata per device
- `errors`/`warnings`: Per-device error tracking

### 4. BatchResults

Final aggregated results:
- `device_results`: Dict of device_id → PipelineDataObject
- `successful_devices`/`failed_devices`: Statistics
- `io_timing`/`cpu_timing`: Performance metrics


### 5. Database Plugin System [Core]

The plugin system defines a clean boundary between the pipeline and external datastores. It consists of an abstract connector interface and a manager responsible for registration, lifecycle, and pooled connection reuse per configuration.

- **DBConnector (interface)**:
  - Purpose: Define a uniform contract for database connectors that return pandas DataFrames.
  - Public API: `connect() -> (Any, Any)`, `execute_query(query: str) -> pd.DataFrame`, `disconnect() -> None`. Includes context manager support via `__enter__/__exit__` for safe lifecycle management.
  - Responsibility: Encapsulate connection setup/teardown and ensure `execute_query` returns a DataFrame suitable for higher layers.

- **PluginManager (registry + lifecycle + pooling)**:
  - Purpose: Register connector classes and retrieve active instances keyed by `(plugin_name, stable_config_hash)` so different tenants/credentials/configs are isolated while still reusing connections for identical configs.
  - Public API: `register_plugin(name, plugin_cls)`, `get_plugin(name, **kwargs) -> DBConnector`, `list_plugins() -> List[str]`, `cleanup()` to close active connections.
  - Strategy: Centralizes plugin discovery and enables safe connection pooling. `get_plugin()` computes a stable, non-reversible hash of constructor kwargs to scope instances per configuration. This avoids cross-tenant leakage while eliminating redundant connection creation.

#### Connection pooling and per-config isolation (recommended)

- **How it works**:
  - Each connector implementation maintains its underlying client connections in a shared manager (e.g., `ADXConnectionManager`, `MySQLConnectionManager`).
  - The manager pools connections keyed by a stable hash of configuration kwargs (e.g., tenant, cluster, database, host, user).
  - The `DBConnector` instance itself is lightweight; `disconnect()` clears only local references. The pooled client lives in the connection manager and is reused across identical configs.

- **Why this is recommended**:
  - Reduces connection churn and rate limiting against external services.
  - Ensures strict isolation across tenants/configs while allowing efficient reuse within the same config.
  - Keeps higher layers simple: call `plugin.connect()` and `plugin.execute_query()`; conversion to pandas happens inside the connector.

- **Usage pattern**:
  - Provide per-tenant/config kwargs when retrieving a plugin instance; the manager handles pooling and reuse behind the scenes.

```python
# Register once at startup
PluginManager.register_plugin("adx", ADXConnector)
PluginManager.register_plugin("mysql", MySQLConnector)

# Retrieve per-tenant scoped instances (pooled by config hash)
adx = PluginManager.get_plugin(
    "adx",
    tenant_id="<AAD_TENANT>",
    cluster="<https://<cluster>.kusto.windows.net>",
    database="<DB>"
)

mysql = PluginManager.get_plugin(
    "mysql",
    host="db.internal",
    user="svc_reader",
    password="***",
    database="telemetry",
    port=3306
)

# Normal usage in DataSource or batch I/O stage
adx.connect()
df_adx = adx.execute_query("MyKustoTable | take 100")

mysql.connect()
df_mysql = mysql.execute_query("SELECT * FROM some_table LIMIT 100")

# Optional at end of run (best-effort close of pooled connections)
PluginManager.cleanup()
```

- **Example plugin: ADXConnector**
  - File: `src/core/plugins/data_access/implementations/adx_plugin.py`
  - Purpose: Connect to Azure Data Explorer (ADX), execute Kusto queries, convert responses to DataFrames, and manage retries.
  - Notes: Uses a shared `ADXConnectionManager` with a pooled `(config → client)` map; supports per-tenant scoping via kwargs (`tenant_id`, `cluster`, `database`). Handles type conversions (including datetime and timespan) before returning a pandas DataFrame.

See core reference: `src/core/plugins/data_access/data_access_plugin_core.py`.

### 6. Data Source Abstraction [Core]

DataSource wraps the plugin system to provide a clean interface for data acquisition.
Plugins are responsible for returning pandas DataFrames; data sources should not
perform response-to-DataFrame conversion.

**Guidance for implementers:**

- **Purpose**: Encapsulate data acquisition concerns (authentication, query construction, execution) and return raw results as a pandas DataFrame.
- **Responsibilities**:
  - Build a datastore-specific query (`build_query`).
  - Use a registered `DBConnector` via the plugin manager to execute queries (`fetch`).
  - Provide lightweight connectivity checks (`validate_connection`).
  - Expose non-sensitive source metadata (`get_metadata`).
- **Public API**:
  - `build_query() -> str`: Construct the query string used by the plugin.
  - `fetch() -> pandas.DataFrame`: Execute the query through the plugin and return a DataFrame. Implementations should call `self.plugin.connect()` before executing the query, then `self.plugin.execute_query(query)`. Plugins handle conversion to DataFrame.
  - `validate_connection() -> bool`: Probe basic reachability using the plugin.
  - `get_metadata() -> Dict[str, Any]`: Return name, plugin, and a snapshot of configuration.
  - `plugin` property: Lazily retrieves the configured `DBConnector` from the plugin manager.
- **Strategy**: Solution-specific sources subclass `DataSource` to implement query construction while remaining decoupled from networking/driver details. This keeps I/O in Phase 1 and ensures normalized processing in later phases. See `src/core/data_source.py`.

### 7. Data Normalizer Abstraction [Core]

The `DataNormalizer` transforms raw source data into the standardized schema required by processors. Normalizers are pure and deterministic with simple Pandera-based validation.

- **Public API:**
  - `normalize(data: DataFrame, metadata: Optional[Dict]) -> DataFrame`: Template method that runs basic input validation, calls `transform`, then validates output with optional Pandera schema.
  - `transform(data: DataFrame, metadata: Optional[Dict]) -> DataFrame`: **Pure** transformation logic only (no I/O, no validation).
  - `get_schema() -> Optional[pa.DataFrameSchema]`: Returns optional Pandera schema for the normalized output.

```python
# Example normalizer with validation
class HVACPowerNormalizer(DataNormalizer):
    def get_schema(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema({
            "timestamp": pa.Column(pa.DateTime, nullable=False),
            "power_kw": pa.Column(pa.Float, ge=0, nullable=False),
            "stage": pa.Column(pa.String, nullable=False),
            "cycle_id": pa.Column(pa.String, nullable=True)
        })
    
    def transform(self, data, metadata=None):
        # pure, deterministic transforms...
        normalized = data.rename(columns={"PowerKW": "power_kw", ...})
        return normalized
```

- **Responsibilities:**
  - Define deterministic transformations in `transform`
  - Optionally declare output schema via `get_schema`
  - Remain pure with no side effects

- **Strategy:** Phase 2 consumes raw DataFrames from Phase 1 and applies normalizers per source, optionally validating against schemas and emitting warnings if validation fails. Normalizers remain side-effect free and contain no I/O. See `src/core/data_normalizer.py`.

### 8. Pipeline Data Object [Core]

The `PipelineDataObject` is the central state container that flows through all phases. It accumulates datasets, metadata, results, logs, warnings, and errors.

- **Purpose**: Provide a strongly-typed, traceable model of pipeline state for each run.
- **Core data**:
  - `normalized_data`: Dict[str, pandas.DataFrame] of named, distinct datasets (e.g., `"hvac_power"`, `"connectivity"`). **Read-only after Phase 2** - processors cannot modify these datasets and must write outputs to `artifacts` instead.
  - `data_sources`, `normalization_metadata`: Metadata captured during Phases 1–2.
  - `artifacts`: Combined storage for processor outputs and stage summaries (DataFrames, dicts, text) stored by namespaced keys for discoverability. **Primary destination for all processor outputs**.
  - `errors`, `warnings`: Execution telemetry with timestamps.
  - `artifact_schemas`: Optional Pandera schemas for validating DataFrame artifacts.

- **Public methods**:
  - `get_dataset(key: str) -> DataFrame`: Retrieve normalized dataset by key.
  - `set_artifact(key, value) -> PipelineDataObject`: Store outputs with optional DataFrame validation if schema is registered for the key.
  - `get_artifact(key, default=None) -> Any`: Retrieve stored artifacts by key.
  - `register_artifact_schema(key, schema) -> PipelineDataObject`: Register a Pandera schema for validating DataFrame artifacts with a specific key.
  - `list_artifacts(prefix=None) -> List[str]`: List artifact keys, optionally filtered by prefix.
  - `add_error(message) -> PipelineDataObject`: Record timestamped errors.
  - `add_warning(message) -> PipelineDataObject`: Record timestamped warnings.

- **DataFrame Validation**: When `set_artifact()` is called with a DataFrame and a schema is registered for that key, the DataFrame is validated. Validation failures result in warnings but don't stop execution.

- **Strategy**: Processors mutate and return the same data object instance. The data object enables auditing and downstream visualization (e.g., in Streamlit). See `src/core/data_object.py`.


### 9. Processor Base Class [Core]

The `Processor` is the base for all Phase 3 steps that transform the data object's data and write results back into the data object.

- **Purpose**: Provide a consistent, error-handled execution wrapper around domain-specific `process` implementation.
- **Responsibilities**:
  - Register output schemas for DataFrame validation (`register_output_schemas`).
  - Validate prerequisites before running (`validate_prerequisites`).
  - Execute domain logic (`process`) and optionally validate outcomes (`validate_output`).
  - Log timing and capture errors with a configurable stop/continue policy.

- **Public API**:
  - `process(data_object) -> PipelineDataObject` (to implement in subclasses).
  - `register_output_schemas(data_object)` (hook to register schemas for DataFrame outputs).
  - `__call__(data_object) -> PipelineDataObject` (adds timing, logging, and error policy).
  - Hooks: `validate_prerequisites(data_object)`, `validate_output(data_object)`.

**Example processor with DataFrame validation:**

```python
class FilterValidCyclesProcessor(Processor):
    def register_output_schemas(self, data_object):
        schema = pa.DataFrameSchema({
            "timestamp": pa.Column(pa.DateTime, nullable=False),
            "power_kw": pa.Column(pa.Float, ge=0, nullable=False),
            "cycle_id": pa.Column(pa.String, nullable=False),
        })
        data_object.register_artifact_schema("cycles/filtered_data", schema)

    def process(self, data_object):
        hvac_data = data_object.get_dataset("hvac_power")
        filtered_data = self._filter_cycles(hvac_data)
        # This will be validated against the registered schema
        data_object.set_artifact("cycles/filtered_data", filtered_data)
        return data_object
```

- **Strategy**: Compose multiple processors to build complex analyses. Each processor is stateless and focuses on one concern, relying on the data object for inputs and outputs. See `src/core/processor.py`.


## Solution Implementation Guide

### 1. Define Factory Functions

Create functions that generate components for each device. The HVAC pipeline factories in `src/phd_pipeline/factories/pipeline_factory_power.py` compose the real implementations shown below:

```python
from typing import Any, Dict, List

# Imports for DeviceUnit/DataSource/DataNormalizer/Processor/HVACPipelineDataObject
# come from src.core.pipeline and src.phd_pipeline.* modules in the repository.

def create_hvac_data_sources(device: DeviceUnit) -> Dict[str, DataSource]:
    """Build HVAC data sources using device parameters."""
    params = device.parameters or {}
    return {
        "hvac_power": PowerDataRetrieval(
            location_id=str(params["location_id"]),
            control_id=str(params["control_id"]),
            start_date=params["start_date"],
            end_date=params["end_date"],
            adx_cluster=params.get("adx_cluster"),
            adx_database=params.get("adx_database"),
            adx_tenant_id=params.get("adx_tenant_id"),
        )
    }

def create_hvac_normalizers(device: DeviceUnit) -> Dict[str, DataNormalizer]:
    """Return the PowerDataNormalizer for the HVAC dataset."""
    return {"hvac_power": PowerDataNormalizer()}

def create_hvac_processors(device: DeviceUnit) -> List[Processor]:
    """Ordered HVAC processor chain mirroring the production pipeline."""
    return [
        FilterValidCycles(),
        CheckSufficientCycles(),
        ClassifyPowerVarianceFilteredData(),
        CurateCycleMediansPerStage(),
        ClassifyPowerVarianceCycleMedian(),
        CalculateBaselinePowerThresholds(),
    ]

def create_hvac_data_object(device_id: str, pipeline_name: str, metadata: Dict[str, Any]) -> HVACPipelineDataObject:
    """Create the HVAC-specific PipelineDataObject with canonical artifact schemas."""
    return HVACPipelineDataObject(
        pipeline_name=f"{pipeline_name}_{device_id}",
        config={"device_id": device_id, "pipeline_type": "hvac_power", **(metadata or {})},
    )
```

### 2. Configure and Run Pipeline

```python
from datetime import date

# Define devices to process (location + control pair)
devices = [
    DeviceUnit(
        device_id=control_id,
        parameters={
            "location_id": "11688",
            "control_id": control_id,
            "start_date": date(2025, 2, 5),
            "end_date": date(2025, 5, 5),
        },
    )
    for control_id in ["2916276", "2916277"]
]

# Create pipeline with HVAC factories and run with producer-consumer overlap
pipeline = Pipeline(
    data_source_factory=create_hvac_data_sources,
    normalizer_factory=create_hvac_normalizers,
    processor_factory=create_hvac_processors,
    data_object_factory=create_hvac_data_object,
    results_adapter=HVACResultsAdapter(),
    config=PipelineConfig(
        pipeline_name="phd_hvac_power",
        batch_size=10,
        stop_on_error=False,
        io_max_workers=5,
        cpu_max_workers=5,
    ),
)

results = pipeline.run_producer_consumer(devices)

# Access per-device artifacts
for device_id, dobj in results.device_results.items():
    issues = dobj.get_artifact(HVACPipelineDataObject.ARTIFACT_ISSUES_PER_STAGE, {})
    thresholds = dobj.get_artifact(HVACPipelineDataObject.ARTIFACT_BASELINE_THRESHOLDS, {})
    print(device_id, issues, thresholds)
```

### 3. Aggregate Results

```python
import pandas as pd

# Use the HVACResultsAdapter for a tabular summary
adapter = HVACResultsAdapter()
summary_df = adapter.to_dataframe(results)
summary_df.to_csv("batch_results.csv", index=False)

# Or build a custom view
metrics = []
for device_id, dobj in results.device_results.items():
    metrics.append({
        "device_id": device_id,
        "issues_per_stage": dobj.get_artifact(HVACPipelineDataObject.ARTIFACT_ISSUES_PER_STAGE, {}),
        "baseline_thresholds": dobj.get_artifact(HVACPipelineDataObject.ARTIFACT_BASELINE_THRESHOLDS, {}),
        "errors": len(dobj.errors),
        "warnings": len(dobj.warnings),
    })
custom_df = pd.DataFrame(metrics)
```

## Performance Considerations

- **Batch Size**: Balance between memory usage and efficiency
  - Larger batches: Better I/O efficiency, higher memory usage
  - Smaller batches: Lower memory, more I/O overhead
  
- **Stage Timing**: Monitor io_timing vs cpu_timing
  - If I/O dominant: Consider batch queries or caching
  - If CPU dominant: Focus on processor optimization

- **Error Handling**: Use stop_on_error=False for resilience
  - Process all devices even with failures
  - Review failed devices separately

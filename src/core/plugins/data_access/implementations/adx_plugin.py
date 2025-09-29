# uv run python -m src.plugins.implementations.adx_plugin

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import time
from typing import Any, Tuple, Dict, Type
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.response import KustoResultTable

from src.core.plugins.data_access.data_access_plugin_core import DBConnector, PluginManager

class ADXConnectionManager:
    """
    A singleton class for managing Azure Data Explorer (ADX) connections using Service Principal authentication.
    
    This manager handles connection setup and maintenance using Azure Service Principal credentials.
    All credentials must be configured via environment variables.
    
    Required Environment Variables:
        AZURE_TENANT_ID: Azure AD tenant ID
        AZURE_CLIENT_ID: Service Principal client ID
        AZURE_CLIENT_SECRET: Service Principal secret
        KUSTO_CLUSTER: ADX cluster URL
        KUSTO_DATABASE: Target database name    
    """
    load_dotenv()
    _instance = None
    _pool = {}  # key -> (database, client)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ADXConnectionManager, cls).__new__(cls)
        return cls._instance

    def get_connection(self, cfg: Dict[str, str] | None = None):
        """
        Get the ADX connection, establishing it if it doesn't exist.
        Returns:
            tuple: (KUSTO_DATABASE, KUSTO_CLIENT)
        """
        key = self._make_key(cfg)
        if key not in self._pool:
            self._pool[key] = self._establish_adx_connection(cfg)
        return self._pool[key]

    def _make_key(self, cfg):
        import hashlib, json
        payload = json.dumps(cfg or {}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _establish_adx_connection(self, cfg: Dict[str, str] | None = None):
        """
        Establish connection to Azure Data Explorer (ADX) using environment variables.
        Returns:
            tuple: (KUSTO_DATABASE, KUSTO_CLIENT)
        Raises:
            ValueError: If required environment variables are missing.
            ConnectionError: If connection fails.
        """
        # Retrieve required environment variables
        AAD_TENANT_ID = (cfg or {}).get('tenant_id') or os.environ.get('AZURE_TENANT_ID')
        CLIENT_ID = os.environ.get('AZURE_CLIENT_ID')
        CLIENT_SECRET = os.environ.get('AZURE_CLIENT_SECRET')
        # Prefer explicit cfg values; then standard env
        KUSTO_CLUSTER = (cfg or {}).get('cluster') or os.environ.get('KUSTO_CLUSTER')
        KUSTO_DATABASE = (cfg or {}).get('database') or os.environ.get('KUSTO_DATABASE')

        # Check for missing variables
        missing_vars = []
        for var_name, var_value in [
            ('AZURE_TENANT_ID', AAD_TENANT_ID),
            ('AZURE_CLIENT_ID', CLIENT_ID),
            ('AZURE_CLIENT_SECRET', CLIENT_SECRET),
            ('KUSTO_CLUSTER', KUSTO_CLUSTER),
            ('KUSTO_DATABASE', KUSTO_DATABASE)
        ]:
            if not var_value:
                missing_vars.append(var_name)
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Ensure KUSTO_CLUSTER has a proper URL scheme
        if not KUSTO_CLUSTER.startswith(('https://', 'http://')):
            KUSTO_CLUSTER = f'https://{KUSTO_CLUSTER}'

        try:
            kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
                KUSTO_CLUSTER,
                CLIENT_ID,
                CLIENT_SECRET,
                AAD_TENANT_ID
            )
            KUSTO_CLIENT = KustoClient(kcsb)
            return KUSTO_DATABASE, KUSTO_CLIENT
        except Exception as e:
            raise ConnectionError(f"Failed to establish ADX connection: {str(e)}") from e

class ADXConnector(DBConnector):
    """
    ADXConnector is a plugin that interacts with Azure Data Explorer (ADX).
    It implements the common DBConnector interface with connection state management.
    """
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, **kwargs):
        """
        Initialize the ADXConnector.

        Args:
            max_retries (int): Maximum number of retry attempts for queries.
            retry_delay (float): Delay in seconds between retries.
            **kwargs: Additional keyword arguments for extensibility.
        """
        self.database = None
        self.client = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connected = False
        import threading
        self._lock = threading.RLock()
        # Persist constructor configuration so connect() can forward to the manager
        # Expected keys include: 'cluster', 'database', 'tenant_id'
        self._cfg: Dict[str, str] = {}
        for key in ("cluster", "database", "tenant_id"):
            val = kwargs.get(key)
            if val:
                self._cfg[key] = val

    @property
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self._connected and self.client is not None

    def connect(self) -> Tuple[str, KustoClient]:
        """
        Establish the connection using the ADXConnectionManager singleton.

        Returns:
            tuple: (database, client) The database name and KustoClient instance

        Raises:
            ConnectionError: If connection cannot be established
        """
        try:
            with self._lock:
                if not self.is_connected:
                    cfg = getattr(self, "_cfg", None)
                    self.database, self.client = ADXConnectionManager().get_connection(cfg)
                    self._connected = True
                return self.database, self.client
        except Exception as e:
            self._connected = False
            print(f"Connection failed: {type(e).__name__}: {str(e)}")
            raise ConnectionError(f"Failed to connect to ADX: {str(e)}") from e

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a query against ADX with retry logic and return a DataFrame.

        Args:
            query (str): A Kusto query string

        Returns:
            pd.DataFrame: Query results as a pandas DataFrame

        Raises:
            ConnectionError: If connection fails after retries
            ValueError: If query is invalid
        """
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            try:
                self.connect()  # idempotent + locked
                response = self.client.execute(self.database, query)
                # Convert to DataFrame inside the plugin to keep callers simple
                df = dataframe_from_response(response)
                return df
            except Exception as e:
                last_error = e
                attempts += 1
                self._connected = False
                print(f"Attempt {attempts} failed: {type(e).__name__}: {str(e)}")
                if attempts < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)

        raise ConnectionError(f"Query failed after {self.max_retries} attempts. Last error: {str(last_error)}")

    def disconnect(self) -> None:
        """
        Close the connection and clean up resources.
        """
        with self._lock:
            # Only clear local refs; underlying ADX client lives in the manager
            self.client = None
            self.database = None
            self._connected = False

# =============================================================================
# 3. ADX Helper Functions
# =============================================================================

def to_pandas_timedelta(raw_value: "int | float | str") -> pd.Timedelta:
    """
    Convert a raw value to a pandas Timedelta.
    """
    if isinstance(raw_value, (int, float)):
        # 1 tick == 100 nanoseconds
        return pd.to_timedelta(raw_value * 100, unit="ns")
    if isinstance(raw_value, str):
        parts = raw_value.split(":")
        if "." not in parts[0]:
            return pd.to_timedelta(raw_value)
        else:
            formatted_value = raw_value.replace(".", " days ", 1)
            return pd.to_timedelta(formatted_value)


def dataframe_from_response(response: Any, nullable_bools: bool = False) -> pd.DataFrame:
    """
    Convert a Kusto response directly into a pandas DataFrame with optimized type conversion.
    
    Args:
        response: The response from ADX query execution
        nullable_bools (bool): When True, converts boolean columns to pandas BooleanDtype
        
    Returns:
        pd.DataFrame: Converted DataFrame with appropriate dtypes
        
    Raises:
        ValueError: If response is invalid, empty, or malformed
        TypeError: If input table is not a valid Kusto result table
    """
    if not response or not hasattr(response, 'primary_results'):
        raise ValueError("Invalid response or no results found")
    
    if not response.primary_results:
        return pd.DataFrame()  # Return empty DataFrame if no results
    
    table = response.primary_results[0]
    
    if not table or not hasattr(table, 'columns'):
        raise ValueError("Invalid or empty table provided")

    # Define type conversion mapping
    type_converters: Dict[str, Any] = {
        "bool": lambda x: pd.BooleanDtype() if nullable_bools else bool,
        "int": lambda x: pd.Int32Dtype(),
        "long": lambda x: pd.Int64Dtype(),
        "real": lambda x: pd.Float64Dtype(),
        "decimal": lambda x: pd.Float64Dtype(),
        "datetime": lambda x: "datetime64[ns]",
        "timespan": lambda x: "timedelta64[ns]"
    }

    # Extract column names and types
    columns = [col.column_name for col in table.columns]
    if not columns:
        raise ValueError("No columns found in table")

    # Create DataFrame
    frame = pd.DataFrame(table.raw_rows, columns=columns)
    if frame.empty:
        return frame

    # Process columns
    for col in table.columns:
        try:
            col_name = col.column_name
            col_type = col.column_type

            if col_type in type_converters:
                if col_type in ("real", "decimal"):
                    # Handle special numeric cases
                    frame[col_name] = (frame[col_name]
                        .replace({"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf})
                        .pipe(lambda x: pd.to_numeric(x, errors="coerce"))
                        .astype(type_converters[col_type](None)))
                
                elif col_type == "datetime":
                    # Convert to datetime and truncate to seconds
                    frame[col_name] = (pd.to_datetime(frame[col_name], errors="coerce")
                                     .dt.floor('s'))  # Truncate to seconds
                
                elif col_type == "timespan":
                    frame[col_name] = frame[col_name].apply(to_pandas_timedelta)
                
                else:
                    # Standard type conversion
                    frame[col_name] = frame[col_name].astype(type_converters[col_type](None))

        except Exception as e:
            # Log error but continue processing other columns
            print(f"Warning: Failed to convert column {col_name}: {str(e)}")
            continue

    return frame

def get_string_tail_lower_case(val: str, length: int) -> str:
    """
    Return the lower-case tail of a string with the specified length.
    """
    if length <= 0:
        return ""
    if length >= len(val):
        return val.lower()
    return val[len(val) - length:].lower()



# LOCAL TESTING
# uv run python -m src.plugins_data_access.implementations.adx_plugin
if __name__ == "__main__":
    # Register the ADXConnector plugin under the name "adx"
    PluginManager.register_plugin("adx", ADXConnector)

    # Retrieve the ADX plugin from the manager
    adx_plugin = PluginManager.get_plugin("adx")
    adx_plugin.connect()

    # Define a sample Kusto query
    query = """
.show tables
| project TableName
"""
    df = adx_plugin.execute_query(query)
    print(df)
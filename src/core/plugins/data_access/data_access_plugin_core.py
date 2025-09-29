# data_access_plugin_core.py

from abc import ABC, abstractmethod
from typing import Any, Tuple, Type, Dict, List, TypedDict, Literal, Union
import threading
import hashlib
import json
import pandas as pd


class RequestSpec(TypedDict, total=False):
    """
    Generic request specification for REST-like connectors.

    Fields:
        method: HTTP method, e.g., "GET", "POST"
        path: Endpoint path relative to a base URL
        params: Query string parameters
        headers: HTTP headers to add/override
        json: JSON body for POST/PUT/PATCH
        data: Form-encoded body
        paginate: Hints for pagination handling (connector-specific)
        response_path: List-like path (e.g., ["data", "items"]) to the array
                       of rows in the JSON response to convert into a DataFrame
    """
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    path: str
    params: Dict[str, Any]
    headers: Dict[str, str]
    json: Dict[str, Any]
    data: Dict[str, Any]
    paginate: Dict[str, Any]
    response_path: List[str]


class DBConnector(ABC):
    """
    Abstract base class defining the interface for database connectors.

    Public API:
    - connect() -> Tuple[Any, Any]
    - execute_query(query: str) -> pd.DataFrame
    - disconnect() -> None

    Implementations should encapsulate connection lifecycle and ensure
    execute_query returns a result type suitable for the calling layer.
    """

    @abstractmethod
    def connect(self) -> Tuple[Any, Any]:
        """
        Establish a connection to the database.

        Returns:
            tuple: Connection details specific to the database implementation

        Raises:
            ConnectionError: If connection cannot be established
            ValueError: If configuration is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a query against the database and return the results as a DataFrame.

        Args:
            query (str): The query string to execute

        Returns:
            pd.DataFrame: Query results as a pandas DataFrame. Implementations
            should perform any necessary conversion from native response types.

        Raises:
            ConnectionError: If connection is lost during query execution
            ValueError: If query is invalid
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the database connection and clean up resources.

        Raises:
            ConnectionError: If disconnection fails
        """
        raise NotImplementedError

    def execute_request(self, request: Union["RequestSpec", Dict[str, Any], str]) -> pd.DataFrame:
        """
        Execute a request against the datastore and return the results as a DataFrame.

        Default behavior:
            - If `request` is a str, delegate to `execute_query(request)`.
            - Otherwise, raise NotImplementedError and let REST-aware connectors
              override this method.

        Args:
            request: Either a query string or a structured RequestSpec for REST APIs.

        Returns:
            pd.DataFrame: Request results as a pandas DataFrame.
        """
        if isinstance(request, str):
            return self.execute_query(request)
        raise NotImplementedError("This connector does not support RequestSpec-based execution")

    @classmethod
    def pool_key_fields(cls) -> List[str]:
        """
        Optional override for plugins to declare which constructor kwargs define
        the identity of a pooled connection (e.g., base_url, tenant, database).
        Secrets like client_secret/api_key must NOT be included here.

        Returns:
            List[str]: Names of kwargs to use for the pool key. If empty, the
            manager will derive a key from all non-secret-like kwargs.
        """
        return []

    def __enter__(self):
        """Enable context manager support."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup of resources."""
        self.disconnect()


class PluginManager:
    """
    Manager for registering and retrieving database connector plugins.

    Implements a simple singleton to maintain a shared registry and to keep
    one active instance per (plugin name, stable config hash) to avoid cross-
    tenant/credential collisions when the same plugin is used with different
    configurations.
    """

    _instance = None
    _plugins: Dict[str, Type[DBConnector]] = {}
    _active_connections: Dict[Tuple[str, str], DBConnector] = {}
    _lock: threading.RLock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register_plugin(cls, name: str, plugin_cls: Type[DBConnector]) -> None:
        """
        Register a new database connector plugin.

        Args:
            name (str): The name under which to register the plugin.
            plugin_cls (Type[DBConnector]): The plugin class to register.

        Raises:
            ValueError: If plugin name is already registered or class is invalid.
        """
        with cls._lock:
            if name in cls._plugins:
                raise ValueError(f"Plugin already registered under name: {name}")
            if not issubclass(plugin_cls, DBConnector):
                raise ValueError("Plugin class must inherit from DBConnector")
            cls._plugins[name] = plugin_cls

    @classmethod
    def get_plugin(cls, name: str, **kwargs: Any) -> DBConnector:
        """
        Retrieve and instantiate a registered plugin.

        Args:
            name (str): The name of the plugin to retrieve.
            **kwargs: Additional arguments to pass to the plugin constructor.

        Returns:
            DBConnector: An instance of the requested plugin.

        Raises:
            ValueError: If no plugin is registered under the given name.
        """
        plugin_cls = cls._plugins.get(name)
        if plugin_cls is None:
            raise ValueError(f"No plugin registered under the name: {name}")

        # Compute a stable, non-reversible hash for pooling that excludes secrets.
        # Plugins can opt-in to a specific key via `pool_key_fields()`.
        pool_view = cls._pool_view(plugin_cls, kwargs)
        config_hash = cls._stable_config_hash(pool_view)
        key = (name, config_hash)

        with cls._lock:
            if key not in cls._active_connections:
                cls._active_connections[key] = plugin_cls(**kwargs)
            return cls._active_connections[key]

    @classmethod
    def list_plugins(cls) -> List[str]:
        """List all registered plugin names."""
        with cls._lock:
            return list(cls._plugins.keys())

    @classmethod
    def cleanup(cls) -> None:
        """
        Clean up all active connections by calling `disconnect` and clearing
        the active connection cache.
        """
        with cls._lock:
            for connection in cls._active_connections.values():
                try:
                    connection.disconnect()
                except Exception:
                    # Best-effort cleanup
                    pass
            cls._active_connections.clear()

    def __del__(self):
        """Ensure cleanup on deletion of the manager instance."""
        self.cleanup()

    # ===== Helpers =====
    @staticmethod
    def _is_secret_key(key: str) -> bool:
        """
        Heuristic to detect secret-like keys that should not define pooling identity.
        """
        key_l = (key or "").lower()
        secret_markers = [
            "secret", "password", "passwd", "pwd",
            "token", "apikey", "api_key", "key", "access_key", "private_key",
            "client_secret"
        ]
        return any(marker in key_l for marker in secret_markers)

    @classmethod
    def _pool_view(cls, plugin_cls: Type[DBConnector], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a filtered view of kwargs to use for the pool key.

        Preference order:
            1. plugin_cls.pool_key_fields() if provided and non-empty.
            2. Otherwise, all kwargs EXCEPT secret-like keys.
        """
        fields: List[str] = []
        if hasattr(plugin_cls, "pool_key_fields"):
            try:
                fields = plugin_cls.pool_key_fields()  # type: ignore[attr-defined]
            except Exception:
                fields = []
        if fields:
            return {k: kwargs.get(k) for k in fields if k in kwargs}
        # fallback: exclude secret-like keys
        return {k: v for k, v in (kwargs or {}).items() if not cls._is_secret_key(k)}

    @staticmethod
    def _stable_config_hash(config_kwargs: Dict[str, Any]) -> str:
        """
        Create a stable SHA-256 hash from constructor kwargs.

        Args:
            config_kwargs (Dict[str, Any]): Keyword arguments used to construct
                the plugin instance. Can contain nested structures.

        Returns:
            str: Hex digest representing a stable hash of the kwargs.
        """

        def make_json_serializable(obj: Any) -> Any:
            # Convert objects to JSON-serializable, deterministically ordered structures
            if isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in sorted(obj.items(), key=lambda i: str(i[0]))}
            if isinstance(obj, (list, tuple, set)):
                return [make_json_serializable(v) for v in (sorted(obj) if isinstance(obj, set) else obj)]
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            # Fallback to string representation for unsupported types
            return str(obj)

        normalized = make_json_serializable(config_kwargs or {})
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

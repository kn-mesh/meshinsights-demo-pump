"""
Core data acquisition abstraction for the MeshInsights pipeline.

This module defines the `DataSource` abstract base class which encapsulates
the responsibilities of authenticating to external systems, preparing queries,
and retrieving raw data. Concrete implementations (solution-specific) should
inherit from this class and use the plugin system to communicate with actual
datastores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging
import pandas as pd

from src.core.plugins.data_access.data_access_plugin_core import PluginManager, DBConnector


class DataSource(ABC):
    """
    Abstract base class for acquiring raw data from external systems.

    Responsibilities:
    - Build a query (or equivalent) for the underlying datasource.
    - Use a registered plugin (`DBConnector`) to execute the retrieval.
    - Return raw results as a pandas DataFrame. Conversion to DataFrame is
      handled by the plugin, not the data source.

    Public API:
    - build_query() -> str
    - fetch() -> pandas.DataFrame
    - validate_connection() -> bool
    - get_metadata() -> Dict[str, Any]

    Notes:
    - Use `PluginManager.get_plugin(plugin_name, **kwargs)` to obtain a
      connector. Plugins in this repository return `pandas.DataFrame` from
      `execute_query(...)`. Keep network/database specifics in the plugin and
      keep this class as orchestration-only.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        plugin_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        plugin_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a data source.

        Args:
            name (Optional[str]): Human-friendly identifier for this data source.
                Defaults to the class name.
            plugin_name (Optional[str]): Name of the registered database
                connector to use via `PluginManager`.
            config (Optional[Dict[str, Any]]): Non-sensitive configuration for
                query construction (e.g., identifiers, date ranges). Stored for
                metadata.
            plugin_kwargs (Optional[Dict[str, Any]]): Keyword arguments
                forwarded to plugin construction when first accessed.
        """
        self.name: str = name or self.__class__.__name__
        self.plugin_name: Optional[str] = plugin_name
        self.config: Dict[str, Any] = config or {}
        self._plugin_kwargs: Dict[str, Any] = plugin_kwargs or {}
        self._plugin: Optional[DBConnector] = None

        self.logger = logging.getLogger(f"datasource.{self.name}")

    # ===== Orchestration: Public API =====
    @property
    def plugin(self) -> DBConnector:
        """
        Lazily retrieve or create the configured database connector plugin.

        Returns:
            DBConnector: The active connector instance for this data source.

        Raises:
            ValueError: When `plugin_name` is not provided or not registered.
        """
        if self._plugin is None:
            if not self.plugin_name:
                raise ValueError(
                    f"No plugin_name configured for data source '{self.name}'"
                )
            self._plugin = PluginManager.get_plugin(
                self.plugin_name, **self._plugin_kwargs
            )
        return self._plugin

    @abstractmethod
    def build_query(self) -> str:
        """
        Build the query string for data retrieval.

        Returns:
            str: A datastore-specific query string appropriate for the plugin.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """
        Fetch raw data from the external source.

        Expectations:
        - Implementations should call `self.plugin.connect()` prior to executing
          the query, then call `self.plugin.execute_query(query)`.
        - Plugins are responsible for returning a `pd.DataFrame`; data sources
          should not perform response conversion.
        - Implementations must not perform normalization. They should return raw
          datasets to be handled by `DataNormalizer` instances.

        Returns:
            pandas.DataFrame: Raw, unprocessed data.
        """
        raise NotImplementedError

    def validate_connection(self) -> bool:
        """Lightweight, non-destructive probe. Does NOT disconnect."""
        try:
            self.plugin.connect()  # idempotent; connector can no-op if already connected
            return True
        except Exception as exc:
            self.logger.error(f"Connection validation failed: {exc}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return non-sensitive metadata describing this data source.

        Returns:
            Dict[str, Any]: Metadata including name, plugin, and a snapshot of
            the configuration used to construct queries. Sensitive credentials
            must not be included.
        """
        return {
            "name": self.name,
            "plugin": self.plugin_name,
            "config": self.config,
        }

    # ===== Helpers: Private granularity =====
    def _execute(self, query: str) -> Any:
        """
        Execute a query via the configured plugin and return the raw response.

        Args:
            query (str): The query string to execute.

        Returns:
            Any: Plugin-specific raw response suitable for conversion by the
            concrete `fetch` implementation.
        """
        connector = self.plugin
        return connector.execute_query(query)
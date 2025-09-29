"""
Local file plugin implementing the `DBConnector` interface for reading data from
CSV, JSON, and TXT files on the local filesystem.

Design goals:
- Simple swap with cloud database plugins via `DataSource` using the same public API
  (`connect`, `execute_query`, `disconnect`).
- Out-of-the-box support for `.csv`, JSON arrays, JSON Lines (jsonl), and `.txt`.
- Extensible via `pandas_read_kwargs` passed at construction time.

Usage example:
    from src.plugins_data_access.data_access_plugin_core import PluginManager

    connector = PluginManager.get_plugin(
        "local_file",
        file_path="/path/to/data.csv",
        file_format="csv",  # optional; inferred from extension if omitted
        pandas_read_kwargs={"parse_dates": ["timestamp"]},
    )

    connector.connect()
    df = connector.execute_query("value > 0 and status == 'OK'")  # optional filter
    connector.disconnect()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import os
import pandas as pd

from src.core.plugins.data_access.data_access_plugin_core import DBConnector, PluginManager


class LocalFileConnector(DBConnector):
    """
    DBConnector implementation for local files.

    Public API:
    - connect() -> Tuple[Any, Any]
    - execute_query(query: str) -> pd.DataFrame
    - disconnect() -> None

    Constructor parameters:
    - file_path (str): Absolute or relative path to the file on disk.
    - file_format (Optional[str]): "csv", "json", or "txt". If omitted, inferred
      from the file extension.
    - pandas_read_kwargs (Optional[Dict[str, Any]]): Extra keyword arguments
      forwarded to `pandas.read_csv` or `pandas.read_json`. For TXT files,
      the following custom options are recognized and removed from this dict:
        - as_blob (bool, default False): If True, return a single-row DataFrame
          with the entire file content under column "content".
        - column_name (str, default "text"): Column name when splitting into
          one row per line.
        - encoding (str, optional): File encoding for reading text.

    Notes on execute_query(query):
    - `query` is optional. If provided, it's applied as a `pandas.DataFrame.query`
      filter to the loaded data.
    - For JSON, supports arrays of objects and JSON Lines (jsonl) when
      `lines=True` is set or the extension is `.jsonl`.
    - For TXT, default behavior is one row per line in a column named `text`.
      Set `as_blob=True` in `pandas_read_kwargs` to return a single-row DataFrame
      with the entire file content.
    """

    def __init__(
        self,
        *,
        file_path: str,
        file_format: Optional[str] = None,
        pandas_read_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.file_path: str = file_path
        self.file_format: Optional[str] = (file_format or "").strip().lower() or None
        self.pandas_read_kwargs: Dict[str, Any] = dict(pandas_read_kwargs or {})
        self._connected: bool = False

    # ===== Public API =====
    def connect(self) -> Tuple[Any, Any]:
        """
        Validate file existence and determine effective file format.

        Returns:
            tuple: (resolved_path, effective_format)

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported or cannot be inferred.
        """
        resolved_path = os.path.abspath(self.file_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Local file not found: {resolved_path}")

        effective_format = self._determine_format(resolved_path)
        self._connected = True
        return resolved_path, effective_format

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Load the local file into a DataFrame and optionally filter using
        `DataFrame.query` when a non-empty query string is provided.

        Args:
            query (str): Optional pandas query expression to filter rows.

        Returns:
            pd.DataFrame: The loaded (and possibly filtered) data.

        Raises:
            ConnectionError: If called before `connect`.
            ValueError: If file format is unsupported or loading fails.
        """
        if not self._connected:
            raise ConnectionError("LocalFileConnector is not connected. Call connect() first.")

        resolved_path = os.path.abspath(self.file_path)
        effective_format = self._determine_format(resolved_path)

        if effective_format == "csv":
            df = self._read_csv(resolved_path)
        elif effective_format == "json":
            df = self._read_json(resolved_path)
        elif effective_format == "txt":
            df = self._read_txt(resolved_path)
        else:
            raise ValueError(f"Unsupported file format: {effective_format}")

        if isinstance(query, str) and query.strip():
            try:
                df = df.query(query)
            except Exception as exc:
                raise ValueError(f"Failed to apply pandas query filter: {exc}") from exc

        return df

    def disconnect(self) -> None:
        """No-op for local files; marks the connector as disconnected."""
        self._connected = False

    # ===== Private helpers =====
    def _determine_format(self, path: str) -> str:
        if self.file_format in {"csv", "json", "txt"}:
            return self.file_format

        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext in {".csv"}:
            return "csv"
        if ext in {".json", ".jsonl"}:
            return "json"
        if ext in {".txt"}:
            return "txt"
        raise ValueError(
            "Unable to infer file format from extension. Specify file_format as 'csv', 'json', or 'txt'."
        )

    def _read_csv(self, path: str) -> pd.DataFrame:
        read_kwargs = {**self.pandas_read_kwargs}
        try:
            return pd.read_csv(path, **read_kwargs)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file '{path}': {exc}") from exc

    def _read_json(self, path: str) -> pd.DataFrame:
        read_kwargs = {**self.pandas_read_kwargs}
        # Support JSON Lines by default if extension is .jsonl or caller sets lines=True
        _, ext = os.path.splitext(path)
        if ext.lower() == ".jsonl" and "lines" not in read_kwargs:
            read_kwargs["lines"] = True
        try:
            return pd.read_json(path, **read_kwargs)
        except ValueError:
            # Fallback: if file is a standard JSON array but pandas infers wrong, try orient='records'
            try:
                read_kwargs.setdefault("orient", "records")
                return pd.read_json(path, **read_kwargs)
            except Exception as exc:
                raise ValueError(f"Failed to read JSON file '{path}': {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to read JSON file '{path}': {exc}") from exc


    def _read_txt(self, path: str) -> pd.DataFrame:
        # Extract custom options without polluting downstream pandas readers
        as_blob = bool(self.pandas_read_kwargs.pop("as_blob", False))
        column_name = str(self.pandas_read_kwargs.pop("column_name", "text"))
        encoding = self.pandas_read_kwargs.pop("encoding", None)

        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
        except Exception as exc:
            raise ValueError(f"Failed to read TXT file '{path}': {exc}") from exc

        if as_blob:
            return pd.DataFrame([{ "content": content }])

        # Default: one row per line
        lines = content.splitlines()
        return pd.DataFrame({ column_name: lines })


# Register the plugin for easy retrieval via PluginManager
try:
    PluginManager.register_plugin("local_file", LocalFileConnector)
except ValueError:
    # Allow idempotent imports during tests or reloads
    pass


# uv run python -m src.plugins_data_access.implementations.local_file_plugin
if __name__ == "__main__":
    PluginManager.register_plugin("local_file", LocalFileConnector)
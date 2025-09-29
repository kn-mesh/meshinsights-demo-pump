"""
Generic REST API connector plugin.

Exposes ``RestAPIConnector`` – a reusable implementation of :class:`DBConnector` that
works with any REST API and supports the following authentication mechanisms:
- username/password via HTTP Basic Auth
- bearer/token
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple, Union
from urllib.parse import urljoin
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth


from src.core.plugins.data_access.data_access_plugin_core import (
    DBConnector,
    PluginManager,
    RequestSpec,
)


ISO_FMT = "%Y-%m-%dT%H:%M:%S"
AuthType = Literal["none", "basic", "token"]


class RestAPIConnector(DBConnector):
    """Generic REST connector with optional Basic or token authentication."""

    def __init__(
        self,
        base_url: str,
        auth_type: AuthType = "none",
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        token_location: Literal["header", "query"] = "header",
        token_header: str = "Authorization",
        token_prefix: str = "Bearer ",
        token_param_name: str = "access_token",
        default_timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        retry_backoff_max: float = 8.0,
        max_concurrent: int = 8,
        tenant: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if requests is None:
            raise RuntimeError("The 'requests' package is required for RestAPIConnector")

        auth_type_l = auth_type.lower()
        if auth_type_l not in {"none", "basic", "token"}:
            raise ValueError(f"Unsupported auth_type: {auth_type}")

        if auth_type_l == "basic":
            if not username or not password:
                raise ValueError("Username and password are required for basic authentication")
        if auth_type_l == "token":
            if not token:
                raise ValueError("Token value is required for token authentication")
            if token_location not in {"header", "query"}:
                raise ValueError("token_location must be either 'header' or 'query'")

        self.base_url = base_url.rstrip("/") + "/"
        self.auth_type: AuthType = auth_type_l  # type: ignore[assignment]
        self.username = username
        self.password = password
        self.token = token
        self.token_location = token_location
        self.token_header = token_header
        self.token_prefix = token_prefix
        self.token_param_name = token_param_name
        self.default_timeout = float(default_timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.retry_backoff_max = float(retry_backoff_max)
        self.tenant = tenant
        self.default_headers = default_headers or {"Accept": "application/json"}

        self._session: Optional["requests.Session"] = None
        self._connected = False
        self._lock = threading.RLock()
        self._sema = threading.Semaphore(max_concurrent if max_concurrent and max_concurrent > 0 else 1)
        self._basic_auth: Optional[HTTPBasicAuth] = None
        if self.auth_type == "basic" and HTTPBasicAuth is not None:
            self._basic_auth = HTTPBasicAuth(username, password)  # type: ignore[arg-type]

    @classmethod
    def pool_key_fields(cls) -> Iterable[str]:
        return ["base_url", "tenant", "auth_type", "token_location", "token_header", "token_param_name"]

    def connect(self) -> Tuple[str, Any]:
        with self._lock:
            if not self._connected or self._session is None:
                sess = requests.Session()
                for key, value in (self.default_headers or {}).items():
                    sess.headers.setdefault(key, value)
                if self.auth_type == "basic" and self._basic_auth is not None:
                    sess.auth = self._basic_auth
                if self.auth_type == "token" and self.token and self.token_location == "header":
                    header_value = self._format_token()
                    if self.token_header:
                        sess.headers.setdefault(self.token_header, header_value)
                self._session = sess
                self._connected = True
        return self.base_url, self._session

    def execute_query(self, query: str) -> pd.DataFrame:
        raise NotImplementedError("RestAPIConnector: use execute_request(RequestSpec) for REST calls")

    def execute_request(self, request: Union[RequestSpec, Dict[str, Any], str]) -> pd.DataFrame:  # type: ignore[override]
        if isinstance(request, str):
            spec: RequestSpec = {  # type: ignore[assignment]
                "method": "GET",
                "path": request,
                "params": {},
                "headers": {},
            }
        else:
            spec = request  # type: ignore[assignment]

        base_url, sess = self.connect()
        assert sess is not None

        paginate = (spec.get("paginate") or {}) if isinstance(spec, Mapping) else {}
        if isinstance(paginate, Mapping) and paginate.get("type") == "date_window":
            frames: list[pd.DataFrame] = []
            for params in self._iterate_date_windows(spec):
                spec_win: Dict[str, Any] = dict(spec)
                spec_win["params"] = params
                frames.append(self._single_request_to_frame(sess, base_url, spec_win))
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        return self._single_request_to_frame(sess, base_url, spec)

    def disconnect(self) -> None:
        with self._lock:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:
                    pass
            self._session = None
            self._connected = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _single_request_to_frame(
        self,
        sess: "requests.Session",
        base_url: str,
        spec: Mapping[str, Any],
    ) -> pd.DataFrame:
        method = (spec.get("method") or "GET").upper()
        path = (spec.get("path") or "/").lstrip("/")
        url = urljoin(base_url, path)

        params = self._apply_query_auth(dict(spec.get("params") or {}))
        headers = dict(spec.get("headers") or {})
        if self.auth_type == "token" and self.token and self.token_location == "header":
            header_value = self._format_token()
            if self.token_header and self.token_header not in headers:
                headers[self.token_header] = header_value

        json_body: Optional[Dict[str, Any]] = spec.get("json") or None
        data_body: Optional[Dict[str, Any]] = spec.get("data") or None
        timeout = float(spec.get("timeout", self.default_timeout))
        raw_response = bool(spec.get("raw_response"))

        payload_desc = f"{method} {url}"
        attempts = 0
        backoff = self.retry_backoff

        while True:
            attempts += 1
            try:
                with self._sema:
                    resp = sess.request(
                        method=method,
                        url=url,
                        params=params if params else None,
                        headers=headers if headers else None,
                        json=json_body,
                        data=data_body,
                        timeout=timeout,
                    )

                if resp.status_code == 204:
                    return pd.DataFrame()

                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    if attempts <= self.max_retries:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                backoff = max(backoff, float(retry_after))
                            except Exception:
                                pass
                        time.sleep(min(backoff, self.retry_backoff_max))
                        backoff = min(self.retry_backoff_max, backoff * 2)
                        continue
                    raise ConnectionError(f"HTTP {resp.status_code} after {attempts} attempts for {payload_desc}")

                if 400 <= resp.status_code < 500:
                    raise ValueError(f"HTTP {resp.status_code} for {payload_desc}: {resp.text[:200]}")

                content_type = resp.headers.get("Content-Type", "application/json")
                if "json" in content_type:
                    js = resp.json()
                else:
                    try:
                        js = resp.json()
                    except Exception:
                        return pd.DataFrame([{"raw": resp.text}])

                if raw_response:
                    return pd.DataFrame([{"response": js}])

                response_path = spec.get("response_path")
                if response_path:
                    data = self._walk_path(js, list(response_path))
                else:
                    data = js
                return self._make_frame(data)

            except Exception as exc:
                if attempts <= self.max_retries and isinstance(
                    exc,
                    (
                        requests.Timeout,
                        requests.ConnectionError,
                        ConnectionError,
                    ),
                ):
                    time.sleep(min(backoff, self.retry_backoff_max))
                    backoff = min(self.retry_backoff_max, backoff * 2)
                    continue
                raise

    def _iterate_date_windows(self, spec: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
        paginate = spec.get("paginate") or {}
        params = dict(spec.get("params") or {})
        start_key = paginate.get("start_param", "startTime")
        end_key = paginate.get("end_param", "endTime")
        window_hours = float(paginate.get("window_hours", 6))
        overlap_minutes = float(paginate.get("overlap_minutes", 0))

        start_raw = params.get(start_key)
        end_raw = params.get(end_key)
        if not start_raw or not end_raw:
            yield self._apply_query_auth(params)
            return

        start_dt = self._parse_dt(str(start_raw))
        end_dt = self._parse_dt(str(end_raw))
        if start_dt is None or end_dt is None or end_dt <= start_dt:
            yield self._apply_query_auth(params)
            return

        window = timedelta(hours=window_hours)
        overlap = timedelta(minutes=overlap_minutes)
        current = start_dt
        while current < end_dt:
            cur_end = min(end_dt, current + window)
            segment = dict(params)
            segment[start_key] = current.strftime(ISO_FMT)
            segment[end_key] = cur_end.strftime(ISO_FMT)
            yield self._apply_query_auth(segment)
            if cur_end == end_dt:
                break
            current = cur_end - overlap

    def _apply_query_auth(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.auth_type == "token" and self.token and self.token_location == "query":
            if self.token_param_name and self.token_param_name not in params:
                params = dict(params)
                params[self.token_param_name] = self.token
        return params

    @staticmethod
    def _walk_path(root: Any, path: Iterable[str]) -> Any:
        current = root
        for key in path:
            if isinstance(current, Mapping):
                current = current.get(key)
            else:
                return None
        return current

    def _make_frame(self, data: Any) -> pd.DataFrame:
        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            if all(isinstance(item, Mapping) for item in data):
                return pd.DataFrame(data)
            return pd.DataFrame({"value": data})

        if isinstance(data, Mapping):
            rows = [{"key": key, "value": value} for key, value in data.items()]
            return pd.DataFrame(rows)

        if data is None:
            return pd.DataFrame()

        return pd.DataFrame([{"value": data}])

    @staticmethod
    def _parse_dt(value: str) -> Optional[datetime]:
        for fmt in (ISO_FMT, "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    def _format_token(self) -> str:
        if self.token is None:
            return ""
        return f"{self.token_prefix}{self.token}" if self.token_prefix else str(self.token)


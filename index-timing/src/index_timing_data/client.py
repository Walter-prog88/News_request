from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import requests

from .models import ApiTable, QuerySpec


class TwoShareError(RuntimeError):
    """Base exception that never embeds credentials."""


class TwoShareTransportError(TwoShareError):
    """Network, HTTP, or JSON transport failure."""


class TwoShareApiError(TwoShareError):
    """The server returned a non-zero API code."""


class TwoShareSchemaError(TwoShareError):
    """The response was successful but structurally invalid."""


def _redact(text: str, token: str) -> str:
    return text.replace(token, "<redacted>") if token else text


class TwoShareClient:
    """Small HTTPS client for the documented Tushare-compatible proxy."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://jiaoch.site",
        timeout_seconds: float = 35.0,
        session: Any | None = None,
    ) -> None:
        cleaned_token = token.strip()
        if not cleaned_token:
            raise ValueError("a non-empty token is required")

        cleaned_url = base_url.rstrip("/")
        parsed = urlparse(cleaned_url)
        if parsed.scheme != "https":
            raise ValueError("2Share base_url must use HTTPS")
        if not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("2Share base_url must be an HTTPS origin or base path")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._token = cleaned_token
        self.base_url = cleaned_url
        self.timeout_seconds = float(timeout_seconds)
        self._session = session or requests.Session()

    @classmethod
    def from_env(
        cls,
        *,
        token_env: str = "TWOSHARE_TOKEN",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> TwoShareClient:
        token = os.getenv(token_env, "")
        if not token:
            raise ValueError(f"{token_env} is not set")
        resolved_url = base_url or os.getenv("TWOSHARE_BASE_URL", "https://jiaoch.site")
        return cls(token, base_url=resolved_url, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, token=<redacted>)"
        )

    def query(self, spec: QuerySpec) -> ApiTable:
        body: dict[str, Any] = {
            "api_name": spec.api_name,
            "token": self._token,
            "params": dict(spec.params),
        }
        if spec.fields:
            body["fields"] = ",".join(spec.fields)

        url = f"{self.base_url}/{spec.api_name}"
        try:
            response = self._session.post(url, json=body, timeout=self.timeout_seconds)
        except Exception as exc:
            safe = _redact(str(exc), self._token)
            raise TwoShareTransportError(
                f"transport failure for {spec.dataset_id}: {type(exc).__name__}: {safe}"
            ) from exc

        if not 200 <= int(response.status_code) < 300:
            raise TwoShareTransportError(
                f"HTTP {response.status_code} for {spec.dataset_id}"
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise TwoShareTransportError(
                f"non-JSON response for {spec.dataset_id}"
            ) from exc

        if not isinstance(payload, dict):
            raise TwoShareSchemaError(f"response root is not an object for {spec.dataset_id}")

        code = payload.get("code")
        if isinstance(code, bool) or not isinstance(code, int):
            raise TwoShareSchemaError(f"response code is not an integer for {spec.dataset_id}")
        message = str(payload.get("msg") or "")
        if code != 0:
            safe_message = _redact(message, self._token)
            raise TwoShareApiError(
                f"API code {code} for {spec.dataset_id}: {safe_message[:300]}"
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            raise TwoShareSchemaError(f"missing data object for {spec.dataset_id}")
        fields = data.get("fields")
        items = data.get("items")
        if not isinstance(fields, list) or any(not isinstance(name, str) for name in fields):
            raise TwoShareSchemaError(f"invalid fields for {spec.dataset_id}")
        if not isinstance(items, list):
            raise TwoShareSchemaError(f"invalid items for {spec.dataset_id}")
        if len(set(fields)) != len(fields):
            raise TwoShareSchemaError(f"duplicate response fields for {spec.dataset_id}")

        normalized_items: list[tuple[Any, ...]] = []
        for row_index, row in enumerate(items):
            if not isinstance(row, list) or len(row) != len(fields):
                raise TwoShareSchemaError(
                    f"row width mismatch for {spec.dataset_id} at row {row_index}"
                )
            normalized_items.append(tuple(row))

        return ApiTable(
            spec=spec,
            fields=tuple(fields),
            items=tuple(normalized_items),
            response_code=code,
            message=message,
            http_status=int(response.status_code),
            received_at_utc=datetime.now(UTC).isoformat(),
        )


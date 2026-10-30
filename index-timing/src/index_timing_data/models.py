from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_ENDPOINT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SECRET_KEYS = {"token", "api_key", "apikey", "secret", "password"}


def _assert_json_serializable(value: Any, *, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON data") from exc


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """One immutable API query definition without credentials."""

    dataset_id: str
    api_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    fields: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.dataset_id):
            raise ValueError(f"unsafe dataset_id: {self.dataset_id!r}")
        if not _SAFE_ENDPOINT.fullmatch(self.api_name):
            raise ValueError(f"unsafe api_name: {self.api_name!r}")

        lowered = {str(key).lower() for key in self.params}
        forbidden = sorted(lowered & _SECRET_KEYS)
        if forbidden:
            raise ValueError(f"credentials are forbidden in query params: {forbidden}")

        if len(set(self.fields)) != len(self.fields):
            raise ValueError(f"duplicate fields in {self.dataset_id}")
        if any(not isinstance(name, str) or not name for name in self.fields):
            raise ValueError(f"fields must be non-empty strings in {self.dataset_id}")

        _assert_json_serializable(dict(self.params), label=f"{self.dataset_id}.params")
        _assert_json_serializable(dict(self.metadata), label=f"{self.dataset_id}.metadata")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QuerySpec:
        raw_fields = value.get("fields") or ()
        if isinstance(raw_fields, str):
            raw_fields = tuple(part.strip() for part in raw_fields.split(",") if part.strip())
        elif isinstance(raw_fields, Sequence):
            raw_fields = tuple(str(part) for part in raw_fields)
        else:
            raise ValueError("fields must be a comma-separated string or a list")

        return cls(
            dataset_id=str(value["dataset_id"]),
            api_name=str(value["api_name"]),
            params=dict(value.get("params") or {}),
            fields=tuple(raw_fields),
            metadata=dict(value.get("metadata") or {}),
        )

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "api_name": self.api_name,
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }
        if self.fields:
            payload["fields"] = list(self.fields)
        return payload


@dataclass(frozen=True, slots=True)
class ApiTable:
    """Validated table returned by one 2Share-compatible endpoint."""

    spec: QuerySpec
    fields: tuple[str, ...]
    items: tuple[tuple[Any, ...], ...]
    response_code: int
    message: str
    http_status: int
    received_at_utc: str

    @property
    def row_count(self) -> int:
        return len(self.items)

    def records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.fields, row, strict=True)) for row in self.items]

    def response_payload(self) -> dict[str, Any]:
        return {
            "code": self.response_code,
            "msg": self.message,
            "data": {
                "fields": list(self.fields),
                "items": [list(row) for row in self.items],
            },
        }

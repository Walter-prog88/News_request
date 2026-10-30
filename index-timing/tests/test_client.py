from __future__ import annotations

import unittest

from index_timing_data.client import (
    TwoShareApiError,
    TwoShareClient,
    TwoShareSchemaError,
)
from index_timing_data.models import QuerySpec


class FakeResponse:
    status_code = 200

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, object], float]] = []

    def post(self, url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        self.calls.append((url, json, timeout))
        return FakeResponse(self.payload)


class ClientTests(unittest.TestCase):
    def test_success_and_token_is_not_in_repr(self) -> None:
        session = FakeSession(
            {
                "code": 0,
                "msg": "success",
                "data": {
                    "fields": ["date", "value"],
                    "items": [["20260101", 1.5]],
                },
            }
        )
        client = TwoShareClient("very-secret-token", session=session)
        table = client.query(QuerySpec("sample", "shibor", {"start_date": "20260101"}))

        self.assertEqual(table.row_count, 1)
        self.assertEqual(table.records(), [{"date": "20260101", "value": 1.5}])
        self.assertNotIn("very-secret-token", repr(client))
        self.assertEqual(session.calls[0][1]["token"], "very-secret-token")

    def test_http_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            TwoShareClient("secret", base_url="http://jiaoch.site")

    def test_nonzero_api_code_is_fail_closed_and_redacted(self) -> None:
        session = FakeSession(
            {"code": 40203, "msg": "bad token very-secret-token", "data": None}
        )
        client = TwoShareClient("very-secret-token", session=session)
        with self.assertRaises(TwoShareApiError) as caught:
            client.query(QuerySpec("sample", "daily"))
        self.assertNotIn("very-secret-token", str(caught.exception))

    def test_row_width_mismatch_is_rejected(self) -> None:
        session = FakeSession(
            {"code": 0, "msg": "", "data": {"fields": ["a", "b"], "items": [[1]]}}
        )
        client = TwoShareClient("secret", session=session)
        with self.assertRaises(TwoShareSchemaError):
            client.query(QuerySpec("sample", "daily"))

    def test_credentials_are_forbidden_in_params(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            QuerySpec("sample", "daily", {"token": "do-not-store"})


if __name__ == "__main__":
    unittest.main()

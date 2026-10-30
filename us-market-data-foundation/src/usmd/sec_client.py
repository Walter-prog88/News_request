"""Bounded SEC JSON transport; the caller owns raw-file archival and lake writes.

Every public getter returns ``(payload, metadata)``. ``fetch_json`` metadata
contains the original JSON entity bytes (after HTTP content decoding), their
SHA-256, source URL and UTC retrieval time. Nothing is written to disk here.

``get_submissions`` merges historical pages into ``filings.recent`` while
preserving every SEC column, including ``acceptanceDateTime``. Its top-level
hash/raw_bytes describe the primary document, not the merged object; archive
each entry in ``metadata['documents']`` to reproduce that merge. Current SEC
downloads are snapshots, not a complete archive of past API revisions.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
import re
import threading
import time
from urllib.parse import urlparse

import requests

from .research_credentials import load_credentials


TICKERS_URL = 'https://www.sec.gov/files/company_tickers_exchange.json'
DATA_BASE = 'https://data.sec.gov'
DEFAULT_USER_AGENT = 'USMarketDataFoundation/1.0 (financial research)'
MAX_RESPONSE_BYTES = 80 * 1024 * 1024
MAX_ATTEMPTS = 3
MAX_HISTORY_FILES = 100
REQUEST_INTERVAL = 0.5
_request_lock = threading.Lock()
_last_request_started = None


class SecRefusalError(RuntimeError):
    """SEC refused access; callers should stop rather than change transports."""


def normalize_cik(cik):
    """Return a validated ten-digit CIK, rejecting floats and empty/zero IDs."""
    if isinstance(cik, bool) or not isinstance(cik, (str, int)):
        raise ValueError('CIK must be a positive integer or decimal string')
    value = str(cik).strip()
    if value.upper().startswith('CIK'):
        value = value[3:]
    if not re.fullmatch(r'[0-9]{1,10}', value) or int(value) == 0:
        raise ValueError('CIK must contain 1 to 10 digits and be nonzero')
    return value.zfill(10)


def _validate_url(url):
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or parsed.hostname not in {'data.sec.gov', 'www.sec.gov'}
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.fragment):
        raise ValueError('Only HTTPS SEC source URLs without credentials are allowed')


def _retry_delay(response, attempt):
    delay = float(2 ** attempt)
    retry_after = response.headers.get('Retry-After') if response is not None else None
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except ValueError:
            try:
                when = parsedate_to_datetime(retry_after)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                delay = max(delay, (when - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(30.0, max(0.0, delay))


def _column_rows(columns):
    """Validate SEC's columnar arrays before joining them by accession."""
    if not isinstance(columns, dict) or not isinstance(columns.get('accessionNumber'), list):
        raise ValueError('SEC submissions page lacks accessionNumber array')
    count = len(columns['accessionNumber'])
    if any(not isinstance(values, list) or len(values) != count for values in columns.values()):
        raise ValueError('SEC submissions columns have inconsistent lengths')
    return [dict(zip(columns, values)) for values in zip(*columns.values())]


def merge_submissions_documents(documents):
    """Reconstruct merged submissions using verified original SEC documents.

    ``documents`` starts with the main submissions object, followed by its
    historical columnar pages in the order they were fetched. Main/earlier
    documents win duplicate accessions, consistently with get_submissions().
    Optional columns are retained and filled with None where absent. Inputs are
    never mutated; the result is a deep copy of the main object plus merged rows.
    This function does not verify hashes: the archive reader must do that first.
    """
    if not isinstance(documents, list) or not documents or any(not isinstance(d, dict) for d in documents):
        raise ValueError('Submissions reconstruction needs a main document and optional history objects')
    result = deepcopy(documents[0])
    filings = result.get('filings')
    if not isinstance(filings, dict):
        raise ValueError('SEC submissions lacks filings object')
    recent = filings.get('recent')
    rows = _column_rows(recent)
    columns = list(recent)
    for historical in documents[1:]:
        rows.extend(_column_rows(historical))
        columns.extend(key for key in historical if key not in columns)
    seen, unique_rows = set(), []
    for row in rows:
        accession = row['accessionNumber']
        if not isinstance(accession, str) or not accession:
            raise ValueError('SEC submission has invalid accession number')
        if accession not in seen:
            seen.add(accession)
            unique_rows.append(row)
    filings['recent'] = {key: [deepcopy(row.get(key)) for row in unique_rows] for key in columns}
    return result


class SecClient:
    """Serial SEC client limited to two request starts/second across instances.

    Use an actual contact-bearing ``user_agent`` or SEC_USER_AGENT when one has
    been configured. The default identifies this application without inventing
    an email. HTTP 401/403 stops immediately; 429/5xx and transient network
    errors receive at most two retries. No browser or alternate-host fallback.
    """

    def __init__(self, user_agent=None, timeout=45):
        user_agent = (user_agent or os.environ.get('SEC_USER_AGENT')
                      or load_credentials().get('SEC_USER_AGENT') or DEFAULT_USER_AGENT)
        if not isinstance(user_agent, str) or not user_agent.strip() or '\n' in user_agent or '\r' in user_agent:
            raise ValueError('SEC user-agent must be a nonempty single line')
        if isinstance(timeout, bool) or not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError('timeout must be a positive finite number of seconds')
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent.strip(), 'Accept': 'application/json'})

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _request(self, url):
        # Hold through body consumption: no concurrent network calls by clients
        # in this process, and request starts (including retries) are throttled.
        global _last_request_started
        with _request_lock:
            now = time.monotonic()
            if _last_request_started is not None:
                pause = REQUEST_INTERVAL - (now - _last_request_started)
                if pause > 0:
                    time.sleep(pause)
            _last_request_started = time.monotonic()
            with self.session.get(url, timeout=(min(10, self.timeout), self.timeout),
                                  stream=True, allow_redirects=False) as response:
                status = response.status_code
                if status in (401, 403):
                    raise SecRefusalError(f'SEC refused HTTP {status}; no alternate transport attempted')
                if 300 <= status < 400:
                    raise ValueError('Unexpected SEC redirect; not followed')
                response.raise_for_status()
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > MAX_RESPONSE_BYTES:
                    raise ValueError('SEC response exceeds JSON byte limit')
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise ValueError('SEC response exceeds JSON byte limit')
                    chunks.append(chunk)
                return b''.join(chunks), dict(response.headers), status

    def fetch_json(self, url):
        """Return a JSON object and source metadata, including immutable bytes."""
        _validate_url(url)
        for attempt in range(MAX_ATTEMPTS):
            try:
                body, headers, status = self._request(url)
                break
            except requests.HTTPError as exc:
                response = exc.response
                status = response.status_code if response is not None else None
                if status != 429 and (status is None or not 500 <= status <= 599):
                    raise
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_retry_delay(response, attempt))
            except (requests.Timeout, requests.ConnectionError):
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_retry_delay(None, attempt))
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError('SEC JSON response must be an object')
        return payload, {
            'source_url': url,
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'sha256': hashlib.sha256(body).hexdigest(),
            'raw_bytes': body,
            'byte_count': len(body),
            'status_code': status,
            'etag': headers.get('ETag'),
            'last_modified': headers.get('Last-Modified'),
        }

    def get_tickers(self):
        """Return the raw SEC exchange mapping ({fields, data}), plus metadata.

        A 403 is surfaced without trying the other mapping endpoint. Curated
        CIKs can instead be verified with their public submissions documents.
        """
        payload, metadata = self.fetch_json(TICKERS_URL)
        if not isinstance(payload.get('fields'), list) or not isinstance(payload.get('data'), list):
            raise ValueError('Unexpected SEC ticker/exchange mapping schema')
        return payload, metadata

    def get_companyfacts(self, cik):
        cik = normalize_cik(cik)
        payload, metadata = self.fetch_json(f'{DATA_BASE}/api/xbrl/companyfacts/CIK{cik}.json')
        if normalize_cik(payload.get('cik')) != cik or not isinstance(payload.get('facts'), dict):
            raise ValueError('SEC companyfacts entity or facts schema does not match request')
        return payload, metadata

    def get_submissions(self, cik, include_history=True, since=None):
        """Return submissions with historical arrays merged into filings.recent.

        ``since`` (YYYY-MM-DD) skips history files whose filingTo precedes that
        date; records within fetched pages are kept intact. The default loads
        every advertised history page for this one company, up to 100 pages.
        Recent records win if an accession appears in more than one page.
        A page error fails the call rather than claiming complete coverage.
        """
        cik = normalize_cik(cik)
        if since is not None:
            if not isinstance(since, str) or len(since) != 10:
                raise ValueError('since must be an ISO date (YYYY-MM-DD)')
            date.fromisoformat(since)
        payload, primary = self.fetch_json(f'{DATA_BASE}/submissions/CIK{cik}.json')
        if normalize_cik(payload.get('cik')) != cik:
            raise ValueError('SEC submissions entity does not match request')
        filings = payload.get('filings')
        if not isinstance(filings, dict):
            raise ValueError('SEC submissions lacks filings object')
        recent = filings.get('recent')
        _column_rows(recent)
        documents = [primary]
        source_payloads = [payload]
        history_files = filings.get('files', [])
        if not isinstance(history_files, list):
            raise ValueError('SEC submissions history index must be a list')
        selected = []
        if include_history:
            for item in history_files:
                if not isinstance(item, dict):
                    raise ValueError('Malformed SEC submissions history index entry')
                if since and item.get('filingTo') and date.fromisoformat(item['filingTo']) < date.fromisoformat(since):
                    continue
                name = item.get('name', '')
                if not re.fullmatch(rf'CIK{cik}-submissions-[0-9]+\.json', name):
                    raise ValueError('Unsafe or mismatched SEC submissions history filename')
                selected.append(name)
            selected = list(dict.fromkeys(selected))
            if len(selected) > MAX_HISTORY_FILES:
                raise ValueError('SEC company history exceeds the bounded page count')
            for name in selected:
                historical, metadata = self.fetch_json(f'{DATA_BASE}/submissions/{name}')
                _column_rows(historical)
                source_payloads.append(historical)
                documents.append(metadata)
        payload = merge_submissions_documents(source_payloads)
        metadata = {**primary, 'documents': documents,
                    'history_included': bool(include_history), 'history_since': since,
                    'history_files_advertised': len(history_files),
                    'history_files_loaded': len(selected),
                    'filing_count': len(payload['filings']['recent']['accessionNumber'])}
        return payload, metadata

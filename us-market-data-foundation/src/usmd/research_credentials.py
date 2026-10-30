"""Local research credentials; never include values in reports or lake metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat

ALLOWED_KEYS = frozenset({
    'SEC_USER_AGENT', 'CENSUS_API_KEY', 'EIA_API_KEY',
    'ALPHA_VANTAGE_API_KEY', 'ALPACA_API_KEY_ID', 'ALPACA_API_SECRET_KEY',
})
DEFAULT_PATH = Path(__file__).resolve().parents[2] / 'config' / 'local_credentials.json'


def load_credentials(path=None):
    """Read a private local file, with environment values taking precedence.

    An absent file is allowed; malformed or group/world-readable files fail
    closed. Callers should report only whether a credential is configured.
    """
    target = Path(path) if path else DEFAULT_PATH
    values = {}
    if target.exists():
        if target.is_symlink() or stat.S_IMODE(target.stat().st_mode) & 0o077:
            raise PermissionError('Research credential file must be private (0600) and not a symlink')
        data = json.loads(target.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or any(key not in ALLOWED_KEYS for key in data):
            raise ValueError('Research credential file has unsupported fields')
        values = data
    for key in ALLOWED_KEYS:
        if os.environ.get(key):
            values[key] = os.environ[key]
    if any(not isinstance(value, str) or not value.strip() or '\n' in value or '\r' in value
           for value in values.values()):
        raise ValueError('Research credentials must be nonempty single-line strings')
    return {key: value.strip() for key, value in values.items()}

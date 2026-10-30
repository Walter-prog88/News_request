"""Portable date and output-path helpers; no connection to a data lake."""
from __future__ import annotations
import datetime as dt
import os
from pathlib import Path

OFFICIAL_LEDGER_DIR = Path(__file__).resolve().parent.parent / 'runtime' / 'ledger'
DEFAULT_LEDGER_DIR = Path(os.environ.get('LENS_LEDGER_DIR', str(OFFICIAL_LEDGER_DIR)))

def today():
    value = os.environ.get('LENS_TODAY')
    return dt.date.fromisoformat(value) if value else dt.datetime.now(dt.timezone.utc).date()

def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

def iso_utc(moment):
    return moment.strftime('%Y-%m-%dT%H:%M:%SZ')

def stamp_utc(moment):
    return moment.strftime('%Y%m%dT%H%M%SZ')

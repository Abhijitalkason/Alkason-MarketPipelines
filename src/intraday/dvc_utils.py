"""DVC operations — failures RAISE (v3_supplementry §12.3, B-6).

The v1 bug was silent DVC staleness (failures logged at debug, remote on /tmp).
Here a non-zero exit raises DVCError; the remote lives on durable storage set
by DVC_REMOTE_URL. Nothing in v3 swallows a DVC failure.
"""

from __future__ import annotations

import logging
import os
import subprocess

from src.intraday import ROOT

logger = logging.getLogger(__name__)


class DVCError(RuntimeError):
    pass


def _run(args: list[str]) -> str:
    proc = subprocess.run(["dvc", *args], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise DVCError(f"`dvc {' '.join(args)}` failed (exit {proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout.strip()


def is_initialized() -> bool:
    return (ROOT / ".dvc").exists()


def add_and_push(paths: list[str]) -> str:
    """dvc add each path then push to the configured remote. Raises on any
    failure. Returns the current DVC rev (git SHA of the .dvc pointers)."""
    if not is_initialized():
        raise DVCError("DVC not initialised — run `dvc init` and configure DVC_REMOTE_URL first")
    remote = os.getenv("DVC_REMOTE_URL", "")
    if not remote:
        raise DVCError("DVC_REMOTE_URL not set — remote must be durable storage, never /tmp")
    for p in paths:
        if (ROOT / p).exists():
            _run(["add", p])
    _run(["push"])
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def current_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"

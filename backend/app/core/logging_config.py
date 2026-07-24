"""Backend-wide logging configuration.

The Tauri host spawns the Python sidecar with stdout/stderr redirected to /dev/null,
so relying on console output is useless in production. We always write a rotating
log file to ``~/.justsay/logs/backend.log`` and also mirror to stderr (seen when
running uvicorn manually).
"""

import logging
import logging.handlers
import os
from pathlib import Path

from app.core.app_paths import resolve_app_data_root

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_log_file: Path | None = None


def _log_dir() -> Path:
    return resolve_app_data_root() / "logs"


def setup_logging() -> Path:
    """Configure root logger once. Returns the log file path for diagnostics."""
    global _log_file

    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "backend.log"

    root = logging.getLogger()
    already_configured = any(
        getattr(h, "_justsay_file", False) for h in root.handlers
    )
    if already_configured:
        return _log_file if _log_file is not None else log_file

    level_name = os.environ.get("JUSTSAY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    file_handler._justsay_file = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    for noisy in ("httpcore", "httpx", "urllib3", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _log_file = log_file
    logging.getLogger(__name__).info("Logging initialised — file: %s", log_file)
    return log_file


def log_file_path() -> Path:
    if _log_file is not None:
        return _log_file
    return _log_dir() / "backend.log"

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

from app.core.user_settings import SETTINGS_DIR

_LOG_DIR = SETTINGS_DIR / "logs"
_LOG_FILE = _LOG_DIR / "backend.log"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging() -> Path:
    """Configure root logger once. Returns the log file path for diagnostics."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # If something else (uvicorn) already set handlers, leave them but ensure our file handler exists.
    already_configured = any(
        getattr(h, "_justsay_file", False) for h in root.handlers
    )
    if already_configured:
        return _LOG_FILE

    level_name = os.environ.get("JUSTSAY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    file_handler._justsay_file = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    # Tame noisy third-party loggers.
    for noisy in ("httpcore", "httpx", "urllib3", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialised — file: %s", _LOG_FILE)
    return _LOG_FILE


def log_file_path() -> Path:
    return _LOG_FILE

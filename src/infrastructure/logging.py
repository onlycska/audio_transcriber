"""Centralized loguru logging configuration for the service."""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any

from loguru import logger as _loguru_logger


def _format_exception_traceback(exc: Any) -> str:
    """Format exception data from loguru record into traceback text."""
    tb = getattr(exc, "traceback", None)
    if tb is None:
        return str(exc)
    return "".join(traceback.format_tb(tb)).strip()


def _extract_source_from_traceback(exc: Any) -> str | None:
    """Extract source (file:function:line) from traceback when possible."""
    tb = getattr(exc, "traceback", None)
    if tb is None:
        return None
    extracted = traceback.extract_tb(tb)
    if not extracted:
        return None
    frame = extracted[-1]
    return f"{frame.filename}:{frame.name}:{frame.lineno}"


def _get_level_name(record: dict[str, Any]) -> str:
    """Get normalized level name from loguru record."""
    level_obj = record.get("level")
    name = getattr(level_obj, "name", None)
    return str(name or "INFO")


def _json_format(record: dict[str, Any]) -> str:
    """Format log record as minimal JSON for structured logging."""
    exc = record.get("exception")
    source = _extract_source_from_traceback(exc) if exc is not None else None
    if source is None:
        name = record.get("name", "")
        func = record.get("function", "")
        line = record.get("line", 0)
        source = f"{name}:{func}:{line}"

    time_val = record.get("time")
    if isinstance(time_val, datetime):
        timestamp = time_val.isoformat()
    else:
        timestamp = str(time_val or "")

    payload: dict[str, Any] = {
        "text": record.get("message", ""),
        "timestamp": timestamp,
        "level": _get_level_name(record),
        "source": source,
    }

    extra = record.get("extra") or {}
    if extra:
        payload["extra"] = dict(extra)

    if exc is not None:
        payload["exception"] = _format_exception_traceback(exc)

    return json.dumps(payload, ensure_ascii=False, default=str, separators=(", ", ": "))


def _text_format(record: dict[str, Any]) -> str:
    """Format log record as human-readable text with optional extra."""
    time_val = record.get("time")
    if isinstance(time_val, datetime):
        time_str = time_val.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    else:
        time_str = str(time_val or "")

    level_name = _get_level_name(record)
    msg = record.get("message", "")

    exc = record.get("exception")
    source = _extract_source_from_traceback(exc) if exc is not None else None
    if source is None:
        name = record.get("name", "")
        func = record.get("function", "")
        line = record.get("line", 0)
        source = f"{name}:{func}:{line}"

    out = f"{time_str} | {level_name: <8} | {source} - {msg}"

    extra = record.get("extra") or {}
    if extra:
        extra_str = " | ".join(f"{k}={v!r}" for k, v in extra.items())
        out += f" | extra: {extra_str}"

    if exc is not None:
        exc_str = _format_exception_traceback(exc).replace("\n", " | ")
        out += f" | {exc_str}"

    return out


def _is_json_enabled() -> bool:
    """Determine whether JSON logs should be enabled by default.

    Priority:
    1. Explicit ``LOG_JSON`` env var.
    2. Local/dev environment defaults to text logs.
    3. Other environments default to JSON logs.
    """
    explicit_value = os.getenv("LOG_JSON")
    if explicit_value is not None:
        return explicit_value.strip().lower() not in {"0", "false", "no"}

    app_env = os.getenv("APP_ENV", "").strip().lower()
    if app_env in {"local", "dev"}:
        return False
    return True


def _get_effective_level(level: str | None) -> str:
    """Resolve effective log level from argument or environment."""
    value = level or os.getenv("LOG_LEVEL", "INFO")
    return str(value).upper()


def _make_format_patcher(use_json: bool):
    """Create a patcher that stores formatted output into record field."""

    def _patch(record: Any) -> None:
        record["_formatted"] = _json_format(record) if use_json else _text_format(record)

    return _patch


def _configure_third_party_log_levels() -> None:
    """Set explicit log levels for third-party libraries."""
    for name, level in {
        "fastapi": logging.ERROR,
        "uvicorn": logging.ERROR,
        "uvicorn.error": logging.ERROR,
        "uvicorn.access": logging.ERROR,
        "sqlalchemy": logging.ERROR,
        "alembic": logging.ERROR,
        "httpx": logging.ERROR,
        "aiokafka": logging.ERROR,
        "kafka": logging.ERROR,
        "asyncio": logging.ERROR,
        "urllib3": logging.ERROR,
        "opentelemetry": logging.ERROR,
        "whisper": logging.ERROR,
        "pyannote": logging.ERROR,
        "torch": logging.ERROR,
    }.items():
        logging.getLogger(name).setLevel(level)


def configure_logging(*, level: str | None = None, json_format: bool | None = None) -> None:
    """Configure service logging once during application startup."""
    effective_level = _get_effective_level(level)
    use_json = json_format if json_format is not None else _is_json_enabled()

    _loguru_logger.configure(patcher=_make_format_patcher(use_json))
    _loguru_logger.remove()

    if use_json:
        _loguru_logger.add(
            sys.stderr,
            level=effective_level,
            format="{_formatted}",
            colorize=False,
            diagnose=False,
        )
    else:
        _loguru_logger.add(
            sys.stderr,
            level=effective_level,
            format="<level>{_formatted}</level>",
            colorize=True,
            diagnose=True,
        )

    _configure_third_party_log_levels()


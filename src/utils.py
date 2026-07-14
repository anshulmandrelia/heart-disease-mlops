"""Shared configuration, filesystem, and structured logging utilities."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


class JsonFormatter(logging.Formatter):
    """Render log records as JSON for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "latency_ms", "prediction", "error"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a process-safe logger with a single JSON stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration, resolving relative paths from repository root."""
    config_path = Path(path) if path else ROOT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(relative_path: str) -> Path:
    """Return and create the parent directory for a repository-relative path."""
    path = ROOT_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


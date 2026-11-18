"""Readable request logging for laptop 1's terminal (the evaluator watches it) plus a log file.

Console:  14:03:22  192.168.1.20   POST /predict   trip=00000040  OK   acc 87.5% (2620/2995) ...
File:     logs/api.log, same lines with dates, rotated at 2 MB; tracebacks go to the file only.

Never logged: request bodies, reference routes or labels, file paths of the data.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "pathsnap"


class _ConsoleFormatter(logging.Formatter):
    LEVEL_TAG = {logging.WARNING: "WARN ", logging.ERROR: "ERROR", logging.CRITICAL: "FATAL"}

    def format(self, record):
        tag = self.LEVEL_TAG.get(record.levelno, "")
        msg = record.getMessage()
        return f"{self.formatTime(record, '%H:%M:%S')}  {tag + '  ' if tag else ''}{msg}"


def setup(log_dir: Path) -> logging.Logger:
    log = logging.getLogger(LOGGER_NAME)
    if getattr(log, "_pathsnap_configured", False):
        return log
    log.setLevel(logging.INFO)
    log.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ConsoleFormatter())
    log.addHandler(console)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_dir / "api.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(fh)
    except OSError as exc:   # read-only checkout: console logging still works
        log.warning("Log file disabled (%s)", exc.strerror)

    log._pathsnap_configured = True
    return log


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"

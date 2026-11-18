"""API settings, read once from environment variables.

| Variable                     | Default                          | Meaning                                                   |
|------------------------------|----------------------------------|-----------------------------------------------------------|
| PATHSNAP_DATA_ROOT           | <repo>/Student folder/Student folder | folder holding the partition folders                  |
| PATHSNAP_SPLIT               | student_test                     | partition served to the UI (evaluator folder on demo day) |
| PATHSNAP_MODELS_DIR          | <repo>/models                    | classifier.joblib, classifier_config.json, hmm_config.json |
| PATHSNAP_EMISSION            | auto                             | auto | classifier | distance (see api/resources.py)      |
| PATHSNAP_REQUIRE_CLASSIFIER  | 0                                | 1 = refuse to start without B's exported classifier       |
| PATHSNAP_CORS_ORIGINS        | *                                | comma-separated list of allowed UI origins                |
| PATHSNAP_LOG_DIR             | <repo>/logs                      | request log file location                                 |

Relative paths are resolved against the repository root, so the API can be started from
any working directory and no path on any one laptop is hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EMISSION_MODES = ("auto", "classifier", "distance")


def _path(env: str, default: Path) -> Path:
    raw = os.environ.get(env)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


def _flag(env: str) -> bool:
    return os.environ.get(env, "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    data_root: Path
    split: str
    models_dir: Path
    emission: str
    require_classifier: bool
    cors_origins: tuple[str, ...]
    log_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        emission = os.environ.get("PATHSNAP_EMISSION", "auto").strip().lower()
        if emission not in EMISSION_MODES:
            raise ValueError(f"PATHSNAP_EMISSION must be one of {EMISSION_MODES}, got {emission!r}")
        origins = tuple(o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip())
        return cls(
            data_root=_path("PATHSNAP_DATA_ROOT", REPO_ROOT / "Student folder" / "Student folder"),
            split=os.environ.get("PATHSNAP_SPLIT", "student_test").strip(),
            models_dir=_path("PATHSNAP_MODELS_DIR", REPO_ROOT / "models"),
            emission=emission,
            require_classifier=_flag("PATHSNAP_REQUIRE_CLASSIFIER"),
            cors_origins=origins or ("*",),
            log_dir=_path("PATHSNAP_LOG_DIR", REPO_ROOT / "logs"),
        )

"""Everything loaded once at startup and shared by all requests (read-only after load).

* Trip catalogue  : trip folders of the served partition + cheap track-only summaries.
* HMM parameters  : role C's ``models/hmm_config.json`` via ``viterbi.HMMParams.from_json``.
* Classifier      : role B's ``models/classifier.joblib`` + ``models/classifier_config.json``
                    (an sklearn estimator/Pipeline exposing ``predict_proba`` and ``classes_``).

Emission modes (``PATHSNAP_EMISSION``):
* ``classifier`` : HMM emissions = log P(candidate is the true arc) from B's classifier.
                   If the classifier cannot be loaded, /predict answers 503 — never a fake model.
* ``distance``   : role C's baseline Gaussian emission on point-to-arc distance.
* ``auto``       : ``classifier`` when B's model is present, otherwise ``distance``; the mode in
                   use is reported by /health, in every /predict response and in the logs.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import viterbi
from src.network import haversine_m

from .config import Settings

log = logging.getLogger("pathsnap")

TRIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
INPUT_SUFFIXES = (".track", ".nodes", ".arcs")


@dataclass(frozen=True)
class TripInfo:
    trip_id: str
    n_points: int
    duration_s: float
    length_km: float
    bbox: tuple[tuple[float, float], tuple[float, float]]   # ((min_lat, min_lon), (max_lat, max_lon))
    has_reference: bool                                     # existence only; contents never read here


@dataclass
class Classifier:
    model: Any
    feature_columns: list[str]
    positive_index: int
    name: str

    def positive_proba(self, feats: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in feats.columns]
        if missing:
            raise ValueError(f"candidate table lacks classifier features: {missing}")
        return self.model.predict_proba(feats[self.feature_columns])[:, self.positive_index]


@dataclass
class Resources:
    settings: Settings
    trips: dict[str, TripInfo] = field(default_factory=dict)
    hmm_params: viterbi.HMMParams | None = None
    hmm_error: str | None = None
    classifier: Classifier | None = None
    classifier_error: str | None = None
    data_error: str | None = None

    # ---- derived state -------------------------------------------------------------
    @property
    def emission_mode(self) -> str:
        """The emission the decoder will actually use (``classifier`` or ``distance``)."""
        if self.settings.emission == "distance":
            return "distance"
        if self.settings.emission == "classifier" or self.settings.require_classifier:
            return "classifier"
        return "classifier" if self.classifier is not None else "distance"

    @property
    def blocking_problems(self) -> list[str]:
        """Reasons /predict cannot run. Empty list = pipeline ready."""
        problems = []
        if self.data_error:
            problems.append(self.data_error)
        if self.hmm_params is None:
            problems.append(self.hmm_error or "HMM config not loaded")
        if self.emission_mode == "classifier" and self.classifier is None:
            problems.append(self.classifier_error or "classifier not loaded")
        return problems

    def trip_dir(self, trip_id: str) -> Path:
        return self.settings.data_root / self.settings.split / trip_id

    # ---- loading -------------------------------------------------------------------
    @classmethod
    def load(cls, settings: Settings) -> "Resources":
        res = cls(settings)
        res._load_trips()
        res._load_hmm()
        res._load_classifier()
        return res

    def _load_trips(self):
        part = self.settings.data_root / self.settings.split
        if not part.is_dir():
            self.data_error = f"partition folder not found: {self.settings.split!r} under PATHSNAP_DATA_ROOT"
            log.error("Data: %s (%s)", self.data_error, part)
            return
        for d in sorted(p for p in part.iterdir() if p.is_dir()):
            tid = d.name
            if not TRIP_ID_RE.match(tid):
                continue
            base = d / tid
            if not all(base.with_suffix(s).is_file() for s in INPUT_SUFFIXES):
                log.warning("Data: skipping %s (missing .track/.nodes/.arcs)", tid)
                continue
            try:
                self.trips[tid] = _summarise_track(tid, base)
            except Exception as exc:  # unreadable track: skip the trip, keep serving the rest
                log.warning("Data: skipping %s (unreadable track: %s)", tid, exc)
        if not self.trips:
            self.data_error = f"no usable trips in partition {self.settings.split!r}"
            log.error("Data: %s", self.data_error)
        else:
            log.info("Data: %d trips in %r", len(self.trips), self.settings.split)

    def _load_hmm(self):
        path = self.settings.models_dir / "hmm_config.json"
        try:
            self.hmm_params = viterbi.HMMParams.from_json(path)
            log.info("HMM: %s", self.hmm_params)
        except FileNotFoundError:
            self.hmm_error = "models/hmm_config.json not found"
        except Exception as exc:
            self.hmm_error = f"hmm_config.json unreadable ({type(exc).__name__})"
        if self.hmm_error:
            log.error("HMM: %s", self.hmm_error)

    def _load_classifier(self):
        if self.settings.emission == "distance":
            log.info("Classifier: skipped (PATHSNAP_EMISSION=distance)")
            return
        model_path = self.settings.models_dir / "classifier.joblib"
        cfg_path = self.settings.models_dir / "classifier_config.json"
        if not model_path.is_file():
            self.classifier_error = "models/classifier.joblib not found (role B export pending)"
        else:
            try:
                self.classifier = _load_classifier(model_path, cfg_path)
                log.info("Classifier: %s on %d features", self.classifier.name,
                         len(self.classifier.feature_columns))
            except Exception as exc:
                self.classifier_error = f"classifier could not be loaded: {exc}"
        if self.classifier is None:
            if self.emission_mode == "classifier":
                log.error("Classifier: %s -> /predict disabled (emission mode requires it)",
                          self.classifier_error)
            else:
                log.warning("Classifier: %s -> using role C's distance-only HMM emissions",
                            self.classifier_error)


def _summarise_track(tid: str, base: Path) -> TripInfo:
    """Cheap, ground-truth-free summary of a trip from its .track file only."""
    track = pd.read_csv(base.with_suffix(".track"), sep=r"\s+", header=None, names=["lon", "lat", "t"])
    lon, lat, t = track.lon.to_numpy(), track.lat.to_numpy(), track.t.to_numpy()
    if len(track) == 0:
        raise ValueError("empty track")
    step = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:]) if len(track) > 1 else np.zeros(0)
    return TripInfo(
        trip_id=tid,
        n_points=len(track),
        duration_s=float(t[-1] - t[0]),
        length_km=float(step.sum() / 1000),
        bbox=((float(lat.min()), float(lon.min())), (float(lat.max()), float(lon.max()))),
        has_reference=base.with_suffix(".route").is_file(),
    )


def _load_classifier(model_path: Path, cfg_path: Path) -> Classifier:
    import joblib

    # joblib unpickles code: only ever point PATHSNAP_MODELS_DIR at the team's own exported model.
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise TypeError("saved model has no predict_proba")
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}

    # feature order: B's config first (brief: "feature column order"), then what sklearn recorded
    cols = cfg.get("feature_columns") or cfg.get("features") or cfg.get("feature_order")
    if not cols and hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    if not cols:
        raise ValueError("feature column order unknown (no classifier_config.json 'feature_columns')")

    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"model classes {classes} do not include the positive label 1")

    name = cfg.get("model") or cfg.get("model_name")
    if not name:
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        name = type(est).__name__
    return Classifier(model=model, feature_columns=list(cols), positive_index=classes.index(1), name=str(name))

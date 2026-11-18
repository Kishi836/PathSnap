"""API tests: endpoints, error handling, classifier integration and ground-truth isolation.

Run from the repository root:  python -m pytest tests -q

The classifier used in ``test_classifier_*`` is a small logistic regression fitted inside the
test on train candidates. It is a TEST FIXTURE ONLY (written to a temp folder) and exists to
exercise the loading/scoring path until role B exports models/classifier.joblib.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import evaluation, pipeline
from api.config import REPO_ROOT, Settings
from api.main import create_app

DATA = REPO_ROOT / "Student folder" / "Student folder"
SMALL_TRIP = "00000052"   # 262 GPS points: fast


def settings(**kw) -> Settings:
    base = Settings(data_root=DATA, split="student_test", models_dir=REPO_ROOT / "models",
                    emission="auto", require_classifier=False, cors_origins=("*",),
                    log_dir=REPO_ROOT / "logs")
    return dataclasses.replace(base, **kw)


def client(**kw) -> TestClient:
    return TestClient(create_app(settings(**kw)))


@pytest.fixture(scope="module")
def api():
    with client() as c:
        yield c


@pytest.fixture()
def tmp_data(tmp_path):
    """A one-trip partition copied from student_test, so tests can remove/break files."""
    dst = tmp_path / "data" / "demo" / SMALL_TRIP
    shutil.copytree(DATA / "student_test" / SMALL_TRIP, dst)
    return tmp_path / "data", dst


# ---- basic endpoints ----------------------------------------------------------------
def test_health_ok(api):
    h = api.get("/health").json()
    assert h["status"] == "ok" and h["pipeline_ready"] and h["hmm_config_loaded"]
    assert h["trips_available"] == 10 and h["split"] == "student_test"
    assert h["emission"] in ("classifier", "distance")


def test_trips_match_folders(api):
    d = api.get("/trips").json()
    on_disk = sorted(p.name for p in (DATA / "student_test").iterdir() if p.is_dir())
    assert [t["trip_id"] for t in d["trips"]] == on_disk and d["count"] == 10
    t = d["trips"][0]
    assert t["n_points"] > 0 and t["length_km"] > 0 and len(t["bbox"]) == 2 and t["has_reference"]


def test_track_preview(api):
    d = api.get(f"/trips/{SMALL_TRIP}/track").json()
    track = np.loadtxt(DATA / "student_test" / SMALL_TRIP / f"{SMALL_TRIP}.track")
    assert len(d["gps_points"]) == len(track)
    assert d["gps_points"][0] == pytest.approx([track[0, 1], track[0, 0]], abs=1e-6)  # [lat, lon]


def test_predict_structure(api):
    r = api.post("/predict", json={"trip_id": SMALL_TRIP})
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    d = r.json()
    n = d["metrics"]["n_points"]
    assert len(d["gps_points"]) == len(d["predicted_arcs"]) == n == 262
    assert d["predicted_route"] and all(len(line) >= 2 for line in d["predicted_route"])
    route = np.array([p for line in d["predicted_route"] for p in line])
    gps = np.array(d["gps_points"])
    lo, hi = gps.min(0) - 0.01, gps.max(0) + 0.01             # route lies over the GPS track (~1 km slack)
    assert np.all((route >= lo) & (route <= hi))
    m = d["metrics"]
    assert m["scored"] and 0 <= m["top1_accuracy"] <= 1 and m["correct_points"] <= m["scored_points"]
    assert set(d["timings_ms"]) == {"load", "candidates", "classifier", "decode", "geometry", "scoring", "total"}


def test_predict_matches_role_c_decoder(api):
    """The API must return exactly what role C's decoder produces for the same inputs."""
    from src import network, viterbi
    from src.build_candidates import trip_features
    from src.data import load_trip

    d = api.post("/predict", json={"trip_id": SMALL_TRIP}).json()
    if d["pipeline"]["emission"] != "distance":
        pytest.skip("reference run below uses distance emissions")
    trip = load_trip("student_test", SMALL_TRIP)
    trip.pop("route")
    feats, *_ = trip_features(trip)
    graph = network.RoadGraph(network.load_trip(SMALL_TRIP, "student_test", with_route=False))
    ref = viterbi.decode(feats, None, graph, viterbi.HMMParams.from_json(REPO_ROOT / "models" / "hmm_config.json"))
    assert d["predicted_arcs"] == ref.tolist()


# ---- error handling -----------------------------------------------------------------
def test_unknown_trip_404(api):
    r = api.post("/predict", json={"trip_id": "99999999"})
    assert r.status_code == 404 and r.json()["code"] == "TRIP_NOT_FOUND"
    assert api.get("/trips/99999999/track").status_code == 404


@pytest.mark.parametrize("body", [{}, {"trip_id": ""}, {"trip_id": "../../etc"}, {"trip_id": 5},
                                  {"trip_id": "x" * 200}])
def test_invalid_body_422(api, body):
    r = api.post("/predict", json=body)
    assert r.status_code == 422 and r.json()["code"] == "INVALID_REQUEST"


def test_missing_partition_degrades(tmp_path):
    with client(data_root=tmp_path) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and not h["pipeline_ready"]
        assert str(tmp_path) not in json.dumps(h)                  # no server paths leaked
        assert c.get("/trips").json()["count"] == 0


def test_missing_hmm_config_503(tmp_path):
    with client(models_dir=tmp_path) as c:
        assert not c.get("/health").json()["hmm_config_loaded"]
        r = c.post("/predict", json={"trip_id": SMALL_TRIP})
        assert r.status_code == 503 and r.json()["code"] == "PIPELINE_UNAVAILABLE"


def test_classifier_mode_without_model_503(tmp_path):
    shutil.copy(REPO_ROOT / "models" / "hmm_config.json", tmp_path)
    with client(models_dir=tmp_path, emission="classifier") as c:
        h = c.get("/health").json()
        assert h["emission"] == "classifier" and not h["classifier"]["loaded"] and h["status"] == "degraded"
        r = c.post("/predict", json={"trip_id": SMALL_TRIP})
        assert r.status_code == 503                       # never silently falls back to another model


def test_corrupt_trip_500_without_leaks(tmp_data):
    root, trip_dir = tmp_data
    (trip_dir / f"{SMALL_TRIP}.arcs").write_text("0 999999999\n")   # arc pointing at a missing node
    with client(data_root=root, split="demo") as c:
        r = c.post("/predict", json={"trip_id": SMALL_TRIP})
        body = r.json()
        assert r.status_code == 500 and body["code"] == "PIPELINE_ERROR" and body["stage"] == "load"
        assert str(root) not in json.dumps(body) and "Traceback" not in json.dumps(body)


# ---- ground-truth isolation ---------------------------------------------------------
def test_prediction_independent_of_reference(tmp_data):
    """Same prediction with or without the .route file; without it, nothing is scored."""
    root, trip_dir = tmp_data
    with client(data_root=root, split="demo") as c:
        with_ref = c.post("/predict", json={"trip_id": SMALL_TRIP}).json()
    (trip_dir / f"{SMALL_TRIP}.route").unlink()
    with client(data_root=root, split="demo") as c:
        assert c.get("/trips").json()["trips"][0]["has_reference"] is False
        no_ref = c.post("/predict", json={"trip_id": SMALL_TRIP}).json()
    assert with_ref["predicted_arcs"] == no_ref["predicted_arcs"]
    assert with_ref["predicted_route"] == no_ref["predicted_route"]
    assert with_ref["metrics"]["scored"] and not no_ref["metrics"]["scored"]
    assert no_ref["metrics"]["top1_accuracy"] is None


def test_reference_read_only_after_decoding(api, monkeypatch):
    events = []
    real_decode, real_ref = pipeline.viterbi.decode, evaluation.load_reference
    monkeypatch.setattr(pipeline.viterbi, "decode", lambda *a, **k: (events.append("decode"), real_decode(*a, **k))[1])
    monkeypatch.setattr(evaluation, "load_reference", lambda *a, **k: (events.append("reference"), real_ref(*a, **k))[1])
    assert api.post("/predict", json={"trip_id": SMALL_TRIP}).status_code == 200
    assert events == ["decode", "reference"]


def test_reference_never_in_response(api):
    d = api.post("/predict", json={"trip_id": SMALL_TRIP}).json()
    route = np.loadtxt(DATA / "student_test" / SMALL_TRIP / f"{SMALL_TRIP}.route", dtype=int).tolist()
    assert not {"route", "reference", "ref_arcs", "labels"} & set(d) and not {"route", "ref"} & set(d["metrics"])
    assert d["predicted_arcs"] != route          # response holds per-point predictions, not the reference list


def test_inference_code_never_opens_route_files():
    src = (REPO_ROOT / "api" / "pipeline.py").read_text()
    assert '".route"' not in src and "load_reference" not in src and "route_labels" not in src


# ---- classifier integration (test-only model) ----------------------------------------
@pytest.fixture(scope="module")
def test_model_dir(tmp_path_factory):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cands = pd.read_parquet(REPO_ROOT / "results" / "candidates_train.parquet")
    cols = [c for c in cands.columns if c.startswith("f_")]
    sample = cands[cands.trip_id.isin(sorted(cands.trip_id.unique())[:5])]
    model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=500))])
    model.fit(sample[cols], sample["label"])
    d = tmp_path_factory.mktemp("models")
    joblib.dump(model, d / "classifier.joblib")
    (d / "classifier_config.json").write_text(json.dumps({"model": "TEST-FIXTURE LogReg",
                                                          "feature_columns": cols}))
    shutil.copy(REPO_ROOT / "models" / "hmm_config.json", d)
    return d


def test_classifier_loaded_and_used(test_model_dir, monkeypatch):
    calls = []
    real = pipeline.viterbi.decode
    monkeypatch.setattr(pipeline.viterbi, "decode",
                        lambda c, e, *a, **k: (calls.append(e), real(c, e, *a, **k))[1])
    with client(models_dir=test_model_dir) as c:
        h = c.get("/health").json()
        assert h["emission"] == "classifier" and h["classifier"]["loaded"] and h["classifier"]["n_features"] == 15
        d = c.post("/predict", json={"trip_id": SMALL_TRIP}).json()
    assert d["pipeline"]["emission"] == "classifier" and d["pipeline"]["classifier"] == "TEST-FIXTURE LogReg"
    em = calls[0]
    assert em is not None and len(em) == d["pipeline"]["n_candidates"] and np.all(em <= 0)  # log-probabilities


def test_distance_mode_ignores_classifier(test_model_dir):
    with client(models_dir=test_model_dir, emission="distance") as c:
        assert c.get("/health").json()["emission"] == "distance"
        assert c.post("/predict", json={"trip_id": SMALL_TRIP}).json()["pipeline"]["emission"] == "distance"


def test_classifier_with_wrong_features_fails_cleanly(test_model_dir, tmp_path):
    for f in test_model_dir.iterdir():
        shutil.copy(f, tmp_path)
    cfg = json.loads((tmp_path / "classifier_config.json").read_text())
    cfg["feature_columns"] = cfg["feature_columns"][:-1] + ["f_does_not_exist"]
    (tmp_path / "classifier_config.json").write_text(json.dumps(cfg))
    with client(models_dir=tmp_path) as c:
        r = c.post("/predict", json={"trip_id": SMALL_TRIP})
        assert r.status_code == 500 and r.json()["stage"] == "classifier"

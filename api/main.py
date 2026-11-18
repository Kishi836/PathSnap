"""PathSnap API (laptop 1): runs the full map-matching pipeline for one trip per request.

Run from the repository root (either works):
    python -m api                                          # prints the LAN address to give laptop 2
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Settings come from environment variables, see ``api/config.py``.

Request flow for POST /predict
    1. validate the trip id against the served partition (also blocks path tricks)
    2. pipeline.run: track + network -> A's candidates/features -> B's classifier -> C's Viterbi
    3. evaluation.score: only now is the .route file read, to compute aggregate metrics
    4. respond with GPS points, predicted route geometry and metrics (never the reference)
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import evaluation, logs, pipeline
from .config import Settings
from .resources import Resources
from .schemas import (ClassifierStatus, ErrorResponse, HealthResponse, PredictRequest, PredictResponse,
                      TrackResponse, TripsResponse, TripSummary)

VERSION = "1.0.0"
QUIET_PATHS = {"/health"}          # the UI polls it; keep the evaluator's terminal readable


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str, stage: str | None = None):
        self.status, self.code, self.detail, self.stage = status, code, detail, stage


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    log = logs.setup(settings.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("PathSnap API %s starting (partition %r)", VERSION, settings.split)
        res = Resources.load(settings)
        app.state.res = res
        app.state.started = time.time()
        if res.blocking_problems:
            log.error("NOT READY: %s", "; ".join(res.blocking_problems))
        else:
            log.info("Ready: %d trips, emission=%s", len(res.trips), res.emission_mode)
        yield
        log.info("PathSnap API stopped")

    app = FastAPI(title="PathSnap API", version=VERSION, lifespan=lifespan,
                  description="GPS map-matching: candidates -> classifier -> HMM/Viterbi.")
    # The UI is served from laptop 2 (another origin), so browsers need CORS.
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["*"],
                       expose_headers=["X-Request-ID"])

    # ---- request logging + error shaping ---------------------------------------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        request.state.request_id = rid
        request.state.log_extra = ""
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("[%s] unhandled error on %s %s", rid, request.method, request.url.path)
            response = _error(500, "INTERNAL_ERROR", "Unexpected server error", rid)
        response.headers["X-Request-ID"] = rid
        ms = (time.perf_counter() - t0) * 1000
        if request.method != "OPTIONS" and request.url.path not in QUIET_PATHS:
            client = request.client.host if request.client else "?"
            level = logging.INFO if response.status_code < 500 else logging.ERROR
            log.log(level, "%-15s %-4s %-24s %d %6.0f ms  %s", client, request.method, request.url.path,
                    response.status_code, ms, request.state.log_extra)
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.detail, request.state.request_id, exc.stage)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return _error(exc.status_code, code, str(exc.detail), getattr(request.state, "request_id", "-"))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", ()) if x != "body")
        request.state.log_extra = f"invalid request ({where or 'body'})"
        return _error(422, "INVALID_REQUEST", f"Invalid request: {where or 'body'} {first.get('msg', '')}".strip(),
                      request.state.request_id)

    # ---- endpoints -------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "PathSnap API", "version": VERSION, "docs": "/docs",
                "endpoints": ["/health", "/trips", "/trips/{trip_id}/track", "/predict"]}

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        res: Resources = request.app.state.res
        problems = res.blocking_problems
        return HealthResponse(
            status="ok" if not problems else "degraded",
            pipeline_ready=not problems,
            problems=problems,
            split=res.settings.split,
            trips_available=len(res.trips),
            emission=res.emission_mode,
            classifier=ClassifierStatus(
                loaded=res.classifier is not None,
                name=res.classifier.name if res.classifier else None,
                n_features=len(res.classifier.feature_columns) if res.classifier else None,
                problem=res.classifier_error,
            ),
            hmm_config_loaded=res.hmm_params is not None,
            uptime_s=round(time.time() - request.app.state.started, 1),
            version=VERSION,
        )

    @app.get("/trips", response_model=TripsResponse)
    def trips(request: Request):
        res: Resources = request.app.state.res
        items = [TripSummary(trip_id=t.trip_id, n_points=t.n_points, duration_s=t.duration_s,
                             length_km=round(t.length_km, 3), bbox=[list(t.bbox[0]), list(t.bbox[1])],
                             has_reference=t.has_reference)
                 for t in res.trips.values()]
        request.state.log_extra = f"{len(items)} trips"
        return TripsResponse(split=res.settings.split, count=len(items), trips=items)

    @app.get("/trips/{trip_id}/track", response_model=TrackResponse,
             responses={404: {"model": ErrorResponse}})
    def track(trip_id: str, request: Request):
        """Raw GPS fixes of a trip (model input only) so the UI can preview it before predicting."""
        res: Resources = request.app.state.res
        _require_trip(res, trip_id)
        request.state.log_extra = f"trip={trip_id}"
        base = res.trip_dir(trip_id) / trip_id
        try:
            pts = np.loadtxt(base.with_suffix(".track"), ndmin=2)[:, [1, 0]]
        except Exception:
            log.exception("[%s] track read failed for %s", request.state.request_id, trip_id)
            raise ApiError(500, "DATA_ERROR", "Trip track could not be read")
        return TrackResponse(trip_id=trip_id, gps_points=np.round(pts, 7).tolist())

    @app.post("/predict", response_model=PredictResponse,
              responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
                         500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def predict(req: PredictRequest, request: Request):
        res: Resources = request.app.state.res
        rid = request.state.request_id
        request.state.log_extra = f"trip={req.trip_id}"
        _require_trip(res, req.trip_id)
        if res.blocking_problems:
            request.state.log_extra += "  REFUSED (pipeline not ready)"
            raise ApiError(503, "PIPELINE_UNAVAILABLE",
                           "Prediction pipeline not ready: " + "; ".join(res.blocking_problems))

        t0 = time.perf_counter()
        try:
            pred = pipeline.run(res, req.trip_id)            # no ground truth in here
        except pipeline.PipelineError as exc:
            log.error("[%s] %s stage failed for trip %s: %r", rid, exc.stage, req.trip_id, exc.__cause__,
                      exc_info=exc.__cause__)
            request.state.log_extra += f"  FAILED at {exc.stage}"
            raise ApiError(500, "PIPELINE_ERROR", f"Prediction failed during {exc.stage}: {exc.message}",
                           stage=exc.stage)
        infer_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        try:
            scores = evaluation.score(res, pred)              # reference read only now, server-side
        except Exception as exc:
            log.error("[%s] scoring failed for trip %s: %r", rid, req.trip_id, exc, exc_info=exc)
            scores = {"scored": False}
        pred.timings_ms["scoring"] = round((time.perf_counter() - t1) * 1000, 1)
        pred.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)

        stats = dict(pred.stats)
        metrics = {**scores, "route_continuity": stats.pop("route_continuity"), "n_points": len(pred.pred_arcs)}
        request.state.log_extra += "  OK  " + _summary(metrics, stats, infer_ms)

        return PredictResponse(
            trip_id=req.trip_id,
            split=res.settings.split,
            gps_points=np.round(pred.track[["lat", "lon"]].to_numpy(), 7).tolist(),
            predicted_route=pred.polylines,
            predicted_arcs=pred.pred_arcs.tolist(),
            metrics=metrics,
            pipeline=stats,
            timings_ms=pred.timings_ms,
            inference_ms=round(infer_ms, 1),
            request_id=rid,
        )

    return app


def _require_trip(res: Resources, trip_id: str):
    # membership in the scanned catalogue is the only way to reach a file -> no path traversal
    if trip_id not in res.trips:
        raise ApiError(404, "TRIP_NOT_FOUND", f"Unknown trip {trip_id!r} in partition {res.settings.split!r}")


def _summary(metrics: dict, stats: dict, infer_ms: float) -> str:
    if metrics.get("scored"):
        acc = (f"acc {logs.pct(metrics['top1_accuracy_moving'])} "
               f"({metrics['correct_points_moving']}/{metrics['scored_points_moving']} moving, parked excluded)")
    else:
        acc = "acc n/a (no reference)"
    return (f"{acc}  continuity {logs.pct(metrics.get('route_continuity'))}  points {metrics['n_points']}  "
            f"emission={stats['emission']}  infer {infer_ms / 1000:.2f}s")


def _error(status: int, code: str, detail: str, rid: str, stage: str | None = None) -> JSONResponse:
    body = ErrorResponse(detail=detail, code=code, request_id=rid, stage=stage)
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True),
                        headers={"X-Request-ID": rid})


app = create_app()

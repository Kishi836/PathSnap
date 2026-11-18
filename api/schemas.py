"""Request / response models (also drive the OpenAPI docs at /docs)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

LatLon = list[float]   # [lat, lon]


class ErrorResponse(BaseModel):
    detail: str
    code: str
    request_id: str
    stage: Optional[str] = None


class ClassifierStatus(BaseModel):
    loaded: bool
    name: Optional[str] = None
    n_features: Optional[int] = None
    problem: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    pipeline_ready: bool
    problems: list[str]
    split: str
    trips_available: int
    emission: Literal["classifier", "distance"]
    classifier: ClassifierStatus
    hmm_config_loaded: bool
    uptime_s: float
    version: str


class TripSummary(BaseModel):
    trip_id: str
    n_points: int
    duration_s: float
    length_km: float = Field(description="GPS track length (sum of fix-to-fix distances)")
    bbox: list[LatLon] = Field(description="[[min_lat, min_lon], [max_lat, max_lon]]")
    has_reference: bool = Field(description="server can score this trip (reference is never sent)")


class TripsResponse(BaseModel):
    split: str
    count: int
    trips: list[TripSummary]


class TrackResponse(BaseModel):
    trip_id: str
    gps_points: list[LatLon]


class PredictRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
                         examples=["00000040"])


class Metrics(BaseModel):
    scored: bool = Field(description="False when the partition has no server-side reference route")
    top1_accuracy: Optional[float] = Field(None, description="share of GPS points on the correct arc")
    correct_points: Optional[int] = None
    scored_points: Optional[int] = None
    top1_accuracy_moving: Optional[float] = Field(None, description="same, excluding parked points (headline metric)")
    correct_points_moving: Optional[int] = None
    scored_points_moving: Optional[int] = None
    route_overlap: Optional[float] = Field(None, description="Jaccard overlap of predicted vs reference arc sets")
    route_continuity: Optional[float] = Field(None, description="share of consecutive predictions that connect")
    n_points: int


class PipelineInfo(BaseModel):
    emission: Literal["classifier", "distance"]
    classifier: Optional[str]
    n_candidates: int
    points_without_candidates: int
    segments: int
    forced_breaks: int
    n_route_arcs: int
    route_length_km: float
    route_breaks: int


class Timings(BaseModel):
    load: float
    candidates: float
    classifier: float
    decode: float
    geometry: float
    scoring: float
    total: float


class PredictResponse(BaseModel):
    trip_id: str
    split: str
    gps_points: list[LatLon]
    predicted_route: list[list[LatLon]] = Field(description="connected pieces of the predicted route")
    predicted_arcs: list[int] = Field(description="predicted arc id per GPS point (-1 = none)")
    metrics: Metrics
    pipeline: PipelineInfo
    timings_ms: Timings
    inference_ms: float
    request_id: str

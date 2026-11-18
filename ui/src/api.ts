// Typed client for the PathSnap API (api/schemas.py is the source of truth for these shapes).

export type LatLon = [number, number];

export interface Health {
  status: "ok" | "degraded";
  pipeline_ready: boolean;
  problems: string[];
  split: string;
  trips_available: number;
  emission: "classifier" | "distance";
  classifier: { loaded: boolean; name: string | null; n_features: number | null; problem: string | null };
  hmm_config_loaded: boolean;
  uptime_s: number;
  version: string;
}

export interface TripSummary {
  trip_id: string;
  n_points: number;
  duration_s: number;
  length_km: number;
  bbox: [LatLon, LatLon];
  has_reference: boolean;
}

export interface TripsResponse {
  split: string;
  count: number;
  trips: TripSummary[];
}

export interface TrackResponse {
  trip_id: string;
  gps_points: LatLon[];
}

export interface Metrics {
  scored: boolean;
  top1_accuracy: number | null;
  correct_points: number | null;
  scored_points: number | null;
  route_overlap: number | null;
  route_continuity: number | null;
  n_points: number;
}

export interface PipelineInfo {
  emission: "classifier" | "distance";
  classifier: string | null;
  n_candidates: number;
  points_without_candidates: number;
  segments: number;
  forced_breaks: number;
  n_route_arcs: number;
  route_length_km: number;
  route_breaks: number;
}

export type Stage = "load" | "candidates" | "classifier" | "decode" | "geometry" | "scoring";
export type Timings = Record<Stage | "total", number>;

export interface Prediction {
  trip_id: string;
  split: string;
  gps_points: LatLon[];
  predicted_route: LatLon[][];
  predicted_arcs: number[];
  metrics: Metrics;
  pipeline: PipelineInfo;
  timings_ms: Timings;
  inference_ms: number;
  request_id: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
    readonly code: string,
    readonly stage?: string,
  ) {
    super(message);
  }
}

const STORAGE_KEY = "pathsnap.apiBase";

/** Normalise user input like "192.168.1.5:8000/" into "http://192.168.1.5:8000". */
export function normaliseBase(raw: string): string {
  let s = raw.trim().replace(/\/+$/, "");
  if (!s) return s;
  if (!/^https?:\/\//i.test(s)) s = `http://${s}`;
  return s;
}

/** Saved choice in this browser > VITE_API_BASE_URL at build time > same host as the UI on :8000. */
export function initialBase(
  envBase: string | undefined = import.meta.env.VITE_API_BASE_URL,
  loc: Pick<Location, "protocol" | "hostname"> = window.location,
): string {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) return saved;
  } catch {
    /* storage blocked: fall through */
  }
  if (envBase && envBase.trim()) return normaliseBase(envBase);
  const host = loc.hostname || "localhost";
  const proto = loc.protocol === "https:" ? "https:" : "http:";
  return `${proto}//${host}:8000`;
}

export function rememberBase(base: string) {
  try {
    window.localStorage.setItem(STORAGE_KEY, base);
  } catch {
    /* non-essential */
  }
}

async function request<T>(base: string, path: string, init?: RequestInit, timeoutMs = 120_000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  const outer = init?.signal;
  outer?.addEventListener("abort", () => ctrl.abort());
  let res: Response;
  try {
    res = await fetch(base + path, { ...init, signal: ctrl.signal });
  } catch {
    if (outer?.aborted) throw new ApiError("Request cancelled", null, "CANCELLED");
    if (ctrl.signal.aborted) throw new ApiError(`No answer from ${base} within ${timeoutMs / 1000} s`, null, "TIMEOUT");
    throw new ApiError(`Can't reach the API at ${base}`, null, "UNREACHABLE");
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let body: { detail?: unknown; code?: string; stage?: string } = {};
    try {
      body = await res.json();
    } catch {
      /* non-JSON error page */
    }
    const detail = typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`;
    throw new ApiError(detail, res.status, body.code ?? "HTTP_ERROR", body.stage);
  }
  return (await res.json()) as T;
}

export const api = {
  health: (base: string, signal?: AbortSignal) => request<Health>(base, "/health", { signal }, 5_000),
  trips: (base: string) => request<TripsResponse>(base, "/trips", undefined, 15_000),
  track: (base: string, tripId: string, signal?: AbortSignal) =>
    request<TrackResponse>(base, `/trips/${encodeURIComponent(tripId)}/track`, { signal }, 30_000),
  predict: (base: string, tripId: string, signal?: AbortSignal) =>
    request<Prediction>(base, "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
      signal,
    }),
};

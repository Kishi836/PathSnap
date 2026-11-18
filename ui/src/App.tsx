import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, initialBase, normaliseBase, rememberBase } from "./api";
import type { LatLon, Prediction, TripsResponse } from "./api";
import { MapView } from "./components/MapView";
import { Connection, Result, RunButton, TripList } from "./components/Rail";
import type { Conn } from "./components/Rail";

const HEALTH_POLL_MS = 10_000;

interface Outcome {
  prediction: Prediction;
  roundTripMs: number;
}

export default function App() {
  const [base, setBase] = useState(initialBase);
  const [conn, setConn] = useState<Conn>({ state: "checking" });
  const [trips, setTrips] = useState<TripsResponse | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [track, setTrack] = useState<{ id: string; points: LatLon[] } | null>(null);
  const [trackError, setTrackError] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, Outcome>>({});
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const trackAbort = useRef<AbortController | null>(null);

  // ---- health polling + trip catalogue (per API address) --------------------------
  useEffect(() => {
    let alive = true;
    let loadedTrips = false;
    const ctrl = new AbortController();
    setConn({ state: "checking" });
    setTrips(null);

    const check = async () => {
      try {
        const health = await api.health(base, ctrl.signal);
        if (!alive) return;
        setConn({ state: "online", health });
        if (!loadedTrips) {
          const t = await api.trips(base);
          if (!alive) return;
          loadedTrips = true;
          setTrips(t);
        }
      } catch (e) {
        if (!alive || (e instanceof ApiError && e.code === "CANCELLED")) return;
        setConn({ state: "offline", message: e instanceof ApiError ? e.message : "Can't reach the API" });
      }
    };
    check();
    const id = window.setInterval(check, HEALTH_POLL_MS);
    return () => {
      alive = false;
      ctrl.abort();
      window.clearInterval(id);
    };
  }, [base]);

  const connect = (raw: string) => {
    const b = normaliseBase(raw);
    if (!b) return;
    rememberBase(b);
    setResults({});
    setSelected(null);
    setTrack(null);
    setBase(b);
  };

  // ---- trip selection: preview its raw GPS track ----------------------------------
  const select = useCallback(
    async (id: string) => {
      if (running) return;
      setSelected(id);
      setRunError(null);
      setTrackError(null);
      trackAbort.current?.abort();
      const ctrl = new AbortController();
      trackAbort.current = ctrl;
      try {
        const t = await api.track(base, id, ctrl.signal);
        setTrack({ id, points: t.gps_points });
      } catch (e) {
        if (e instanceof ApiError && e.code === "CANCELLED") return;
        setTrack(null);
        setTrackError(e instanceof ApiError ? e.message : "Couldn't load the trip");
      }
    },
    [base, running],
  );

  // ---- prediction ------------------------------------------------------------------
  const run = async () => {
    if (!selected || running) return;
    const id = selected;
    setRunning(true);
    setRunError(null);
    const t0 = performance.now();
    try {
      const p = await api.predict(base, id);
      setResults((r) => ({ ...r, [id]: { prediction: p, roundTripMs: performance.now() - t0 } }));
    } catch (e) {
      setRunError(describe(e));
    } finally {
      setRunning(false);
    }
  };

  const outcome = selected ? results[selected] : undefined;
  const gps = outcome?.prediction.gps_points ?? (track && track.id === selected ? track.points : null);
  const ready = conn.state === "online" && conn.health.pipeline_ready;

  return (
    <div className="flex h-full flex-col lg:grid lg:grid-cols-[minmax(340px,380px)_1fr]">
      <main className="order-2 flex min-h-0 flex-1 flex-col overflow-y-auto border-rule bg-panel lg:order-1 lg:border-r">
        <header className="px-5 pb-3 pt-5">
          <h1 className="flex items-center gap-2.5 text-[22px] font-bold" style={{ fontStretch: "115%" }}>
            <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
              <path d="M5 25 C11 23 9 13 16 12 S23 7 27 5" fill="none" stroke="#f2b33d" strokeWidth="3.4" strokeLinecap="round" />
              <circle cx="5" cy="25" r="3.2" fill="#62d2e8" />
              <circle cx="16" cy="12" r="2.2" fill="#62d2e8" />
              <circle cx="27" cy="5" r="2.2" fill="#62d2e8" />
            </svg>
            PathSnap
          </h1>
          <p className="mt-1 max-w-[34ch] text-[13.5px] leading-snug text-muted">
            Recovers the road a car drove from its noisy GPS fixes.
          </p>
        </header>

        <Connection base={base} conn={conn} onConnect={connect} />

        <TripList
          split={trips?.split ?? null}
          trips={trips?.trips ?? null}
          selected={selected}
          locked={running}
          resultFor={(id) => results[id]?.prediction}
          onSelect={select}
        />

        {trips && (
          <div className="sticky bottom-0 z-10 border-y border-rule bg-panel px-5 py-4">
            {/* sticky on short screens: the action must stay visible while scrolling the trip list */}
            <RunButton disabled={!selected || !ready} running={running} hasResult={!!outcome} onRun={run} />
            {!selected && trips.trips.length > 0 && (
              <p className="mt-2 text-[13px] text-muted">Pick a trip to match its route.</p>
            )}
            {runError && (
              <p role="alert" className="mt-2.5 text-[13.5px] leading-snug text-fail">{runError}</p>
            )}
            {trackError && !runError && (
              <p role="alert" className="mt-2.5 text-[13.5px] leading-snug text-fail">{trackError}</p>
            )}
          </div>
        )}

        {outcome && <Result p={outcome.prediction} roundTripMs={outcome.roundTripMs} />}
      </main>

      <section aria-label="Map" className="relative order-1 h-[52vh] shrink-0 lg:order-2 lg:h-full">
        <MapView
          gps={gps}
          route={outcome?.prediction.predicted_route ?? null}
          routeKey={outcome ? outcome.prediction.request_id : null}
          fitKey={selected && gps ? `${selected}:${outcome?.prediction.request_id ?? "track"}` : null}
        />
        {!gps && (
          <div className="pointer-events-none absolute inset-0 z-[1000] grid place-items-center p-6">
            <p className="max-w-xs rounded-md border border-rule bg-panel/95 px-4 py-3 text-center text-[14px] leading-snug text-muted">
              {conn.state === "offline"
                ? "Connect to the PathSnap API to load trips."
                : selected
                  ? "Loading the GPS track…"
                  : "Pick a trip to see its raw GPS track here."}
            </p>
          </div>
        )}
      </section>
    </div>
  );
}

function describe(e: unknown): string {
  if (!(e instanceof ApiError)) return "Prediction failed for an unknown reason. Check the API terminal on laptop 1.";
  switch (e.code) {
    case "UNREACHABLE":
      return `${e.message}. Is the API still running and on the same network?`;
    case "TIMEOUT":
      return `${e.message}. The API may be busy; try again.`;
    case "PIPELINE_UNAVAILABLE":
      return e.message;
    case "PIPELINE_ERROR":
      return `${e.message}. Details are in the API log on laptop 1.`;
    case "TRIP_NOT_FOUND":
      return `${e.message}. Reload the page to refresh the trip list.`;
    default:
      return e.message;
  }
}

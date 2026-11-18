import { useEffect, useState } from "react";
import type { Health, Prediction, Stage, TripSummary } from "../api";
import { duration, int, ms, pct } from "../format";

/* ---------------------------------------------------------------- connection */

export type Conn =
  | { state: "checking" }
  | { state: "online"; health: Health }
  | { state: "offline"; message: string };

export function Connection({
  base,
  conn,
  onConnect,
}: {
  base: string;
  conn: Conn;
  onConnect: (base: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(base);
  useEffect(() => setDraft(base), [base]);
  const open = editing || conn.state === "offline";

  const dot =
    conn.state === "online"
      ? conn.health.pipeline_ready
        ? "bg-ok"
        : "bg-route"
      : conn.state === "offline"
        ? "bg-fail"
        : "bg-faint";
  const label =
    conn.state === "online"
      ? conn.health.pipeline_ready
        ? "API connected"
        : "API connected, pipeline not ready"
      : conn.state === "offline"
        ? "API not reachable"
        : "Checking API";

  return (
    <section aria-label="API connection" className="border-b border-rule px-5 py-3.5">
      <div className="flex items-center gap-2.5 text-[13px]">
        <span className={`h-2 w-2 shrink-0 rounded-full ${dot}`} aria-hidden />
        <span className="text-chart" role="status">{label}</span>
        <span className="min-w-0 truncate text-muted" title={base}>{base.replace(/^https?:\/\//, "")}</span>
        {!open && (
          <button onClick={() => setEditing(true)} className="ml-auto shrink-0 text-muted underline-offset-2 hover:text-chart hover:underline">
            Change
          </button>
        )}
      </div>

      {conn.state === "offline" && (
        <p className="mt-2 text-[13px] leading-snug text-fail">
          {conn.message}. Start the API on laptop 1 with <code className="text-chart">python -m api</code>, join the
          same network, and enter the address it prints.
        </p>
      )}
      {conn.state === "online" && conn.health.problems.length > 0 && (
        <ul className="mt-2 list-disc pl-4 text-[13px] leading-snug text-route">
          {conn.health.problems.map((p) => <li key={p}>{p}</li>)}
        </ul>
      )}

      {open && (
        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            setEditing(false);
            onConnect(draft);
          }}
        >
          <label className="sr-only" htmlFor="api-base">API address</label>
          <input
            id="api-base"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            spellCheck={false}
            autoComplete="off"
            placeholder="http://192.168.1.50:8000"
            className="min-w-0 flex-1 rounded-md border border-rule bg-ink px-3 py-2 text-[14px] text-chart placeholder:text-faint focus:border-muted focus:outline-none"
          />
          <button type="submit" className="rounded-md border border-rule bg-raised px-3.5 text-[14px] font-medium text-chart hover:border-muted">
            Connect
          </button>
        </form>
      )}
    </section>
  );
}

/* ---------------------------------------------------------------- trips */

export function TripList({
  split,
  trips,
  selected,
  locked,
  resultFor,
  onSelect,
}: {
  split: string | null;
  trips: TripSummary[] | null;
  selected: string | null;
  locked: boolean;
  resultFor: (id: string) => Prediction | undefined;
  onSelect: (id: string) => void;
}) {
  if (!trips) return null;
  return (
    <section aria-labelledby="trips-h" className="border-b border-rule px-5 pb-4 pt-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 id="trips-h" className="text-[15px] font-semibold">Trips</h2>
        <span className="text-[13px] text-muted">{trips.length} in {split}</span>
      </div>
      {trips.length === 0 ? (
        <p className="text-[14px] text-muted">The API found no trips in this partition. Check PATHSNAP_DATA_ROOT and PATHSNAP_SPLIT on laptop 1.</p>
      ) : (
        <ul className="-mx-2" role="listbox" aria-label="Trips" aria-activedescendant={selected ? `trip-${selected}` : undefined}>
          {trips.map((t) => {
            const active = t.trip_id === selected;
            const done = resultFor(t.trip_id);
            return (
              <li key={t.trip_id} id={`trip-${t.trip_id}`} role="option" aria-selected={active}>
                <button
                  disabled={locked && !active}
                  onClick={() => onSelect(t.trip_id)}
                  className={`group grid w-full grid-cols-[1fr_auto] items-baseline gap-x-3 rounded-md border-l-[3px] px-2 py-1.5 text-left transition-colors disabled:opacity-40 ${
                    active ? "border-route bg-raised" : "border-transparent hover:bg-raised/60"
                  }`}
                >
                  <span className={`text-[15px] ${active ? "font-semibold text-chart" : "text-chart/90"}`}>{t.trip_id}</span>
                  <span className={`text-[13px] ${done?.metrics.scored ? "text-route" : "text-faint"}`}>
                    {done ? (done.metrics.scored ? pct(done.metrics.top1_accuracy) : "matched") : ""}
                  </span>
                  <span className="col-span-2 text-[12.5px] text-muted">
                    {t.length_km.toFixed(1)} km, {int(t.n_points)} fixes, {duration(t.duration_s)}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

/* ---------------------------------------------------------------- run */

export function RunButton({
  disabled,
  running,
  hasResult,
  onRun,
}: {
  disabled: boolean;
  running: boolean;
  hasResult: boolean;
  onRun: () => void;
}) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) return;
    const t0 = performance.now();
    setElapsed(0);
    const id = window.setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    return () => window.clearInterval(id);
  }, [running]);

  return (
    <button
      onClick={onRun}
      disabled={disabled || running}
      aria-busy={running}
      className="flex w-full items-center justify-center gap-2.5 rounded-md bg-route px-4 py-3 text-[15px] font-semibold text-ink transition-[filter] hover:brightness-110 disabled:cursor-not-allowed disabled:bg-raised disabled:text-faint"
    >
      {running ? (
        <>
          <svg className="spin" width="16" height="16" viewBox="0 0 16 16" aria-hidden>
            <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeOpacity=".3" strokeWidth="2.2" />
            <path d="M8 2a6 6 0 0 1 6 6" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
          </svg>
          Matching route… {elapsed.toFixed(1)} s
        </>
      ) : hasResult ? (
        "Match route again"
      ) : (
        "Match route"
      )}
    </button>
  );
}

/* ---------------------------------------------------------------- result */

const STAGES: { key: Stage; label: string; color: string }[] = [
  { key: "load", label: "Load", color: "#5d7686" },
  { key: "candidates", label: "Candidates", color: "#62d2e8" },
  { key: "classifier", label: "Classifier", color: "#a98bf0" },
  { key: "decode", label: "Viterbi", color: "#f2b33d" },
  { key: "geometry", label: "Geometry", color: "#8aa2b0" },
  { key: "scoring", label: "Scoring", color: "#7fd49a" },
];

export function Result({ p, roundTripMs }: { p: Prediction; roundTripMs: number | null }) {
  const m = p.metrics;
  const emission =
    p.pipeline.emission === "classifier"
      ? `Classifier probabilities (${p.pipeline.classifier})`
      : "Distance baseline (no exported classifier yet)";
  const total = STAGES.reduce((s, st) => s + p.timings_ms[st.key], 0) || 1;

  return (
    <section aria-label={`Result for trip ${p.trip_id}`} aria-live="polite" className="px-5 py-5">
      {m.scored ? (
        <>
          <p className="text-[13px] text-muted">GPS fixes matched to the correct road</p>
          <p className="figure-xl mt-1.5 text-[64px] text-route">{pct(m.top1_accuracy)}</p>
          <p className="mt-2 text-[14px] text-chart">
            {int(m.correct_points)} of {int(m.scored_points)} fixes on the reference arc
          </p>
        </>
      ) : (
        <>
          <p className="figure-xl text-[40px] text-chart">Matched</p>
          <p className="mt-2 text-[14px] leading-snug text-muted">
            The API has no reference route for this trip, so accuracy isn&apos;t computed.
          </p>
        </>
      )}

      <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-rule pt-4 text-[14px]">
        {m.scored && <Stat label="Route overlap" value={pct(m.route_overlap)} hint="Shared arcs between predicted and reference routes (Jaccard)" />}
        <Stat label="Route continuity" value={pct(m.route_continuity)} hint="Consecutive predicted arcs that connect on the road graph" />
        <Stat label="Predicted route" value={`${p.pipeline.route_length_km.toFixed(1)} km`} hint={`${int(p.pipeline.n_route_arcs)} distinct road arcs`} />
        <Stat label="Candidate pairs" value={int(p.pipeline.n_candidates)} hint={`${int(p.pipeline.points_without_candidates)} fixes had no road within 50 m`} />
      </dl>

      <div className="mt-5 border-t border-rule pt-4">
        <div className="flex items-baseline justify-between text-[14px]">
          <span className="font-semibold">Server time</span>
          <span className="text-chart">{ms(p.timings_ms.total)}</span>
        </div>
        <div className="mt-2 flex h-2.5 w-full overflow-hidden rounded-sm bg-ink" role="img"
          aria-label={STAGES.map((s) => `${s.label} ${ms(p.timings_ms[s.key])}`).join(", ")}>
          {STAGES.map((s) => (
            <span key={s.key} style={{ width: `${(100 * p.timings_ms[s.key]) / total}%`, background: s.color }} />
          ))}
        </div>
        <ul className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-1 text-[12.5px] text-muted">
          {STAGES.map((s) => (
            <li key={s.key} className="flex items-center gap-1.5">
              <span className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: s.color }} aria-hidden />
              <span className="truncate">{s.label}</span>
              <span className="ml-auto text-chart/80">{ms(p.timings_ms[s.key])}</span>
            </li>
          ))}
        </ul>
        {roundTripMs != null && (
          <p className="mt-2 text-[12.5px] text-muted">Round trip including network: {ms(roundTripMs)}</p>
        )}
      </div>

      <p className="mt-5 border-t border-rule pt-4 text-[12.5px] leading-relaxed text-muted">
        Emissions: <span className="text-chart">{emission}</span>
        <br />
        Request {p.request_id}, partition {p.split}
      </p>
    </section>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div title={hint}>
      <dt className="text-[12.5px] text-muted">{label}</dt>
      <dd className="text-[17px] font-semibold">{value}</dd>
    </div>
  );
}

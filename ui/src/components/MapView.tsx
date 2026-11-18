import L from "leaflet";
import { useEffect, useMemo, useRef, useState } from "react";
import { CircleMarker, MapContainer, Marker, Pane, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { LatLon } from "../api";

// Esri tiles: they load from pages opened anywhere (OSM refuses some origins) and the team found
// other providers blocked on the campus network.
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";
const BASEMAPS = {
  chart: { label: "Chart", url: `${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, maxNativeZoom: 16, labels: `${ESRI}/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}` },
  streets: { label: "Streets", url: `${ESRI}/World_Street_Map/MapServer/tile/{z}/{y}/{x}`, maxNativeZoom: 19 },
  imagery: { label: "Satellite", url: `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`, maxNativeZoom: 19 },
} as const;
type BasemapKey = keyof typeof BASEMAPS;
const ATTRIBUTION = 'Tiles &copy; <a href="https://www.esri.com">Esri</a>';

// Renderers live in their own panes (created by <Pane> below) so the stacking order is fixed.
const svg = L.svg({ padding: 0.5, pane: "route" });
const canvas = L.canvas({ padding: 0.5, pane: "gps" });

const endpointIcon = (kind: "start" | "end") =>
  L.divIcon({
    className: "",
    iconSize: [18, 18],
    iconAnchor: [9, 9],
    html:
      kind === "start"
        ? `<div style="width:18px;height:18px;border-radius:50%;background:#0f1c26;border:3px solid #62d2e8"></div>`
        : `<div style="width:16px;height:16px;margin:1px;background:#f2b33d;border:3px solid #0f1c26;outline:1.5px solid #f2b33d"></div>`,
  });
const START_ICON = endpointIcon("start");
const END_ICON = endpointIcon("end");

interface Props {
  gps: LatLon[] | null;
  route: LatLon[][] | null;
  routeKey: string | null; // changes when a new prediction arrives (re-runs the draw animation)
  fitKey: string | null; // changes when the map should re-frame
}

export function MapView({ gps, route, routeKey, fitKey }: Props) {
  const [basemap, setBasemap] = useState<BasemapKey>("chart");
  const [show, setShow] = useState({ gps: true, route: true });
  const bm = BASEMAPS[basemap];

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={[50, 20]}
        zoom={4}
        zoomControl={false}
        className="h-full w-full"
        attributionControl
        aria-label="Map of the GPS track and predicted route"
      >
        <TileLayer key={basemap} url={bm.url} maxNativeZoom={bm.maxNativeZoom} maxZoom={20} attribution={ATTRIBUTION} />
        {"labels" in bm && <TileLayer key={`${basemap}-labels`} url={bm.labels} maxNativeZoom={16} maxZoom={20} zIndex={5} />}
        <ZoomControl />
        <FitBounds gps={gps} route={route} fitKey={fitKey} />

        {/* Route above the fixes: cyan stays visible exactly where GPS drifted off the matched road. */}
        <Pane name="route" style={{ zIndex: 420 }}>
          {show.route && route && route.length > 0 && <AnimatedRoute key={routeKey ?? ""} route={route} />}
        </Pane>
        <Pane name="gps" style={{ zIndex: 410 }}>
          {show.gps && gps?.map((p, i) => (
            <CircleMarker
              key={i}
              center={p}
              radius={2.1}
              renderer={canvas}
              pathOptions={{ stroke: false, fillColor: "#62d2e8", fillOpacity: 0.85 }}
            >
              <Tooltip className="fix-tip" direction="top" offset={[0, -4]}>
                GPS fix {i + 1} of {gps.length}
              </Tooltip>
            </CircleMarker>
          ))}
        </Pane>
        {gps && gps.length > 0 && (
          <>
            <Marker position={gps[0]} icon={START_ICON} zIndexOffset={1000}>
              <Tooltip className="fix-tip" direction="right" offset={[10, 0]}>Start</Tooltip>
            </Marker>
            <Marker position={gps[gps.length - 1]} icon={END_ICON} zIndexOffset={1000}>
              <Tooltip className="fix-tip" direction="right" offset={[10, 0]}>End</Tooltip>
            </Marker>
          </>
        )}
      </MapContainer>

      <div className="pointer-events-none absolute inset-x-3 bottom-7 z-[1000] flex flex-wrap items-end justify-between gap-2 sm:inset-x-4">
        <Legend
          hasRoute={!!route?.length}
          hasGps={!!gps?.length}
          show={show}
          onToggle={(k) => setShow((s) => ({ ...s, [k]: !s[k] }))}
        />
        <div
          role="radiogroup"
          aria-label="Basemap"
          className="pointer-events-auto flex overflow-hidden rounded-md border border-rule bg-panel/95 text-[13px]"
        >
          {(Object.keys(BASEMAPS) as BasemapKey[]).map((k) => (
            <button
              key={k}
              role="radio"
              aria-checked={basemap === k}
              onClick={() => setBasemap(k)}
              className={`px-3 py-1.5 transition-colors ${basemap === k ? "bg-raised text-chart" : "text-muted hover:text-chart"}`}
            >
              {BASEMAPS[k].label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

type Layer = "gps" | "route";

function Legend({
  hasGps,
  hasRoute,
  show,
  onToggle,
}: {
  hasGps: boolean;
  hasRoute: boolean;
  show: Record<Layer, boolean>;
  onToggle: (k: Layer) => void;
}) {
  if (!hasGps) return <span />;
  const item = "flex w-full items-center gap-2.5 rounded px-1.5 py-1 text-left hover:bg-raised disabled:hover:bg-transparent";
  return (
    <div className="pointer-events-auto rounded-md border border-rule bg-panel/95 p-1.5 text-[13px] text-chart" aria-label="Map layers">
      <button className={item} aria-pressed={show.gps} onClick={() => onToggle("gps")} title="Show or hide the raw GPS fixes">
        <svg width="26" height="10" aria-hidden className={show.gps ? "" : "opacity-30"}>
          {[4, 13, 22].map((x) => <circle key={x} cx={x} cy={5} r={2.6} fill="#62d2e8" />)}
        </svg>
        <span className={show.gps ? "" : "text-faint line-through"}>Raw GPS fixes</span>
      </button>
      <button className={item} aria-pressed={show.route} disabled={!hasRoute} onClick={() => onToggle("route")}
        title="Show or hide the predicted route">
        <svg width="26" height="10" aria-hidden className={hasRoute && show.route ? "" : "opacity-30"}>
          <line x1="2" y1="5" x2="24" y2="5" stroke={hasRoute ? "#f2b33d" : "#5d7686"} strokeWidth="4" strokeLinecap="round" />
        </svg>
        <span className={!hasRoute ? "text-faint" : show.route ? "" : "text-faint line-through"}>
          Predicted route
        </span>
      </button>
    </div>
  );
}

function AnimatedRoute({ route }: { route: LatLon[][] }) {
  const ref = useRef<L.Polyline>(null);
  useEffect(() => {
    const el = ref.current?.getElement() as SVGPathElement | undefined;
    if (!el || typeof el.getTotalLength !== "function") return;
    const done = () => {
      // drop the dash so later zooms (which change path length) never clip the line
      el.classList.remove("route-draw");
      el.style.strokeDasharray = "";
      el.style.strokeDashoffset = "";
    };
    done();
    const len = el.getTotalLength();
    el.style.strokeDasharray = `${len}`;
    el.style.strokeDashoffset = `${len}`;
    el.getBoundingClientRect(); // commit the start state before transitioning
    el.classList.add("route-draw");
    el.style.strokeDashoffset = "0";
    el.addEventListener("transitionend", done, { once: true });
    const t = window.setTimeout(done, 2200);
    return () => {
      window.clearTimeout(t);
      el.removeEventListener("transitionend", done);
      done();
    };
  }, []);
  return (
    <>
      <Polyline positions={route} renderer={svg} pane="route" interactive={false}
        pathOptions={{ color: "#0f1c26", weight: 7, opacity: 0.7, lineCap: "round", lineJoin: "round" }} />
      <Polyline ref={ref} positions={route} renderer={svg} pane="route" interactive={false}
        pathOptions={{ color: "#f2b33d", weight: 5, opacity: 1, lineCap: "round", lineJoin: "round" }} />
    </>
  );
}

function FitBounds({ gps, route, fitKey }: Pick<Props, "gps" | "route" | "fitKey">) {
  const map = useMap();
  const bounds = useMemo(() => {
    const pts = [...(gps ?? []), ...(route?.flat() ?? [])];
    return pts.length ? L.latLngBounds(pts) : null;
  }, [gps, route]);

  useEffect(() => {
    if (!bounds) return;
    map.invalidateSize();
    map.fitBounds(bounds, { padding: [48, 48], animate: false });
    // fitKey (not bounds) decides when to re-frame, so the user's own zoom isn't overridden
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, map]);

  useEffect(() => {
    const ro = new ResizeObserver(() => map.invalidateSize());
    ro.observe(map.getContainer());
    return () => ro.disconnect();
  }, [map]);

  // "Fit to trip" button
  useEffect(() => {
    const Control = L.Control.extend({
      onAdd() {
        const div = L.DomUtil.create("div", "leaflet-bar");
        const a = L.DomUtil.create("a", "", div);
        a.href = "#";
        a.title = "Fit map to trip";
        a.setAttribute("role", "button");
        a.setAttribute("aria-label", "Fit map to trip");
        a.innerHTML =
          '<svg viewBox="0 0 20 20" width="16" height="16" style="margin-top:7px" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 7V3h4M13 3h4v4M17 13v4h-4M7 17H3v-4"/></svg>';
        L.DomEvent.on(a, "click", (e) => {
          L.DomEvent.preventDefault(e);
          const b = (map as L.Map & { _pathsnapBounds?: L.LatLngBounds })._pathsnapBounds;
          if (b) map.fitBounds(b, { padding: [48, 48] });
        });
        L.DomEvent.disableClickPropagation(div);
        return div;
      },
    });
    const c = new Control({ position: "topright" });
    c.addTo(map);
    return () => {
      c.remove();
    };
  }, [map]);

  useEffect(() => {
    (map as L.Map & { _pathsnapBounds?: L.LatLngBounds | null })._pathsnapBounds = bounds;
  }, [map, bounds]);

  return null;
}

function ZoomControl() {
  const map = useMap();
  useEffect(() => {
    const z = L.control.zoom({ position: "topright" });
    z.addTo(map);
    return () => {
      z.remove();
    };
  }, [map]);
  return null;
}

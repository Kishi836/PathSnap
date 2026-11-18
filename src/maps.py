"""Folium maps for PathSnap (role C; D can reuse ``trip_map`` without ``ref_arcs``).

Layers: raw GPS points, nearby road network, reference route (optional),
predicted route, and points the prediction got wrong (optional). Each layer is a
FeatureGroup so it can be toggled in the LayerControl.
"""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
from scipy.spatial import cKDTree

from .network import RoadGraph

COLORS = {"gps": "#1f77b4", "network": "#9e9e9e", "reference": "#2ca02c",
          "predicted": "#d62728", "wrong": "#ff7f0e"}


def _arc_lines(graph: RoadGraph, arc_ids):
    n = graph.trip.nodes
    f, t = graph.arc_from[arc_ids], graph.arc_to[arc_ids]
    return [[(n[a, 1], n[a, 0]), (n[b, 1], n[b, 0])] for a, b in zip(f, t)]


def _dedupe_consecutive(arcs):
    arcs = [int(a) for a in arcs if a >= 0]
    return [a for i, a in enumerate(arcs) if i == 0 or a != arcs[i - 1]]


def _view(lat, lon, width_px: int = 1100, height_px: int = 650):
    """Centre and the largest web-mercator zoom at which the trip's bounding box fits."""
    def merc_y(la):
        return np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
    lon_span = max(float(lon.max() - lon.min()), 1e-6) / 360
    y_span = max(float(merc_y(lat.max()) - merc_y(lat.min())), 1e-8) / (2 * np.pi)
    zoom = int(np.floor(min(np.log2(width_px / 256 / lon_span), np.log2(height_px / 256 / y_span))))
    center = [float((lat.min() + lat.max()) / 2), float((lon.min() + lon.max()) / 2)]
    return center, int(np.clip(zoom, 3, 18))


def _legend(items):
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in items)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            f'padding:8px 10px;border:1px solid #888;border-radius:4px;font:12px sans-serif">'
            f'<b>Legend</b>{rows}</div>')
    return folium.Element(html)


def trip_map(graph: RoadGraph, pred_arcs, ref_arcs=None, network_radius_m: float = 120.0,
             title: str | None = None, out_path: str | Path | None = None) -> folium.Map:
    trip = graph.trip
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    # centre + zoom computed here (Leaflet's fitBounds gives zoom 0 if the page is still 0-size)
    center, zoom = _view(lat, lon)
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;'
            f'padding:4px 8px;border:1px solid #888;font:14px sans-serif">{title}</div>'))

    # nearby road network: arcs with an endpoint within network_radius_m of the track
    tree = cKDTree(trip.gps_xy)
    near_node = tree.query(graph.node_xy, distance_upper_bound=network_radius_m)[0] < np.inf
    near_arcs = np.where(near_node[graph.arc_from] | near_node[graph.arc_to])[0]
    fg = folium.FeatureGroup(name=f"Road network (≤{network_radius_m:.0f} m)")
    for line in _arc_lines(graph, near_arcs):
        folium.PolyLine(line, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(fg)
    fg.add_to(m)

    legend = [("Road network", COLORS["network"])]
    if ref_arcs is not None:
        fg = folium.FeatureGroup(name="Reference route")
        for line in _arc_lines(graph, _dedupe_consecutive(ref_arcs)):
            folium.PolyLine(line, color=COLORS["reference"], weight=7, opacity=0.55).add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    fg = folium.FeatureGroup(name="Predicted route")
    for line in _arc_lines(graph, _dedupe_consecutive(pred_arcs)):
        folium.PolyLine(line, color=COLORS["predicted"], weight=3, opacity=0.9).add_to(fg)
    fg.add_to(m)
    legend.append(("Predicted route", COLORS["predicted"]))

    fg = folium.FeatureGroup(name="GPS points")
    for i, (la, lo) in enumerate(zip(lat, lon)):
        folium.CircleMarker((la, lo), radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"obs {i}, t={trip.track[i, 2]:.0f}s").add_to(fg)
    fg.add_to(m)
    legend.append(("GPS points", COLORS["gps"]))

    if ref_arcs is not None:
        pred, ref = np.asarray(pred_arcs), np.asarray(ref_arcs)
        wrong = np.where((ref >= 0) & (pred != ref))[0]
        fg = folium.FeatureGroup(name=f"Wrong points ({len(wrong)})", show=True)
        for i in wrong:
            folium.CircleMarker((lat[i], lon[i]), radius=4, color=COLORS["wrong"], fill=False, weight=2,
                                tooltip=f"obs {i}: pred {pred[i]} / ref {ref[i]}").add_to(fg)
        fg.add_to(m)
        legend.append(("Mismatched point", COLORS["wrong"]))

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_path))
    return m

"""Exploratory data analysis for PathSnap (train + student_test partitions only).

Run from the repo root:  python src/eda.py
Writes: results/eda/trip_inventory.csv, results/eda/summary.txt, results/eda/plots/*.png,
        results/eda/maps/*.html
"""
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent))
from data import (PARTITIONS, LocalProjection, arcs_within, load_trip, network_xy, point_segment_dist,
                  stationary_mask, trip_ids)

OUT = Path(__file__).resolve().parents[1] / "results" / "eda"
RADII = [10, 20, 30, 50, 75, 100]          # candidate search radii explored (m)
MAP_TRIPS = [("train", "00000001"), ("student_test", "00000000")]


def network_stats(trip, nodes_xy):
    arcs = trip["arcs"].to_numpy(dtype=np.int64)
    n = len(nodes_xy)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    n_weak, weak_lab = connected_components(g, directed=True, connection="weak")
    n_strong, strong_lab = connected_components(g, directed=True, connection="strong")
    degree = np.bincount(arcs.ravel(), minlength=n)
    pairs = set(map(tuple, arcs))
    length = np.linalg.norm(nodes_xy[arcs[:, 0]] - nodes_xy[arcs[:, 1]], axis=1)
    return {
        "n_nodes": n,
        "n_arcs": len(arcs),
        "invalid_arc_refs": int(((arcs < 0) | (arcs >= n)).sum()),
        "self_loops": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "duplicate_arcs": len(arcs) - len(pairs),
        "isolated_nodes": int((degree == 0).sum()),
        "weak_components": n_weak,
        "largest_weak_frac": np.bincount(weak_lab).max() / n,
        "strong_components": n_strong,
        "largest_strong_frac": np.bincount(strong_lab).max() / n,
        "frac_arcs_two_way": np.mean([(v, u) in pairs for u, v in arcs]),
        "zero_length_arcs": int((length < 0.01).sum()),
        "arc_len_median_m": np.median(length),
    }, weak_lab, strong_lab


def track_stats(track, xy):
    t = track.t.to_numpy()
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=dt > 0)
    return {
        "n_points": len(track),
        "stationary_frac": stationary_mask(xy, t).mean(),
        "nan_values": int(track.isna().sum().sum()),
        "invalid_coords": int((~track.lon.between(-180, 180) | ~track.lat.between(-90, 90)).sum()),
        "duplicate_timestamps": int((dt == 0).sum()),
        "non_increasing_timestamps": int((dt < 0).sum()),
        "t_start": t[0],
        "duration_s": t[-1] - t[0],
        "dt_median_s": np.median(dt),
        "dt_max_s": dt.max(),
        "frac_dt_eq_1": np.mean(dt == 1),
        "gaps_gt_5s": int((dt > 5).sum()),
        "gaps_gt_30s": int((dt > 30).sum()),
        "track_length_km": step.sum() / 1000,
        "speed_median_mps": np.nanmedian(speed),
        "speed_max_mps": np.nanmax(speed),
        "jumps_gt_50mps": int((speed > 50).sum()),     # > 180 km/h: implausible for a city car
        "lon_min": track.lon.min(), "lon_max": track.lon.max(),
        "lat_min": track.lat.min(), "lat_max": track.lat.max(),
    }, dt, speed


def route_stats(trip, route, xy, a, b, weak_lab, strong_lab, arcs):
    r = route.arc_id.to_numpy(dtype=np.int64)
    valid = (r >= 0) & (r < len(arcs))
    r = r[valid]
    ok = arcs[r[:-1], 1] == arcs[r[1:], 0]
    # distance from each GPS point to the nearest arc of the reference route
    d_route, _ = point_segment_dist(xy[:, None], a[r][None], b[r][None])
    nearest_pos = d_route.argmin(1)                 # position along the route of the nearest route arc
    d_min = d_route.min(1)
    return {
        "n_route_arcs": len(route),
        "invalid_route_ids": int((~valid).sum()),
        "route_unique_arcs": len(np.unique(r)),
        "route_consecutive_connected_frac": ok.mean() if len(ok) else np.nan,
        "route_length_km": np.linalg.norm(b[r] - a[r], axis=1).sum() / 1000,
        "route_in_one_weak_component": len(np.unique(weak_lab[arcs[r].ravel()])) == 1,
        "route_in_one_strong_component": len(np.unique(strong_lab[arcs[r].ravel()])) == 1,
        "gps_to_route_p50_m": np.percentile(d_min, 50),
        "gps_to_route_p90_m": np.percentile(d_min, 90),
        "gps_to_route_p99_m": np.percentile(d_min, 99),
        "gps_to_route_max_m": d_min.max(),
        # how often the nearest route arc steps *backwards* along the route: if high, a plain
        # nearest-route-arc rule is a noisy way to derive per-point labels
        "nearest_route_arc_backsteps_frac": np.mean(np.diff(nearest_pos) < 0),
        "first_gps_to_route_start_m": np.linalg.norm(xy[0] - a[r[0]]),
        "last_gps_to_route_end_m": np.linalg.norm(xy[-1] - b[r[-1]]),
    }, d_min, d_route


def candidate_stats(xy, a, b, d_route_min):
    pi, _, d, _ = arcs_within(xy, a, b, radius=max(RADII))
    nearest_any = np.full(len(xy), np.inf)
    np.minimum.at(nearest_any, pi, d)
    out = {"nearest_arc_p50_m": np.median(nearest_any), "nearest_arc_max_m": nearest_any.max()}
    counts = {}
    for r in RADII:
        c = np.bincount(pi[d <= r], minlength=len(xy))
        counts[r] = c
        out[f"cands_mean_r{r}"] = c.mean()
        out[f"no_cand_frac_r{r}"] = np.mean(c == 0)
        # route reachable within r: upper-bound proxy for candidate coverage before labels exist
        out[f"route_within_r{r}_frac"] = np.mean(d_route_min <= r)
    return out, counts


def run():
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    rows, pool = [], {"dt": [], "speed": [], "d_route": [], "stationary": [], "counts": {r: [] for r in RADII}}

    for part in PARTITIONS:
        for tid in trip_ids(part):
            trip = load_trip(part, tid)
            base = trip_dir_sizes(part, tid)
            proj = LocalProjection(trip["track"].lat.mean(), trip["track"].lon.mean())
            xy = proj(trip["track"].lon.to_numpy(), trip["track"].lat.to_numpy())
            nodes_xy, a, b = network_xy(trip, proj)
            arcs = trip["arcs"].to_numpy(dtype=np.int64)

            net, weak_lab, strong_lab = network_stats(trip, nodes_xy)
            trk, dt, speed = track_stats(trip["track"], xy)
            row = {"partition": part, "trip_id": tid, **base, **trk, **net}
            if trip["route"] is not None:
                rst, d_min, _ = route_stats(trip, trip["route"], xy, a, b, weak_lab, strong_lab, arcs)
                cst, counts = candidate_stats(xy, a, b, d_min)
                row.update(rst)
                row.update(cst)
                moving = ~stationary_mask(xy, trip["track"].t.to_numpy())
                row["gps_to_route_p50_moving_m"] = np.median(d_min[moving]) if moving.any() else np.nan
                row["far_gt30m_frac_moving"] = np.mean(d_min[moving] > 30) if moving.any() else np.nan
                if part == "train":
                    pool["d_route"].append(d_min)
                    pool["stationary"].append(~moving)
                    for r in RADII:
                        pool["counts"][r].append(counts[r])
            rows.append(row)
            pool["dt"].append(dt)
            pool["speed"].append(speed)
            print(f"{part}/{tid}: {trk['n_points']} pts, {net['n_arcs']} arcs")

    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "trip_inventory.csv", index=False)
    write_summary(inv)
    make_plots(inv, pool)
    for part, tid in MAP_TRIPS:
        folium_map(part, tid)
    print(f"done -> {OUT}")


def trip_dir_sizes(part, tid):
    from data import DATA_ROOT
    d = DATA_ROOT / part / tid
    files = {p.suffix.lstrip("."): p.stat().st_size for p in d.iterdir()}
    return {"n_files": len(files), **{f"kb_{k}": v / 1024 for k, v in sorted(files.items())}}


def write_summary(inv):
    lines = []
    for part, g in inv.groupby("partition", sort=False):
        lines.append(f"== {part}: {len(g)} trips, {g.n_points.sum()} GPS points")
        lines.append(f"files per trip: {sorted(g.n_files.unique())}")
        lines.append(g.describe().T[["mean", "min", "50%", "max"]].round(3).to_string())
        lines.append("")
    flags = {
        "trips with NaN values": inv.nan_values > 0,
        "trips with invalid coords": inv.invalid_coords > 0,
        "trips with duplicate timestamps": inv.duplicate_timestamps > 0,
        "trips with non-increasing timestamps": inv.non_increasing_timestamps > 0,
        "trips with a gap > 30 s": inv.gaps_gt_30s > 0,
        "trips with a >50 m/s jump": inv.jumps_gt_50mps > 0,
        "trips whose route is not fully connected": inv.route_consecutive_connected_frac < 1,
        "trips with invalid route ids": inv.invalid_route_ids > 0,
        "trips with isolated nodes": inv.isolated_nodes > 0,
        "trips with >1 weak component": inv.weak_components > 1,
    }
    lines.append("== anomaly flags (trip ids)")
    for name, mask in flags.items():
        lines.append(f"{name}: {int(mask.sum())} {inv.trip_id[mask].tolist()}")
    (OUT / "summary.txt").write_text("\n".join(lines))


def make_plots(inv, pool):
    P = OUT / "plots"
    colors = {"train": "tab:blue", "student_test": "tab:orange"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for part, g in inv.groupby("partition", sort=False):
        ax.hist(g.n_points, bins=20, alpha=0.6, label=f"{part} (n={len(g)})", color=colors[part])
    ax.set(xlabel="GPS points per trip", ylabel="trips", title="Trip length (train + student_test)")
    ax.legend(); fig.tight_layout(); fig.savefig(P / "points_per_trip.png", dpi=130); plt.close(fig)

    dt = np.concatenate(pool["dt"])
    vals, cnt = np.unique(np.minimum(dt, 31), return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(vals, cnt); ax.set_yscale("log")
    ax.set(xlabel="seconds between consecutive points (31 = >30)", ylabel="count (log)",
           title=f"Sampling intervals, all 90 trips ({np.mean(dt == 1):.1%} exactly 1 s)")
    fig.tight_layout(); fig.savefig(P / "sampling_intervals.png", dpi=130); plt.close(fig)

    sp = np.concatenate(pool["speed"]) * 3.6
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sp[np.isfinite(sp)], bins=100, range=(0, 200))
    ax.set(xlabel="speed between consecutive points (km/h)", ylabel="steps", title="Implied speed, all 90 trips")
    fig.tight_layout(); fig.savefig(P / "speed_distribution.png", dpi=130); plt.close(fig)

    d = np.concatenate(pool["d_route"])
    st = np.concatenate(pool["stationary"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for mask, name, col in ((~st, "moving", "tab:blue"), (st, "stationary (parked)", "tab:red")):
        p50, p90 = np.percentile(d[mask], [50, 90])
        ax.hist(np.minimum(d[mask], 100), bins=100, range=(0, 100), alpha=0.6, color=col,
                label=f"{name}: n={mask.sum()}, p50={p50:.1f} m, p90={p90:.1f} m")
    ax.set_yscale("log")
    ax.set(xlabel="distance from GPS point to nearest reference-route arc (m, 100 = ≥100)", ylabel="GPS points (log)",
           title="GPS noise relative to the reference route (train)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(P / "gps_to_route_distance.png", dpi=130); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    mean_c = [np.concatenate(pool["counts"][r]).mean() for r in RADII]
    cover = [np.mean(d <= r) for r in RADII]
    ax1.plot(RADII, mean_c, "o-", label="mean candidates / point")
    ax1.set(xlabel="candidate search radius (m)", ylabel="mean candidate arcs per point")
    ax2 = ax1.twinx(); ax2.plot(RADII, cover, "s--", c="tab:green", label="route within radius")
    ax2.set_ylabel("fraction of points with route arc in radius"); ax2.set_ylim(0, 1.02)
    ax1.set_title("Candidate set size vs coverage (train)")
    fig.legend(loc="lower right", fontsize=8); fig.tight_layout(); fig.savefig(P / "candidates_vs_radius.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for part, g in inv.groupby("partition", sort=False):
        axs[0].scatter(g.n_nodes, g.n_arcs, s=14, label=part, color=colors[part])
        axs[1].scatter(g.n_arcs, g.largest_strong_frac, s=14, label=part, color=colors[part])
    axs[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
    axs[1].set(xscale="log", xlabel="arcs", ylabel="largest strongly-connected component / nodes",
               title="Network connectivity per trip")
    axs[0].legend(); fig.tight_layout(); fig.savefig(P / "network_size.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for part, g in inv.groupby("partition", sort=False):
        ax.scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, s=16, label=part, color=colors[part])
    ax.set(xlabel="longitude", ylabel="latitude", title="Trip centres")
    ax.legend(); fig.tight_layout(); fig.savefig(P / "trip_locations.png", dpi=130); plt.close(fig)

    part, tid = MAP_TRIPS[0]
    trip = load_trip(part, tid)
    tr, nd = trip["track"], trip["nodes"]
    arcs = trip["arcs"].to_numpy(dtype=np.int64)
    pad = 0.003
    inbox = nd.lon.between(tr.lon.min() - pad, tr.lon.max() + pad) & nd.lat.between(tr.lat.min() - pad, tr.lat.max() + pad)
    sel = inbox.to_numpy()[arcs].all(1)
    fig, ax = plt.subplots(figsize=(8, 8))
    for u, v in arcs[sel]:
        ax.plot(nd.lon[[u, v]], nd.lat[[u, v]], c="0.8", lw=0.4)
    r = trip["route"].arc_id.to_numpy(dtype=np.int64)
    path = np.r_[arcs[r, 0], arcs[r[-1], 1]]
    ax.scatter(tr.lon, tr.lat, s=12, c="tab:blue", label="GPS points", zorder=3)
    ax.plot(nd.lon[path], nd.lat[path], c="tab:green", lw=1.2, label="reference route", zorder=4)
    ax.set(xlabel="longitude", ylabel="latitude", title=f"{part}/{tid}: GPS vs road network vs route")
    ax.set_aspect(1 / np.cos(np.radians(tr.lat.mean())))
    ax.legend(); fig.tight_layout(); fig.savefig(P / f"example_trip_{tid}.png", dpi=130); plt.close(fig)


def folium_map(part, tid, near_m=100):
    trip = load_trip(part, tid)
    tr, nd = trip["track"], trip["nodes"]
    arcs = trip["arcs"].to_numpy(dtype=np.int64)
    proj = LocalProjection(tr.lat.mean(), tr.lon.mean())
    xy = proj(tr.lon.to_numpy(), tr.lat.to_numpy())
    _, a, b = network_xy(trip, proj)
    _, near, _, _ = arcs_within(xy, a, b, radius=near_m)   # only road geometry near the trip
    near = np.unique(near)

    m = folium.Map(tiles="cartodbpositron")
    net = folium.FeatureGroup(name=f"Road network (arcs within {near_m} m of GPS)")
    for u, v in arcs[near]:
        folium.PolyLine([(nd.lat[u], nd.lon[u]), (nd.lat[v], nd.lon[v])], color="#888", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    if trip["route"] is not None:
        r = trip["route"].arc_id.to_numpy(dtype=np.int64)
        path = np.r_[arcs[r, 0], arcs[r[-1], 1]]
        folium.PolyLine(list(zip(nd.lat[path], nd.lon[path])), color="#2ca02c", weight=5, opacity=0.8,
                        tooltip="reference route").add_to(folium.FeatureGroup(name="Reference route").add_to(m))

    gps = folium.FeatureGroup(name="Raw GPS points")
    for i, (lon, lat, t) in enumerate(tr.itertuples(index=False)):
        folium.CircleMarker((lat, lon), radius=2, color="#1f77b4", fill=True, fill_opacity=0.8,
                            tooltip=f"obs {i}, t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((tr.lat.iloc[0], tr.lon.iloc[0]), tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((tr.lat.iloc[-1], tr.lon.iloc[-1]), tooltip="end", icon=folium.Icon(color="red")).add_to(m)

    legend = ("<div style='position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;"
              "border:1px solid #999;font-size:13px'>"
              f"<b>{part}/{tid}</b><br>"
              "<span style='color:#1f77b4'>&#9679;</span> raw GPS points<br>"
              "<span style='color:#2ca02c'>&#9644;</span> reference route<br>"
              "<span style='color:#888'>&#9644;</span> road network</div>")
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    m.save(OUT / "maps" / f"{part}_{tid}.html")


if __name__ == "__main__":
    run()

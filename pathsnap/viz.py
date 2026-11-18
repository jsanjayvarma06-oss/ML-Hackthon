"""Folium maps for the report. OWNER: Sanjay.

Raw files are lon, lat but folium wants [lat, lon]. Swap only here.
Every map needs distinct colours per layer, a legend or labels, and LayerControl.
Use only train or student_test trips for report maps, never evaluator trips.
"""
import folium
import numpy as np

from pathsnap.metrics import UNLABELLED

M_PER_DEG_LAT = 111_320.0


def _latlon(lonlat):
    """(N, 2+) lon, lat array -> list of [lat, lon]."""
    return [[float(lat), float(lon)] for lon, lat in lonlat[:, :2]]


def _arc_lines(trip, arc_ids):
    """Arc ids -> list of 2-point [lat, lon] lines, skipping -1 and consecutive repeats."""
    lines, prev = [], None
    for a in arc_ids:
        a = int(a)
        if a == UNLABELLED or a == prev:
            continue
        u, v = trip.arcs[a]
        lines.append(_latlon(trip.nodes[[u, v]]))
        prev = a
    return lines


def _nearby_arcs(trip, radius_m):
    """Arc ids with both nodes within about radius_m of the GPS path.

    A bounding box is too big for long diagonal trips (trip 13 gave an 8 MB map), so
    nodes are kept only if they fall in a radius_m grid cell next to a GPS point.
    Reverse twins (u->v and v->u) are drawn once.
    """
    cos_lat = np.cos(np.radians(trip.track[:, 1].mean()))

    def cells(lonlat):
        x = lonlat[:, 0] * M_PER_DEG_LAT * cos_lat / radius_m
        y = lonlat[:, 1] * M_PER_DEG_LAT / radius_m
        return np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)

    gx, gy = cells(trip.track)
    near = {(x + dx, y + dy) for x, y in set(zip(gx.tolist(), gy.tolist()))
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    nx, ny = cells(trip.nodes)
    inside = np.fromiter(((x, y) in near for x, y in zip(nx.tolist(), ny.tolist())),
                         dtype=bool, count=len(nx))
    keep = inside[trip.arcs[:, 0]] & inside[trip.arcs[:, 1]]
    seen, out = set(), []
    for a in np.flatnonzero(keep):
        key = frozenset(trip.arcs[a])
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def trip_map(trip, route=None, pred_arc=None, network_radius_m=300):
    """Folium map with layers: GPS points, nearby road network, reference route
    (if given), predicted route (if given). Returns a folium.Map.
    """
    gps = _latlon(trip.track)
    m = folium.Map(location=gps[0], zoom_start=15)

    net = folium.FeatureGroup(name="Road network (grey)")
    net_lines = [_latlon(trip.nodes[trip.arcs[a]]) for a in _nearby_arcs(trip, network_radius_m)]
    if net_lines:
        # one multi-line layer instead of thousands of PolyLines keeps the HTML small
        folium.PolyLine(net_lines, color="#888888", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    if route is not None:
        ref = folium.FeatureGroup(name="Reference route (green)")
        folium.PolyLine(_arc_lines(trip, route), color="green", weight=6, opacity=0.7).add_to(ref)
        ref.add_to(m)

    if pred_arc is not None:
        pred = folium.FeatureGroup(name="Predicted route (red)")
        lines = _arc_lines(trip, pred_arc)
        if lines:
            folium.PolyLine(lines, color="red", weight=3, opacity=0.9).add_to(pred)
        pred.add_to(m)

    pts = folium.FeatureGroup(name="GPS points (blue)")
    step = -(-len(gps) // 3000)  # ceil, so at most 3000 dots
    for p in gps[::step]:
        # small and see-through so the routes underneath stay visible
        folium.CircleMarker(p, radius=2, color="blue", weight=1, opacity=0.5,
                            fill=True, fill_opacity=0.3).add_to(pts)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(pts)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black")).add_to(pts)
    pts.add_to(m)

    lats, lons = trip.track[:, 1], trip.track[:, 0]
    m.fit_bounds([[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    return m

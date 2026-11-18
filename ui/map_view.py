"""Folium map of one API result for the evaluator UI. OWNER: Sanjay.

Input is the /predict JSON from docs/CONTRACTS.md. Its coordinates are already
[lat, lon], so no swapping here. The reference route never reaches the UI, so this
map shows only GPS points and the predicted route.
"""
import folium

MAX_GPS_DOTS = 3000


def _zoom_for(bounds, map_px=700):
    """Web-map zoom level that fits the bounding box into about map_px pixels.

    st_folium does not always apply fit_bounds (the map can open zoomed out to a whole
    continent), so the starting zoom is computed here as well. At zoom z the world is
    256 * 2**z pixels wide.
    """
    import math
    span = max(bounds[1][1] - bounds[0][1],
               (bounds[1][0] - bounds[0][0]) / max(math.cos(math.radians(bounds[0][0])), 0.2), 1e-4)
    return int(max(2, min(18, math.floor(math.log2(360.0 / span * map_px / 256.0)))))


def make_result_map(result):
    """Folium map with the GPS points, the predicted route and start/end markers.

    result: /predict JSON (coordinates already [lat, lon]). Opens centred on the trip,
    zoomed to fit it. Never draws the reference route.
    """
    gps, path = result["gps"], result["predicted_path"]
    if not gps:
        return folium.Map(location=[0, 0], zoom_start=2)

    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = folium.Map(location=center, zoom_start=_zoom_for(bounds))

    pts = folium.FeatureGroup(name="GPS points (blue)")
    step = -(-len(gps) // MAX_GPS_DOTS)  # ceil: thin out long trips so the browser keeps up
    for lat, lon in gps[::step]:
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True, weight=1).add_to(pts)
    pts.add_to(m)

    route = folium.FeatureGroup(name="Predicted route (red)")
    for seg in path:
        folium.PolyLine(seg, color="red", weight=4, opacity=0.8).add_to(route)
    route.add_to(m)

    ends = folium.FeatureGroup(name="Start (green) / End (black)")
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(ends)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="black")).add_to(ends)
    ends.add_to(m)

    m.fit_bounds(bounds)
    folium.LayerControl(collapsed=False).add_to(m)  # layer names double as the legend
    return m

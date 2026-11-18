"""Folium map of one API result for the evaluator UI. OWNER: Sanjay.

Input is the /predict JSON from docs/CONTRACTS.md. Its coordinates are already
[lat, lon], so no swapping here. The reference route never reaches the UI, so this
map shows only GPS points and the predicted route.
"""
import folium

MAX_GPS_DOTS = 3000


def make_result_map(result):
    gps, path = result["gps"], result["predicted_path"]
    if not gps:
        return folium.Map(location=[0, 0], zoom_start=2)

    m = folium.Map(location=gps[0], zoom_start=15)

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

    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl(collapsed=False).add_to(m)  # layer names double as the legend
    return m

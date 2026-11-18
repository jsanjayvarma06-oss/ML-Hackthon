"""Folium maps. OWNER: Samarth.

Raw files are lon, lat but folium wants [lat, lon]. Swap only here.
Every map needs distinct colours per layer, a legend or labels, and LayerControl.
"""


def trip_map(trip, route=None, pred_arc=None, network_radius_m=300):
    """Folium map with layers: GPS points, nearby road network, reference route
    (if given), predicted route (if given). Returns a folium.Map.
    """
    raise NotImplementedError("TODO(Samarth)")

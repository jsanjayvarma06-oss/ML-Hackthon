"""Coordinate conversion and geometry. OWNER: Sanjay.

Lon/lat are angles, so never measure distance on them directly. Each trip is
converted to metres with a local equirectangular projection centred on the trip:
    x = R * dlon * cos(lat0),  y = R * dlat   (radians)
Trips are spread across the world, so one fixed projection would not work. Over a
trip's extent (mostly a few km, at most ~60 km) this stays within a fraction of a
percent of true distance, which is far below GPS noise (5-20 m).
"""
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def to_local_xy(lon, lat, origin_lon, origin_lat):
    """Convert lon/lat arrays to x/y metres around an origin. Returns (x, y)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = np.radians(lon - origin_lon) * EARTH_RADIUS_M * np.cos(np.radians(origin_lat))
    y = np.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x, y


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments and position along each segment (0..1).

    Returns (distance_m, t). Vectorised: all inputs broadcast against each other.
    Zero-length segments return the distance to their single point and t = 0.
    """
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / len2
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(x1, y1, x2, y2):
    """Compass bearing in degrees from point 1 to point 2 (0 = north, 90 = east)."""
    return np.degrees(np.arctan2(x2 - x1, y2 - y1)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in 0..180 degrees."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


@dataclass
class TripXY:
    """A trip converted to local metres. Arrays are aligned with the raw files."""
    origin: tuple          # (lon0, lat0)
    track_xy: np.ndarray   # (N, 2)
    node_xy: np.ndarray    # (M, 2)
    seg: np.ndarray        # (A, 4) ax, ay, bx, by for every arc
    arc_len: np.ndarray    # (A,) metres
    arc_bearing: np.ndarray  # (A,) degrees, direction of travel from_node -> to_node


def project_trip(trip):
    """Convert track, nodes and arcs of a trip to local metres."""
    lon0 = float(np.mean(trip.track[:, 0]))
    lat0 = float(np.mean(trip.track[:, 1]))
    tx, ty = to_local_xy(trip.track[:, 0], trip.track[:, 1], lon0, lat0)
    nx, ny = to_local_xy(trip.nodes[:, 0], trip.nodes[:, 1], lon0, lat0)
    node_xy = np.column_stack([nx, ny])
    a = node_xy[trip.arcs[:, 0]]
    b = node_xy[trip.arcs[:, 1]]
    return TripXY(
        origin=(lon0, lat0),
        track_xy=np.column_stack([tx, ty]),
        node_xy=node_xy,
        seg=np.hstack([a, b]),
        arc_len=np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]),
        arc_bearing=bearing_deg(a[:, 0], a[:, 1], b[:, 0], b[:, 1]),
    )

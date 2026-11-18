"""Coordinate conversion and geometry. OWNER: Sanjay.

Lon/lat are angles, so never measure distance on them directly. Convert each trip
to metres around its own centre (trips are spread across the world).
"""


def to_local_xy(lon, lat, origin_lon, origin_lat):
    """Convert lon/lat arrays to x/y metres around an origin. Returns (x, y)."""
    raise NotImplementedError("TODO(Sanjay)")


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments and position along each segment (0..1).

    Returns (distance_m, t). Must be vectorised over numpy arrays.
    """
    raise NotImplementedError("TODO(Sanjay)")


def bearing_deg(x1, y1, x2, y2):
    """Compass-style bearing in degrees from point 1 to point 2."""
    raise NotImplementedError("TODO(Sanjay)")

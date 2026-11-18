"""Read raw trip files. OWNER: shared (done; change only after telling the team).

File formats (whitespace separated, no header):
    .track  lon lat t      GPS observations, t in seconds
    .nodes  lon lat        row index = node_id (0-based)
    .arcs   from to        row index = arc_id (0-based), directed from_node -> to_node
    .route  arc_id         reference route, ordered arc ids (NOT one row per GPS point)

Node and arc ids are local to each trip. Arc 5 in one trip has nothing to do with
arc 5 in another trip.

The route is loaded separately on purpose: prediction code must never see it.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray  # (N, 3) float: lon, lat, t
    nodes: np.ndarray  # (M, 2) float: lon, lat
    arcs: np.ndarray   # (A, 2) int: from_node, to_node


def _read(path, dtype, ncols):
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def list_trips(partition_dir):
    """Sorted trip ids (folder names) in a partition folder."""
    return sorted(p.name for p in Path(partition_dir).iterdir() if p.is_dir())


def load_trip(partition_dir, trip_id):
    """Load track, nodes and arcs for one trip. Does not load the route."""
    d = Path(partition_dir) / trip_id
    return Trip(
        trip_id=trip_id,
        track=_read(d / f"{trip_id}.track", float, 3),
        nodes=_read(d / f"{trip_id}.nodes", float, 2),
        arcs=_read(d / f"{trip_id}.arcs", int, 2),
    )


def load_route(partition_dir, trip_id):
    """Reference route as an int array of arc ids. For labels and scoring only."""
    path = Path(partition_dir) / trip_id / f"{trip_id}.route"
    return np.loadtxt(path, dtype=int, ndmin=1)

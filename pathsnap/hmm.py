"""HMM + Viterbi route reconstruction. OWNER: Sanjay.

Hidden state at GPS point i = one of its candidate arcs.
Emission   = how well the candidate explains the GPS point: log classifier probability,
             or (the "no ML" baseline) a Gaussian on the point-to-arc distance.
Transition = how plausible it is to drive from candidate a (at point i) to candidate b
             (at point i+1). A car moves roughly as far on the road network as the GPS
             says it moved, so we score exp(-|network distance - GPS distance| / beta).
             Network distance comes from a bounded Dijkstra search on the arc graph.
             Unreachable pairs get a fixed low score (a floor), never -inf, so one bad
             point cannot kill the whole path. Switching to the reverse twin of the
             current arc costs an extra `uturn` nats.
Everything is in log space. The trip is decoded separately for every segment (a time gap
longer than 60 s starts a new segment, see candidates.clean_track), but segments whose
end points are close together are bridged again, see `bridge_segments`.

The expensive part (network distances) does not depend on the tunable parameters, so it is
computed once by `network_distances` and reused by `decode` for every parameter setting.
"""
import heapq

import numpy as np

DEFAULT_PARAMS = {
    "beta": 30.0,         # metres; scale of the exponential transition penalty
    "floor": 25.0,        # nats; largest transition penalty, also used when unreachable
    "sigma": 10.0,        # metres; Gaussian width of the distance-only emission
    "emission_weight": 1.0,  # multiplies the emission log-probability
    "dist_weight": 0.0,   # v2: adds dist_weight * Gaussian(distance) to the classifier emission
    "prob_eps": 1e-6,     # classifier probabilities are clipped to [eps, 1] before the log
    "uturn": 100.0,       # nats; extra cost of switching to the reverse twin of the current arc
}
# Search radius for the network distance: twice the GPS distance plus slack, capped.
SEARCH_SLACK_M = 200.0
SEARCH_CAP_M = 5000.0
# Segments split at a time gap are joined again when the two GPS points are at most this far
# apart: a parked car that reports again after a pause is still on the same road.
BRIDGE_MAX_M = SEARCH_CAP_M


def merge_params(params=None):
    """Return DEFAULT_PARAMS updated with any keys in `params` (None = defaults)."""
    out = dict(DEFAULT_PARAMS)
    if params:
        out.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
    return out


def distance_emission(cand_dist, params=None):
    """Distance-only emission (the "no ML" baseline).

    cand_dist: list (one entry per original track row) of arrays of point-to-arc
    distances in metres. Returns a list of the same shape with Gaussian log-likelihoods
    -0.5 * (d / sigma)^2. Used when no trained model is available, and as the baseline
    in the report.
    """
    p = merge_params(params)
    return [-0.5 * (np.asarray(d) / p["sigma"]) ** 2 for d in cand_dist]


def mixed_emission(prob_list, cand_dist, params=None):
    """Classifier emission plus a distance term, in the units `decode` expects.

    decode multiplies every emission by emission_weight, so the result here is
        log p + (dist_weight / emission_weight) * (-0.5 (d / sigma)^2)
    which after decoding gives emission_weight * log p + dist_weight * Gaussian(d).
    With dist_weight 0 this is exactly classifier_emission. Chosen on train (E5).
    """
    p = merge_params(params)
    clf = classifier_emission(prob_list, params)
    if not p["dist_weight"]:
        return clf
    r = p["dist_weight"] / p["emission_weight"]
    return [c + r * d for c, d in zip(clf, distance_emission(cand_dist, params))]


def classifier_emission(prob_list, params=None):
    """Classifier emission: log of the model's probability that each candidate is right.

    prob_list: list (one entry per original track row) of probability arrays aligned with
    the candidate arrays. Probabilities are clipped to at least `prob_eps` so the log is finite.
    """
    p = merge_params(params)
    return [np.log(np.clip(np.asarray(q, dtype=float), p["prob_eps"], 1.0)) for q in prob_list]


def build_graph(arcs, arc_len):
    """Adjacency lists for the arc graph: node -> list of (to_node, length_m).

    arcs: (A, 2) from/to node ids; arc_len: (A,) metres. Self-loops are dropped (they never
    help a shortest path). Built once per trip and reused for every Dijkstra search.
    """
    n_nodes = int(arcs.max()) + 1 if len(arcs) else 0
    adj = [[] for _ in range(n_nodes)]
    for (u, v), w in zip(arcs, arc_len):
        if u != v:
            adj[u].append((int(v), float(w)))
    return adj


def bounded_dijkstra(adj, source, limit):
    """Shortest network distance (metres) from `source` to every node within `limit`.

    Returns a dict node -> distance. Stops exploring once the distance exceeds `limit`,
    so cost depends on the local neighbourhood, not on the size of the network (some
    trips have 300k arcs).
    """
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, np.inf):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd <= limit and nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def bridge_segments(keep_idx, segment_id, track_xy, max_gap_m=BRIDGE_MAX_M):
    """Merge segments that were split at a time gap when the car cannot have gone far.

    clean_track starts a new segment after a gap > 60 s. Features keep that split, but the HMM
    should not: a parked car with a 70 s gap in its fixes would otherwise restart Viterbi with
    no memory, and on a two-way road the first point then picks a direction at random (found on
    train trip 84: runs of 100-250 wrong points). We link the last point before the gap to the
    first point after it unless they are more than `max_gap_m` apart (then no path can be
    scored anyway). Returns a new segment_id array (same length).
    """
    seg = np.asarray(segment_id)
    out = seg.copy()
    pts = track_xy[keep_idx]
    cur = 0
    for i in range(1, len(seg)):
        if seg[i] != seg[i - 1] and np.hypot(*(pts[i] - pts[i - 1])) > max_gap_m:
            cur += 1
        out[i] = cur
    return out


def _steps(cand_arcs, keep_idx, segment_id):
    """List of (row_prev, row_next) pairs of consecutive kept points that both have
    candidates and lie in the same segment. Points without candidates are skipped over."""
    steps, last, last_seg = [], None, None
    for r, s in zip(keep_idx, segment_id):
        if len(cand_arcs[r]) == 0:
            continue
        if last is not None and s == last_seg:
            steps.append((last, r))
        last, last_seg = r, s
    return steps


def network_distances(cand_arcs, trip, txy, keep_idx, segment_id):
    """Network and GPS distance between every pair of candidates of consecutive points.

    Why: this is the slow part of the HMM and does not depend on beta/floor/sigma, so it is
    computed once per trip and reused by `decode` for any parameter setting.

    Returns a dict {row_next: (row_prev, d_net, d_gps, twin)} where d_net is a
    (K_prev, K_next) matrix (np.inf = unreachable within the search radius), d_gps is
    the straight-line distance in metres between the two GPS points and twin is a boolean
    (K_prev, K_next) matrix marking reverse-direction pairs. A pair of arcs is
    measured from the projection of the point on arc a to the projection on arc b:
        same arc:       |difference in position along the arc|
        different arcs: rest of a + shortest path (end of a -> start of b) + part of b
    """
    from pathsnap.geo import point_to_segment

    arcs, arc_len, seg = trip.arcs, txy.arc_len, txy.seg
    segment_id = bridge_segments(keep_idx, segment_id, txy.track_xy)
    adj = build_graph(arcs, arc_len)
    cache = {}
    out = {}
    for r0, r1 in _steps(cand_arcs, keep_idx, segment_id):
        a, b = cand_arcs[r0], cand_arcs[r1]
        p0, p1 = txy.track_xy[r0], txy.track_xy[r1]
        d_gps = float(np.hypot(*(p1 - p0)))
        limit = min(2.0 * d_gps + SEARCH_SLACK_M, SEARCH_CAP_M)
        # distance travelled along each arc up to / from the projection of the point
        sa, sb = seg[a], seg[b]
        _, ta = point_to_segment(p0[0], p0[1], sa[:, 0], sa[:, 1], sa[:, 2], sa[:, 3])
        _, tb = point_to_segment(p1[0], p1[1], sb[:, 0], sb[:, 1], sb[:, 2], sb[:, 3])
        rest_a = (1.0 - ta) * arc_len[a]
        part_b = tb * arc_len[b]
        mat = np.full((len(a), len(b)), np.inf)
        for i, arc_a in enumerate(a):
            to_node = int(arcs[arc_a, 1])
            key = (to_node, int(limit // 50))
            if key not in cache:
                if len(cache) > 20000:
                    cache.clear()
                cache[key] = bounded_dijkstra(adj, to_node, (key[1] + 1) * 50.0)
            reach = cache[key]
            for j, arc_b in enumerate(b):
                if arc_a == arc_b:
                    mat[i, j] = abs(tb[j] - ta[i]) * arc_len[arc_a]
                    continue
                d = reach.get(int(arcs[arc_b, 0]))
                if d is not None:
                    mat[i, j] = rest_a[i] + d + part_b[j]
        # twin[i, j] is True when candidate j is the reverse of candidate i (same nodes, opposite
        # direction): switching to it means the car turned round on the spot.
        twin = (arcs[a][:, None, 0] == arcs[b][None, :, 1]) & (arcs[a][:, None, 1] == arcs[b][None, :, 0])
        out[r1] = (r0, mat, d_gps, twin)
    return out


def decode(cand_arcs, emission_logp, net, keep_idx, segment_id, n_rows, params=None):
    """Viterbi decoding given precomputed network distances.

    cand_arcs / emission_logp: per original track row, aligned arrays.
    net: output of `network_distances`. n_rows: length of the original track.
    Returns an int array of length n_rows: arc id per original GPS row, -1 where the row
    was removed by cleaning or has no candidate.
    """
    p = merge_params(params)
    pred = np.full(n_rows, -1, dtype=int)
    w = p["emission_weight"]

    # chains of consecutive rows linked by `net`
    chain = []
    for r, s in zip(keep_idx, segment_id):
        if len(cand_arcs[r]) == 0:
            continue
        if r in net and chain:
            chain.append(r)
        else:
            if chain:
                _viterbi_chain(chain, cand_arcs, emission_logp, net, p, w, pred)
            chain = [r]
    if chain:
        _viterbi_chain(chain, cand_arcs, emission_logp, net, p, w, pred)
    return pred


def _viterbi_chain(chain, cand_arcs, emission_logp, net, p, w, pred):
    """Run Viterbi over one chain of rows (all in one segment) and write arcs into pred."""
    score = w * np.asarray(emission_logp[chain[0]], dtype=float)
    back = []
    for r_prev, r in zip(chain[:-1], chain[1:]):
        _, d_net, d_gps, twin = net[r]
        pen = np.abs(d_net - d_gps) / p["beta"]
        trans = -np.minimum(pen, p["floor"])          # exp(-|dnet - dgps| / beta), floored
        trans[~np.isfinite(d_net)] = -p["floor"]      # unreachable: floor, never -inf
        trans = trans - p["uturn"] * twin             # turning round on a two-way road is rare
        cand = score[:, None] + trans
        best = np.argmax(cand, axis=0)
        score = cand[best, np.arange(cand.shape[1])] + w * np.asarray(emission_logp[r], dtype=float)
        back.append(best)
    k = int(np.argmax(score))
    for step in range(len(chain) - 1, -1, -1):
        pred[chain[step]] = cand_arcs[chain[step]][k]
        if step > 0:
            k = int(back[step - 1][k])


def viterbi(cand_arcs, emission_logp, trip, txy, keep_idx, segment_id, params=None):
    """Best candidate arc for every GPS point (classifier or distance emission + HMM).

    cand_arcs: list per original track row of candidate arc id arrays (empty = none).
    emission_logp: list of arrays of the same shapes (see classifier_emission /
        distance_emission).
    trip, txy: the loaded trip and geo.project_trip(trip). keep_idx, segment_id: from
        candidates.clean_track.
    params: overrides for DEFAULT_PARAMS.
    Returns an int array, one arc id per original track row (-1 = none).
    """
    net = network_distances(cand_arcs, trip, txy, keep_idx, segment_id)
    return decode(cand_arcs, emission_logp, net, keep_idx, segment_id, len(trip.track), params)


def argmax_arcs(cand_arcs, emission_logp, keep_idx, n_rows):
    """Best candidate per point with no HMM (classifier-only / nearest-arc baseline).

    Returns an int array of length n_rows (-1 where a row has no candidate).
    """
    pred = np.full(n_rows, -1, dtype=int)
    for r in keep_idx:
        if len(cand_arcs[r]):
            pred[r] = cand_arcs[r][int(np.argmax(emission_logp[r]))]
    return pred

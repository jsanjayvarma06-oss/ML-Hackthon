import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
W = Path("/tmp/claude-0/-home-user/046e96e8-de52-54d3-81eb-e50048ac2b7c/scratchpad")/os.environ.get("PS_WORK","work")
sys.path.insert(0, str(W))
from pathsnap import features, hmm, io, labels, pipeline
from pathsnap.config import TRAIN_DIR
OOF = os.environ.get("PS_OOF","/mnt/project-files/handoff/work/A/oof/oof_lightgbm_tuned.parquet")
OUTD = Path("/tmp/claude-0/-home-user/046e96e8-de52-54d3-81eb-e50048ac2b7c/scratchpad")/os.environ.get("PS_CACHE","cache")
def one(tid):
    trip = io.load_trip(TRAIN_DIR, tid); route = io.load_route(TRAIN_DIR, tid)
    df, cand_arcs, cand_dist, keep_idx, segment_id, txy = features.trip_table(trip)
    true = labels.make_point_labels(trip, route, keep_idx=keep_idx, txy=txy)
    net = hmm.network_distances(cand_arcs, trip, txy, keep_idx, segment_id)
    oof = pd.read_parquet(OOF, filters=[("trip_id","==",tid)])
    prob = df[["point_idx","arc_id"]].merge(oof[["point_idx","arc_id","prob"]], on=["point_idx","arc_id"], how="left")["prob"].fillna(0.5)
    P = pipeline.split_by_point(df, prob.to_numpy(), cand_arcs, keep_idx)
    F = {c: pipeline.split_by_point(df, df[c].to_numpy(), cand_arcs, keep_idx) for c in ["speed_mps","heading_diff_deg","heading_reliable","proj_t"]}
    d = dict(tid=tid, arcs=trip.arcs, arc_len=txy.arc_len, cand_arcs=cand_arcs, cand_dist=cand_dist, keep_idx=keep_idx,
             segment_id=segment_id, true=true, net=net, prob=P, feat=F, track_xy=txy.track_xy, arc_bearing=txy.arc_bearing,
             t=trip.track[:,2], n=len(trip.track))
    pickle.dump(d, open(OUTD/f"{tid}.pkl","wb"))
    return tid
if __name__=="__main__":
    ids = io.list_trips(TRAIN_DIR)
    with ProcessPoolExecutor(4) as ex: r=list(ex.map(one, ids))
    print(len(r))

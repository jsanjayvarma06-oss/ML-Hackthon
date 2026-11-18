import sys, pickle, numpy as np
from pathlib import Path
import os
SP = Path("/tmp/claude-0/-home-user/046e96e8-de52-54d3-81eb-e50048ac2b7c/scratchpad")
WORK = SP/os.environ.get("PS_WORK","work")
CACHE = SP/os.environ.get("PS_CACHE","cache")
sys.path.insert(0, str(WORK))
from pathsnap import hmm
def load_all():
    return [pickle.load(open(p,"rb")) for p in sorted((CACHE).glob("0*.pkl"))]
def baseline(d, w=0.1, beta=10, floor=25):
    emis = [np.log(np.clip(np.asarray(q,float),1e-6,1)) for q in d["prob"]]
    return hmm.decode(d["cand_arcs"], emis, d["net"], d["keep_idx"], d["segment_id"], d["n"], dict(beta=beta,floor=floor,emission_weight=w))
def acc(ds, fn):
    c=n=0
    for d in ds:
        pred=fn(d); lab=d["true"]!=-1
        c+=int((pred[lab]==d["true"][lab]).sum()); n+=int(lab.sum())
    return c/n
def load_net(D):
    ds=load_all()
    for d in ds:
        n=pickle.load(open(CACHE/f"net_{d['tid']}.pkl","rb"))
        d["net_orig"]=d["net"]; d["net"]=n[D]
    return ds
def swap_prob(ds, oof_path):
    import pandas as pd
    o=pd.read_parquet(oof_path)
    for tid,g in o.groupby("trip_id"):
        d=next(x for x in ds if x["tid"]==tid)
        m={(int(p),int(a)):float(q) for p,a,q in zip(g.point_idx,g.arc_id,g.prob)}
        d["prob"]=[np.array([m.get((r,int(a)),0.5) for a in ca]) if len(ca) else np.empty(0) for r,ca in enumerate(d["cand_arcs"])]
    return ds

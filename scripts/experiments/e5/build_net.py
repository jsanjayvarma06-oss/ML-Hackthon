import sys, pickle, numpy as np
from lib import *
from pathsnap import io, geo
from pathsnap.config import TRAIN_DIR
from concurrent.futures import ProcessPoolExecutor
DS=[1e9]
def merged(d, D):
    k=d["keep_idx"]; seg=d["segment_id"]; xy=d["track_xy"][k]; ca=d["cand_arcs"]
    new=np.zeros(len(k),int); cur=0
    # merge by hopping over points without candidates: compare to previous kept point
    for i in range(1,len(k)):
        if seg[i]!=seg[i-1]:
            if np.hypot(*(xy[i]-xy[i-1]))>D: cur+=1
        new[i]=cur
    return new
def one(path):
    d=pickle.load(open(path,"rb")); trip=io.load_trip(TRAIN_DIR,d["tid"]); txy=geo.project_trip(trip)
    out={}
    for D in DS:
        s=merged(d,D); out[D]=hmm.network_distances(d["cand_arcs"],trip,txy,d["keep_idx"],s)
    pickle.dump(out,open(str(Path(path).parent/("net_"+Path(path).name)),"wb")); return 1
if __name__=="__main__":
    ps=sorted(CACHE.glob("0*.pkl"))
    with ProcessPoolExecutor(4) as ex: print(sum(ex.map(one,ps)))

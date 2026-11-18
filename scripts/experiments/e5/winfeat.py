import sys, numpy as np, pandas as pd
from lib import *
from pathsnap import io, geo, candidates
from pathsnap.config import TRAIN_DIR
from concurrent.futures import ProcessPoolExecutor
SCALES={"wm1":[1,2,3],"wm2":[5,10,15],"wm3":[25,50,100]}
def one(tid):
    trip=io.load_trip(TRAIN_DIR,tid); keep,seg,_=candidates.clean_track(trip.track); txy=geo.project_trip(trip)
    df=pd.read_parquet(SP/"work/data/processed/train.parquet",filters=[("trip_id","==",tid)])
    pos=np.full(len(trip.track),-1); pos[keep]=np.arange(len(keep))
    k=pos[df.point_idx.to_numpy()]; arc=df.arc_id.to_numpy()
    s=txy.seg[arc]; xy=txy.track_xy[keep]
    out={}
    for name,offs in SCALES.items():
        tot=np.zeros(len(df)); cnt=np.zeros(len(df))
        for o in offs:
            for sgn in (-1,1):
                j=k+sgn*o; ok=(j>=0)&(j<len(keep)); jj=np.clip(j,0,len(keep)-1); ok&=seg[jj]==seg[k]
                q=xy[jj]; dd,_=geo.point_to_segment(q[:,0],q[:,1],s[:,0],s[:,1],s[:,2],s[:,3])
                tot+=np.where(ok,np.minimum(dd,100),0); cnt+=ok
        out[name]=np.where(cnt>0,tot/np.maximum(cnt,1),np.nan)
    o=pd.DataFrame(out); o["trip_id"]=df.trip_id.to_numpy(); o["point_idx"]=df.point_idx.to_numpy(); o["arc_id"]=arc
    # fill missing with own distance, add relative-to-best features
    for name in SCALES:
        o[name]=o[name].fillna(pd.Series(df.dist_m.to_numpy()))
        o[name+"_rel"]=o[name]-o.groupby("point_idx")[name].transform("min")
    return o
if __name__=="__main__":
    ids=io.list_trips(TRAIN_DIR)
    with ProcessPoolExecutor(4) as ex: r=list(ex.map(one,ids))
    pd.concat(r,ignore_index=True).to_parquet(SP/"winfeat.parquet"); print("ok")

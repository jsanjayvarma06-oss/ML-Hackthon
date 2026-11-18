import sys, numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user/046e96e8-de52-54d3-81eb-e50048ac2b7c/scratchpad/work")
from pathsnap import features, hmm, io, labels, pipeline
from pathsnap.config import TRAIN_DIR
from concurrent.futures import ProcessPoolExecutor
OOF=sys.argv[1]; PRM=eval(sys.argv[2])
def one(tid):
    trip=io.load_trip(TRAIN_DIR,tid); route=io.load_route(TRAIN_DIR,tid)
    df,ca,cd,keep,seg,txy=features.trip_table(trip)
    true=labels.make_point_labels(trip,route,keep_idx=keep,txy=txy)
    oof=pd.read_parquet(OOF,filters=[("trip_id","==",tid)])
    prob=df[["point_idx","arc_id"]].merge(oof[["point_idx","arc_id","prob"]],on=["point_idx","arc_id"],how="left")["prob"].fillna(0.5)
    em=hmm.classifier_emission(pipeline.split_by_point(df,prob.to_numpy(),ca,keep))
    pred=hmm.viterbi(ca,em,trip,txy,keep,seg,PRM); lab=true!=-1
    return int((pred[lab]==true[lab]).sum()), int(lab.sum())
if __name__=="__main__":
    ids=io.list_trips(TRAIN_DIR)
    with ProcessPoolExecutor(4) as ex: r=list(ex.map(one,ids))
    print(PRM, sum(a for a,b in r)/sum(b for a,b in r))

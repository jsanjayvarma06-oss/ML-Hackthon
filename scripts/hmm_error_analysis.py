"""HMM error breakdown on train (out-of-fold probabilities). OWNER: Sanjay.

Run from the repo root:  python scripts/hmm_error_analysis.py
For every labelled train point, decodes with the chosen HMM params (hmm_params.json) and
the tuned LightGBM OOF probabilities, then sorts each wrong point into one bucket:
  miss      the reference arc is not among the 8 candidates (candidate stage)
  twin      predicted the reverse-direction twin of the reference arc
  adjacent  predicted an arc that touches the reference arc (junction / arc boundary)
  other     anything else
Writes results/pipeline/hmm_error_breakdown.csv (one row per trip). Train only.
"""
import sys, numpy as np, pandas as pd, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathsnap import io, features, labels, hmm, pipeline
from pathsnap.config import TRAIN_DIR
P=json.load(open('results/pipeline/hmm_params.json'))['classifier_hmm']
def run(tid):
    trip=io.load_trip(TRAIN_DIR,tid); route=io.load_route(TRAIN_DIR,tid)
    df,ca,cd,ki,seg,txy=features.trip_table(trip)
    true=labels.make_point_labels(trip,route,keep_idx=ki,txy=txy)
    oof=pd.read_parquet('results/models/oof_lightgbm_tuned.parquet',filters=[('trip_id','==',tid)])
    prob=df[['point_idx','arc_id']].merge(oof[['point_idx','arc_id','prob']],on=['point_idx','arc_id'],how='left')['prob'].fillna(0.5)
    em=hmm.classifier_emission(pipeline.split_by_point(df,prob.to_numpy(),ca,ki))
    net=hmm.network_distances(ca,trip,txy,ki,seg)
    pred=hmm.decode(ca,em,net,ki,seg,len(trip.track),P)
    A=trip.arcs; c={'n':0,'ok':0,'miss':0,'twin':0,'adjacent':0,'other':0,'nopred':0}
    for i in np.flatnonzero(true!=-1):
        t,p=true[i],pred[i]; c['n']+=1
        if p==t: c['ok']+=1
        elif p==-1: c['nopred']+=1
        elif t not in set(ca[i]): c['miss']+=1
        elif A[p,0]==A[t,1] and A[p,1]==A[t,0]: c['twin']+=1
        elif A[p,1]==A[t,0] or A[t,1]==A[p,0] or A[p,0]==A[t,0] or A[p,1]==A[t,1]: c['adjacent']+=1
        else: c['other']+=1
    return tid,c
def main():
    ids=io.list_trips(TRAIN_DIR)
    with ProcessPoolExecutor(4) as ex: res=list(ex.map(run,ids))
    d=pd.DataFrame([dict(trip=t,**c) for t,c in res]); s=d.drop(columns='trip').sum()
    print(s.to_dict(), (s/s['n']).round(4).to_dict())
    d['acc']=d.ok/d.n; print(d.sort_values('acc').head(10).to_string())
    d.to_csv('results/pipeline/hmm_error_breakdown.csv',index=False)


if __name__ == "__main__":
    main()

"""Mixed-emission HMM grid on train (out-of-fold probabilities). OWNER: Sanjay.

Run from the repo root:  python scripts/hmm_mixed_emission.py
Emission = w_clf * log(classifier prob) + w_dist * (-0.5 (d / 10 m)^2), over a grid of
w_clf, w_dist, beta and floor. Every setting is logged to
results/pipeline/hmm_mixed_emission_grid.csv (pooled train Top-1). Train only.
"""
import sys, itertools, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathsnap import io, features, labels, hmm, pipeline
from pathsnap.config import TRAIN_DIR
GRID=list(itertools.product([0.1,0.25,0.5,1.0],[0,0.25,0.5,1.0,2.0],[5,10,20,50],[25,50]))
def run(tid):
    trip=io.load_trip(TRAIN_DIR,tid); route=io.load_route(TRAIN_DIR,tid)
    df,ca,cd,ki,seg,txy=features.trip_table(trip)
    true=labels.make_point_labels(trip,route,keep_idx=ki,txy=txy)
    oof=pd.read_parquet('results/models/oof_lightgbm_tuned.parquet',filters=[('trip_id','==',tid)])
    prob=df[['point_idx','arc_id']].merge(oof[['point_idx','arc_id','prob']],on=['point_idx','arc_id'],how='left')['prob'].fillna(0.5)
    ec=hmm.classifier_emission(pipeline.split_by_point(df,prob.to_numpy(),ca,ki))
    ed=hmm.distance_emission(cd)
    net=hmm.network_distances(ca,trip,txy,ki,seg)
    lab=true!=-1; out=[]
    for w,wd,b,f in GRID:
        em=[w*a+wd*c for a,c in zip(ec,ed)]
        pred=hmm.decode(ca,em,net,ki,seg,len(trip.track),dict(beta=b,floor=f,emission_weight=1.0))
        out.append(int((pred[lab]==true[lab]).sum()))
    return int(lab.sum()), out
def main():
    ids=io.list_trips(TRAIN_DIR)
    with ProcessPoolExecutor(4) as ex: res=list(ex.map(run,ids))
    n=sum(r[0] for r in res); tot=np.sum([r[1] for r in res],axis=0)/n
    d=pd.DataFrame(GRID,columns=['w_clf','w_dist','beta','floor']); d['top1']=tot
    d.to_csv('results/pipeline/hmm_mixed_emission_grid.csv',index=False)
    print(d.sort_values('top1',ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()

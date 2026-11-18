import sys, json, numpy as np, pandas as pd
from lib import *
from pathsnap import features, models
from sklearn.model_selection import GroupKFold
tag=sys.argv[1]
df=pd.read_parquet(WORK/"data/processed/train.parquet")
df=df[df.true_arc!=-1].reset_index(drop=True)
w=pd.read_parquet(SP/"winfeat.parquet")
df=df.merge(w,on=["trip_id","point_idx","arc_id"],how="left") if tag=="win" else df
print(len(df))
base=list(features.FEATURE_COLUMNS); extra=["wm1","wm2","wm3","wm1_rel","wm2_rel","wm3_rel"]
P=json.load(open(SP/"work/results/models/best_params.json"))["params"]
cols=base+extra if tag=="win" else base
oof=np.zeros(len(df))
for f,(tr,va) in enumerate(GroupKFold(5).split(df,groups=df.trip_id)):
    m=models.lightgbm(**P); m.fit(df.loc[tr,cols],df.loc[tr,"label"]); oof[va]=m.predict_proba(df.loc[va,cols])[:,1]; print("fold",f,flush=True)
out=df[["trip_id","point_idx","arc_id"]].copy(); out["prob"]=oof; out.to_parquet(SP/f"oof_{tag}{os.environ.get('PS_SUF','')}.parquet")
df["prob"]=oof
top=df.loc[df.groupby(["trip_id","point_idx"]).prob.idxmax()]
print(tag,"classifier-only top1", top.label.mean())

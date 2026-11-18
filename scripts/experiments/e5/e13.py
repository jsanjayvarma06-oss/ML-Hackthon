from dec import *
import collections
ds=load_net(1e9); P=dict(beta=30,floor=50,w=0.1,uturn=100)
cnt=collections.Counter(); N=0; per=collections.Counter(); tot=0
for d in ds:
    em=[np.log(np.clip(np.asarray(q,float),1e-6,1)) for q in d["prob"]]
    pred=run(d,em,P); lab=d["true"]!=-1; arcs=d["arcs"]; N+=lab.sum()
    for r in np.flatnonzero(lab&(pred!=d["true"])):
        t=d["true"][r]; p=pred[r]; per[d["tid"]]+=1
        if t not in set(d["cand_arcs"][r]): cnt["miss"]+=1; continue
        if p==-1: cnt["nopred"]+=1; continue
        if arcs[t,0]==arcs[p,1] and arcs[t,1]==arcs[p,0]: cnt["twin"]+=1
        elif len({*arcs[t]}&{*arcs[p]})>0: cnt["adjacent"]+=1
        else: cnt["other"]+=1
print({k:round(v/N,4) for k,v in cnt.items()}, "err", round(sum(cnt.values())/N,4))
print(per.most_common(8))

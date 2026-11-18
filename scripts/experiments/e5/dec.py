from lib import *
from scipy.special import logsumexp

def chains_of(d):
    net=d["net"]; ca=d["cand_arcs"]; out=[]; chain=[]
    for r in d["keep_idx"]:
        if len(ca[r])==0: continue
        if r in net and chain: chain.append(r)
        else:
            if chain: out.append(chain)
            chain=[r]
    if chain: out.append(chain)
    return out

def trans_base(d, r, p):
    r0, dn, dg = d["net"][r]
    pen = np.abs(dn-dg)/(p["beta"]+p.get("b1",0)*dg)
    t = -np.minimum(pen, p["floor"]); t[~np.isfinite(dn)] = -p["floor"]
    if p.get("uturn",0)>0:
        a=d["cand_arcs"][r0]; b=d["cand_arcs"][r]; arcs=d["arcs"]
        tw = (arcs[a][:,None,0]==arcs[b][None,:,1])&(arcs[a][:,None,1]==arcs[b][None,:,0])
        t = t - p["uturn"]*tw
    if p.get("sw",0)>0:
        a=d["cand_arcs"][r0]; b=d["cand_arcs"][r]
        t = t - p["sw"]*(a[:,None]!=b[None,:])
    return t

def run(d, emis, p, mode="viterbi", tfn=trans_base):
    pred=np.full(d["n"],-1,int); ca=d["cand_arcs"]
    for ch in chains_of(d):
        E=[p["w"]*np.asarray(emis[r],float) for r in ch]
        Ts=[tfn(d,r,p) for r in ch[1:]]
        if mode=="viterbi":
            sc=E[0]; back=[]
            for T,e in zip(Ts,E[1:]):
                c=sc[:,None]+T; b=np.argmax(c,0); sc=c[b,np.arange(c.shape[1])]+e; back.append(b)
            k=int(np.argmax(sc))
            for s in range(len(ch)-1,-1,-1):
                pred[ch[s]]=ca[ch[s]][k]
                if s>0: k=int(back[s-1][k])
        else:
            tmp=p.get("temp",1.0)
            al=[E[0]]
            for T,e in zip(Ts,E[1:]): al.append(logsumexp(al[-1][:,None]+T,0)+e)
            be=[np.zeros(len(E[-1]))]
            for T,e in zip(Ts[::-1],E[::-1][:-1]): be.append(logsumexp(T+(e+be[-1])[None,:],1))
            be=be[::-1]
            for s,r in enumerate(ch):
                pred[r]=ca[r][int(np.argmax(al[s]+be[s]))]
    return pred

def evalp(ds, p, mode="viterbi", tfn=trans_base, emisf=None):
    c=n=0
    for d in ds:
        em = emisf(d) if emisf else [np.log(np.clip(np.asarray(q,float),1e-6,1)) for q in d["prob"]]
        pred=run(d,em,p,mode,tfn); lab=d["true"]!=-1
        c+=int((pred[lab]==d["true"][lab]).sum()); n+=int(lab.sum())
    return c/n

import multiprocessing as mp
_G={}
def _one(args):
    i,p,mode,tfn_name,emname=args
    d=_G["ds"][i]
    em=_G["emfs"][emname](d,p)
    pred=run(d,em,p,mode,_G["tfns"][tfn_name]); lab=d["true"]!=-1
    return int((pred[lab]==d["true"][lab]).sum()), int(lab.sum())
def setup(ds, emfs=None, tfns=None):
    _G["ds"]=ds
    _G["emfs"]={"clf":lambda d,p:[np.log(np.clip(np.asarray(q,float),1e-6,1)) for q in d["prob"]],
    "mix":lambda d,p:[np.log(np.clip(np.asarray(q,float),1e-6,1))+p["wd"]*(-0.5*(np.asarray(c)/10.0)**2)/max(p["w"],1e-9) for q,c in zip(d["prob"],d["cand_dist"])]}
    if emfs: _G["emfs"].update(emfs)
    _G["tfns"]={"base":trans_base}
    if tfns: _G["tfns"].update(tfns)
def pev(p, mode="viterbi", tfn="base", em="clf"):
    with mp.get_context("fork").Pool(4) as pool:
        r=pool.map(_one,[(i,p,mode,tfn,em) for i in range(len(_G["ds"]))])
    return sum(a for a,b in r)/sum(b for a,b in r)

def pev_trips(p, mode="viterbi", tfn="base", em="clf"):
    with mp.get_context("fork").Pool(4) as pool:
        return pool.map(_one,[(i,p,mode,tfn,em) for i in range(len(_G["ds"]))])

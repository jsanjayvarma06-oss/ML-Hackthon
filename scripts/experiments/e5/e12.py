from dec import *
import itertools
ds=load_net(1e9); setup(ds)
B=dict(beta=20,floor=50,w=0.1,uturn=100)
def P(**k): return dict(B,**k)
for beta in [30,50,100]: print("beta",beta,round(pev(P(beta=beta)),4),flush=True)
for fl in [100,200]: print("floor",fl,round(pev(P(floor=fl)),4),flush=True)
for sw in [0.5,1,2,4]: print("sw",sw,round(pev(P(sw=sw)),4),flush=True)
for wd in [0.01,0.02,0.05,0.1]:
    for w in [0.1]: print("mix wd",wd,round(pev(P(wd=wd),em="mix"),4),flush=True)
print("END")

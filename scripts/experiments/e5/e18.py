from dec import *
import itertools
ds=load_net(1e9); setup(ds)
res=[]
for beta,w,fl in itertools.product([15,20,30],[0.02,0.03,0.05,0.07],[50,100]):
    a=pev(dict(beta=beta,floor=fl,w=w,uturn=100)); res.append((a,beta,w,fl)); print(beta,w,fl,round(a,4),flush=True)
print("BEST",sorted(res)[-3:]); print("END")

from dec import *
ds=load_net(1e9); setup(ds)
import time; t0=time.time()
print("base+bridge", round(pev(dict(beta=10,floor=25,w=0.1)),4), time.time()-t0, flush=True)
for u in [2,5,10,20,40]:
    print("uturn",u, round(pev(dict(beta=10,floor=25,w=0.1,uturn=u)),4), flush=True)

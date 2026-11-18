from dec import *
import itertools, csv
ds=load_net(1e9); setup(ds)
rows=[]
for beta,floor,w,u in itertools.product([5,10,20],[25,50],[0.05,0.1,0.25,0.5],[20,40,100]):
    a=pev(dict(beta=beta,floor=floor,w=w,uturn=u)); rows.append((beta,floor,w,u,a)); print(beta,floor,w,u,round(a,4),flush=True)
rows.sort(key=lambda x:-x[-1]); print("BEST",rows[:5])

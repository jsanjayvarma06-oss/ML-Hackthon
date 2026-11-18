from dec import *
import itertools, json
ds=load_net(1e9); setup(ds)
tids=[d["tid"] for d in ds]
grid=list(itertools.product([10,15,20,30,40],[0.03,0.05,0.07,0.1],[50],[100]))
R={}
for beta,w,fl,u in grid:
    R[(beta,w,fl,u)]=pev_trips(dict(beta=beta,floor=fl,w=w,uturn=u))
    print(beta,w,fl,u, round(sum(a for a,b in R[(beta,w,fl,u)])/sum(b for a,b in R[(beta,w,fl,u)]),4),flush=True)
def sc(k,idx): r=R[k]; return sum(r[i][0] for i in idx)/sum(r[i][1] for i in idx)
n=len(ds); A=[i for i in range(n) if i%2==0]; B=[i for i in range(n) if i%2==1]
for tr,te,name in [(A,B,"A->B"),(B,A,"B->A")]:
    k=max(R,key=lambda k:sc(k,tr)); print(name,"chosen",k,"train-half",round(sc(k,tr),4),"held-out half",round(sc(k,te),4))
allk=max(R,key=lambda k:sc(k,range(n))); print("best overall",allk,round(sc(allk,range(n)),4))
json.dump({str(k):v for k,v in R.items()},open("/tmp/claude-0/-home-user/046e96e8-de52-54d3-81eb-e50048ac2b7c/scratchpad/e19_k16.json","w"))
print("END")

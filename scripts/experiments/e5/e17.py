from dec import *
ds=load_net(1e9); setup(ds)
for P in [dict(beta=10,floor=25,w=0.1),dict(beta=30,floor=50,w=0.1,uturn=100),dict(beta=20,floor=50,w=0.1,uturn=100),dict(beta=30,floor=50,w=0.05,uturn=100),dict(beta=30,floor=50,w=0.2,uturn=100)]:
    print(P, round(pev(P),4), flush=True)
print("END")

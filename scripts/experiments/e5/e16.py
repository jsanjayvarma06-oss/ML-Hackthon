from dec import *
ds=load_net(1e9); setup(ds)
for beta,b1 in [(30,0),(20,0.1),(20,0.3),(10,0.3),(10,0.5),(15,0.2),(30,0.1),(10,1.0)]:
    print(beta,b1, round(pev(dict(beta=beta,b1=b1,floor=50,w=0.1,uturn=100)),4), flush=True)
print("END")

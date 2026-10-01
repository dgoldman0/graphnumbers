import random, itertools
from indep import *
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match
random.seed(1)
def relabel(adj, root):
    n=len(adj); perm=list(range(n)); random.shuffle(perm)
    new=[set() for _ in range(n)]
    for v in range(n):
        for w in adj[v]: new[perm[v]].add(perm[w])
    return new, perm[root]
def to_root0(adj, root):
    # move root to 0
    n=len(adj); perm=list(range(n)); perm[0],perm[root]=perm[root],perm[0]
    # perm maps new->old ; build
    inv={old:new for new,old in enumerate(perm)}
    out=[set() for _ in range(n)]
    for v in range(n):
        for w in adj[v]: out[inv[v]].add(inv[w])
    return out
# 1) invariance under relabelling
bad=0
for trial in range(300):
    n=random.randint(2,9); p=random.random()
    G=nx.gnp_random_graph(n,p,seed=random.randint(0,10**9))
    if not nx.is_connected(G): continue
    adj=[set(G[v]) for v in range(n)]
    root=random.randrange(n)
    a=to_root0(adj,root)
    b,rb=relabel(adj,root); b=to_root0(b,rb)
    if canonical_form(a)!=canonical_form(b): bad+=1
print("relabel failures",bad)
# 2) agreement with VF2 on all pairs from a pool
pool=[]
for trial in range(400):
    n=random.randint(1,7); p=random.random()
    G=nx.gnp_random_graph(n,p,seed=random.randint(0,10**9))
    if not nx.is_connected(G): continue
    adj=[set(G[v]) for v in range(n)]
    root=random.randrange(n)
    pool.append(to_root0(adj,root))
cf=[canonical_form(a) for a in pool]
Gs=[nx_rooted(a) for a in pool]
mism=0
for i in range(len(pool)):
    for j in range(i+1,len(pool)):
        iso=GraphMatcher(Gs[i],Gs[j],node_match=categorical_node_match('tag',None)).is_isomorphic()
        if iso!=(cf[i]==cf[j]): mism+=1
print("pairs",len(pool)*(len(pool)-1)//2,"mismatches",mism, "classes", len(set(cf)))

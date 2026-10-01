import sys, random, time, pickle
sys.path.insert(0,'.')
from glib import *
from types_gen import types_r2
from check_realization import rank_mod, exact_rank, sample_graphs, P1, P2
from collections import defaultdict
D, r = 3, 2
types = types_r2(3)
idx = {canon(t,0): i for i,t in enumerate(types)}
n = len(types)
t0=time.time()
sizes = [(len(t), len(edges_of(t))) for t in types]
M = []
for i,F in enumerate(types):
    row=[0]*n
    for j,B in enumerate(types):
        if sizes[i][0]<=sizes[j][0] and sizes[i][1]<=sizes[j][1]:
            row[j]=inj_rooted(F,0,B,0)
    M.append(row)
groups=defaultdict(list)
for i,F in enumerate(types): groups[canon(F,None)].append(i)
diffs=[]
for g,mem in groups.items():
    for i in mem[1:]:
        diffs.append([a-b for a,b in zip(M[i],M[mem[0]])])
print('dimK\'', exact_rank(diffs), 'time', time.time()-t0)
def hvec(g):
    v=[0]*n
    for o in g: v[idx[canon(ball(g,o,r),0)]]+=1
    return v
graphs = sample_graphs(D, 40, 4000)
seen=set()
H=[]
for g in graphs:
    h=hvec(g); H.append(h)
    seen |= {i for i,x in enumerate(h) if x}
print('types seen in random sample:', len(seen), 'of', n)
missing=[i for i in range(n) if i not in seen]
print('missing type sizes:', sorted(len(types[i]) for i in missing))
# add each type itself as a graph
for t in types:
    H.append(hvec(t))
print('rank with types-as-graphs added:', rank_mod(H,P1), rank_mod(H,P2))
pickle.dump({'types':types,'diffs':diffs,'H':H}, open('r32.pkl','wb'))

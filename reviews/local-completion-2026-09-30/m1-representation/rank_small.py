import sys, pickle
sys.path.insert(0,'.')
from check_realization import rank_mod, P1
d = pickle.load(open('r32.pkl','rb'))
H = d['H']
sizes = [sum(h) for h in H]   # histogram mass = vertex count
for thr in [7, 8, 9, 10, 13, 52]:
    sub = [h for h, s in zip(H, sizes) if s <= thr]
    print('graphs with <=', thr, 'vertices:', len(sub), 'rank', rank_mod(sub, P1))

import mpmath as mp
from core import lap
mp.mp.dps = 100
n = 5
G = [(0,1),(0,2),(0,3),(0,4),(1,2)]
cuts = [(0,1),(0,3),(0,4)]
atoms = []
for mask in range(8):
    E = [e for e in G if e not in [cuts[i] for i in range(3) if mask>>i&1]]
    ev, _ = mp.eigsy(mp.matrix(lap(n, E)))
    sg = (-1)**(3-bin(mask).count("1"))
    for x in ev: atoms.append((x, sg))
# merge equal atoms
atoms.sort(key=lambda a: a[0])
merged = []
for x, s in atoms:
    if merged and abs(merged[-1][0]-x) < mp.mpf(10)**-60: merged[-1][1] += s
    else: merged.append([x, s])
merged = [(x, s) for x, s in merged if s != 0]
print("signed spectral atoms:", [(mp.nstr(x, 12), s) for x, s in merged])
H = lambda t: sum(s*mp.e**(-t*x) for x, s in merged)
for t in [1, 3, 10, 12, 15, 20, 40]:
    print(t, mp.nstr(H(mp.mpf(t)), 20))
print("roots:", mp.nstr(mp.findroot(H, (1, 3), solver="bisect"), 15), mp.nstr(mp.findroot(H, (10, 12), solver="bisect"), 15))
import numpy as np
ts=np.linspace(0.01,60,6000); sg=[mp.sign(H(mp.mpf(float(x)))) for x in ts]; print("sign changes on grid:", [float(ts[i]) for i in range(1,len(ts)) if sg[i]!=sg[i-1]])

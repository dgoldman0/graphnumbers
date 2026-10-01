import sympy as sp
import mpmath as mp
from core import lap
mp.mp.dps = 100
n = 5
G = [(0,1),(0,2),(0,3),(0,4),(1,2)]
cuts = [(0,1),(0,3),(0,4)]
# exact signed spectral measure: sum over S of (-1)^{3-|S|} * eigenvalue multiset
from collections import Counter
meas = Counter()
for mask in range(8):
    E = [e for e in G if e not in [cuts[i] for i in range(3) if mask>>i&1]]
    L = sp.Matrix(lap(n, E))
    for ev, mult in L.eigenvals().items():
        meas[sp.nsimplify(ev)] += (-1)**(3-bin(mask).count("1")) * mult
meas = {k: v for k, v in meas.items() if v != 0}
print("signed spectral measure:", sorted(((sp.N(k, 12), k, v) for k, v in meas.items()), key=lambda x: x[0]))
H = lambda t: sum(v * sp.exp(-t*k) for k, v in meas.items())
for t in [1, 3, 10, 12, 15, 20]:
    print(t, sp.N(H(sp.Integer(t)), 25))
f = sp.lambdify(sp.Symbol('t'), H(sp.Symbol('t')), 'mpmath')
print("roots:", mp.nstr(mp.findroot(f, 1.5), 20), mp.nstr(mp.findroot(f, 13), 20))

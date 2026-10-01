import mpmath as mp
from fractions import Fraction as Q
from math import factorial
from core import lap, brute_moments
mp.mp.dps = 80
n = 5
G = [(0,1),(0,2),(0,3),(0,4),(1,2)]
cuts = [(0,1),(0,3),(0,4)]
def heat_mp(t):
    tot = mp.mpf(0)
    for mask in range(8):
        E = [e for e in G if not (e in [cuts[i] for i in range(3) if mask>>i&1])]
        L = mp.matrix(lap(n, E))
        ev, _ = mp.eigsy(L)
        tr = sum(mp.e**(-t*x) for x in ev)
        tot += (-1)**(3-bin(mask).count("1")) * tr
    return tot
for t in [mp.mpf(1), mp.mpf(3)]:
    print("t=", t, mp.nstr(heat_mp(t), 30))
# exact Taylor with rigorous tail bound: |I_n| <= 8*5*8^n
N = 260
I = brute_moments(n, G, [(u,v,-1) for u,v in cuts], N)
for t in [Q(1), Q(3)]:
    s = sum(Q((-1)**j) * t**j * I[j] / factorial(j) for j in range(N+1))
    # tail: sum_{j>N} 40 (8t)^j / j! <= 40 (8t)^{N+1}/(N+1)! / (1 - 8t/(N+2))
    x = 8*t
    tail = 40 * x**(N+1) / factorial(N+1) / (1 - x/(N+2))
    print("t=", t, "exact partial:", mp.nstr(mp.mpf(s.numerator)/s.denominator, 30), " tail bound:", float(tail))
# find the crossing
f = lambda t: heat_mp(t)
root = mp.findroot(f, 2)
print("root:", mp.nstr(root, 25))
for t in [0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 20]:
    print(t, mp.nstr(heat_mp(mp.mpf(t)), 15))

import cmath, math, random
from powers import *
random.seed(11)
def char_value(T, r, thetas):
    tot = 0
    for cf, c in T.items():
        coords, _ = coordinates(adj_from_cf(cf), r)
        tot += c * cmath.exp(1j * sum(float(x) * th for x, th in zip(coords, thetas)))
    return tot
for r, nmax, m in ((2, 4, 6), (3, 3, 8)):
    Ts = [T_E_power(n, r, m) for n in range(nmax + 1)]
    for lam in (complex(5, 3), complex(-4*r, 0), complex(0.3, -0.1), 4*r*cmath.exp(0.7j)):
        rho, phi = abs(lam), cmath.phase(lam)
        a = rho / (4 * r)
        u = cmath.exp(1j*phi) * complex(a, math.sqrt(1 - a*a)); v = cmath.exp(1j*phi) * complex(-a, math.sqrt(1 - a*a))
        thetas = [cmath.phase(u)] * r + [cmath.phase(v)]
        vals = [char_value(T, r, thetas) for T in Ts]
        err = max(abs(vals[n] - lam**n) / max(1, abs(lam)**n) for n in range(nmax + 1))
        print(f'r={r} lambda={lam:.3f} chi(E)={vals[1]:.6f} max rel err over n<= {nmax}: {err:.2e}')
    # random phases (not tied to lambda): multiplicativity chi(E^n) = chi(E)^n
    for trial in range(3):
        thetas = [random.uniform(-3, 3) for _ in range(r + 1)]
        vals = [char_value(T, r, thetas) for T in Ts]
        print(f'r={r} random phases: max |chi(E^n)-chi(E)^n| = {max(abs(vals[n]-vals[1]**n) for n in range(nmax+1)):.2e}')

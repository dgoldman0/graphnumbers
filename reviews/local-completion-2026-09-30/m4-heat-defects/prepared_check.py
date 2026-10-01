import sys
sys.path.insert(0, '/home/user/graphnumbers/python/src')
import mpmath as mp
from fractions import Fraction as Q
from graphlocal.prepared import PreparedRelativeHeat
from graphlocal.interactions import CutInteraction, LineCutDefect
from graphlocal.defects import CutLineDefect, _relative_plan
mp.mp.dps=40
def M(x): return mp.mpf(x.numerator)/x.denominator
def h(t): t=M(t); return mp.e**(-2*t)*mp.besseli(0,2*t)
def e(t): t=M(t); return (1-mp.e**(-4*t))/2
def p(ell,t): t=M(t); return mp.fsum(mp.e**(-t*(2-2*mp.cos(mp.pi*a/ell))) for a in range(ell))
def J(ell,t): return p(ell,t)-ell*h(t)-e(t)
ok=True
for X,f in ((CutLineDefect(), e), (CutInteraction(3), lambda t: J(3,t))):
    P = PreparedRelativeHeat(X, Q(3), "1e-9")
    for t in [Q(0), Q(1,100), Q(1,7), Q(1,2), Q(1), Q(2), Q(5,2), Q(3)]:
        c = P.at(t)
        if not (M(c.interval.lower) <= f(t) <= M(c.interval.upper)) or c.interval.radius > Q(1,10**9): ok=False; print('fail', t)
# monotonicity of the generic step count in t for fixed parameters
steps=[_relative_plan(Q(k,20), 2, Q(4), Q(1,10**9), 256)[0] for k in range(0,81)]
print('prepared queries valid:', ok, '; step counts nondecreasing in t:', all(a<=b for a,b in zip(steps,steps[1:])))

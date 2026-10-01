import sys
sys.path.insert(0, '/home/user/graphnumbers/python/src')
import mpmath as mp
from fractions import Fraction as Q
from graphlocal import Finite, Line, heat_return, cycle, path, graph, complete
from graphlocal.elements import Element
from graphlocal.local import LocalApproximation, LocalHistogram
from graphlocal.defects import CutLineDefect, relative_heat
from graphlocal.interactions import CutInteraction
mp.mp.dps = 50
def M(x): return mp.mpf(x.numerator)/x.denominator if isinstance(x,Q) else mp.mpf(x)
def h(t): t=M(t); return mp.e**(-2*t)*mp.besseli(0,2*t)
def e(t): t=M(t); return (1-mp.e**(-4*t))/2
def inside(c, truth): return M(c.interval.lower) <= truth <= M(c.interval.upper)
class Perturbed(Element):
    """Adversarial oracle: returns exact histogram plus a signed perturbation of declared size."""
    def __init__(self, base, sign=1):
        self.base, self.sign = base, sign
        for f in ("degree_bound","variation_bound","edit_bound","mass","positive","moment_profile"):
            setattr(self, f, getattr(base, f))
    def approximate(self, radius, k=1, epsilon="0.000001"):
        eps = Q(epsilon)
        exact = self.base.local(radius)
        # move mass eps*0.9 / |B|^k onto the largest type, with chosen sign
        if not exact.values:
            return LocalApproximation(exact, k, Q(0))
        key = max(exact.values, key=lambda g: g.graph.n)
        delta = self.sign * eps * Q(9,10) / key.graph.n**k
        pert = LocalHistogram(radius, [(key.graph, delta)])
        return LocalApproximation(exact + pert, k, eps)
res=[]
for t in (Q(5), Q(10)):
    for X, truth in ((Line(), h(t)), (Line()*Line(), h(t)**2), (3*Line()-2, 3*h(t)-2)):
        c = heat_return(X, t, "1/1000", max_steps=400); res.append(('heat',t,inside(c,truth)))
    for X, truth in ((CutLineDefect(), e(t)), (CutLineDefect()*Finite.scalar(3), 3*e(t))):
        c = relative_heat(X, t, "1e-6", max_steps=400); res.append(('rel',t,inside(c,truth)))
for sgn in (1,-1):
    for t in (Q(1,3), Q(2)):
        c = heat_return(Perturbed(Line()), t, "1e-6"); res.append(('pert-heat', sgn, t, inside(c, h(t))))
        c = heat_return(Perturbed(2*Line()-Finite.from_graph(complete(4),True), sgn), t, "1e-6")
        truth = 2*h(t) - (1+3*mp.e**(-4*M(t)))/4
        res.append(('pert-signed', sgn, t, inside(c, truth)))
        c = relative_heat(Perturbed(CutLineDefect(), sgn), t, "1e-6"); res.append(('pert-rel', sgn, t, inside(c, e(t))))
print(all(r[-1] for r in res), [r for r in res if not r[-1]], len(res))

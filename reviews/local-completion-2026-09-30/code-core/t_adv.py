from fractions import Fraction as Q
import mpmath as mp
import networkx as nx
from common import *
from oracles import *
from graphlocal import *
SL = mp.mpf(10) ** -40

class Adv(Element):
    def __init__(self, value, sign, where):
        self.value, self.sign, self.where = value, sign, where
        for f in ("degree_bound", "variation_bound", "edit_bound", "mass", "positive"):
            setattr(self, f, getattr(value, f))
    def norm_bound(self, radius, k): return self.value.norm_bound(radius, k)
    def approximate(self, radius, k=1, epsilon="1e-6"):
        eta = Q(epsilon)
        h = self.value.local(radius)
        if self.where == "K1" or not h.values:
            pert = LocalHistogram(radius, [(graph(1), self.sign * eta)])
        else:
            key = max(h.values, key=lambda key: key.graph.n)
            pert = LocalHistogram(radius, [(key.graph, self.sign * eta / key.graph.n ** k)])
        return LocalApproximation(h + pert, k, eta)

bad = 0; n = 0
for t in [Q(0), Q(1, 5), Q(1), Q(3)]:
    tt = mpq(t)
    for eps in ["1e-3", "1e-9"]:
        for sign in (1, -1):
            for where in ("K1", "big"):
                for X, truth in [(Line(), hline(tt)), (Line() * Line(), hline(tt) ** 2),
                                 (Finite.from_graph(cycle(5), True), heat_trace(nx.cycle_graph(5), tt) / 5),
                                 (Line() * 2 - Finite.from_graph(path(3)), 2 * hline(tt) - heat_trace(nx.path_graph(3), tt))]:
                    c = heat_return(Adv(X, sign, where), t, eps, max_steps=600)
                    n += 1
                    if not (mpq(c.interval.lower) - SL <= truth <= mpq(c.interval.upper) + SL):
                        bad += 1; print("FAIL heat", t, eps, sign, where)
                for X, truth in [(CutLineDefect(), cuts_value((0,), tt)), (CutLineDefect() * Line(), cuts_value((0,), tt) * hline(tt)),
                                 (CutInteraction(2), cuts_value((0, 2), tt) - 2 * cuts_value((0,), tt))]:
                    c = relative_heat(Adv(X, sign, where), t, eps, max_steps=600)
                    n += 1
                    if not (mpq(c.interval.lower) - SL <= truth <= mpq(c.interval.upper) + SL):
                        bad += 1; print("FAIL rel", t, eps, sign, where)
                    pr = PreparedRelativeHeat(Adv(X, sign, where), max(t, Q(1)), eps, max_steps=600)
                    c = pr.evaluate(t); n += 1
                    if not (mpq(c.interval.lower) - SL <= truth <= mpq(c.interval.upper) + SL):
                        bad += 1; print("FAIL prepared", t, eps, sign, where)
print("adversarial checks", n, "bad", bad)

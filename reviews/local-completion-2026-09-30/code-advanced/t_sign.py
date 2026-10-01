import mpmath
from fractions import Fraction as Q
from graphlocal import *
import oracle
g = graph(5, [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)])
edits = [(0, 1, -1), (0, 3, -1), (0, 4, -1)]
x = EdgeInteraction(g, edits)
print("bridge reduction", x.bridge_reduction_applies, x.active_edits, x.reduction_sign)
for t in (Q(1), Q(3)):
    true = oracle.heat(g.rows, edits, t)
    c = controlled_heat(x, t, "1e-10")
    print(t, mpmath.nstr(true, 20), float(c.interval.lower), float(c.interval.upper), c.interval.lower <= Q(str(mpmath.nstr(true, 30))) <= c.interval.upper)

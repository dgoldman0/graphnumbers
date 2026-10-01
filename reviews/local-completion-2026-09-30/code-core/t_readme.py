from fractions import Fraction
from graphlocal import Finite, Line, cycle, exp, heat_return, path, reconstruct
H = Finite.from_graph(path(2), normalize=True)
assert (H * H).finite() == Finite.from_graph(cycle(4), normalize=True)
L = Line(); grid = L * L
local = grid.approximate(radius=2, k=1, epsilon="1e-8")
assert local.histogram.norm(1) == 13 and local.error == 0
series = exp(H / 10).approximate(radius=1, k=1, epsilon="1e-8")
assert series.error <= Fraction("1e-8")
result = reconstruct(L.local(2), [path(4), path(5)])
assert result.coefficients == (Fraction(-1), Fraction(1)) and result.cost == 9
heat = heat_return(grid, time="1/2", epsilon="1e-8")
print(float(heat.interval.midpoint))
from graphlocal import CutLineDefect, SparseEdgeDifference, relative_heat
cut_cycle = SparseEdgeDifference(cycle(128), [(0, 127, -1)])
print(float(relative_heat(cut_cycle, "1/2").interval.midpoint))
E = CutLineDefect(); assert E.local(5).norm(0) == 20
print(float(relative_heat(E, "1/2").interval.midpoint))
print(float(relative_heat(E * Line(), "1/2").interval.midpoint))
from graphlocal import CutInteraction, LineCutDefect, PreparedLocal, PreparedRelativeHeat, connected_cut_interaction
cuts = LineCutDefect([0, 2, 7])
geometry = PreparedLocal(cuts, radius=16)
responses = PreparedRelativeHeat(geometry, max_time=2, epsilon="1e-8")
for t in ("1/4", "1/2", "1", "2"):
    print(t, responses.evaluate(t).interval.to_data()["midpoint"][:30])
connected = connected_cut_interaction([0, 2, 7])
assert connected.local(4) == (-CutInteraction(7)).local(4)
print("README snippets OK")

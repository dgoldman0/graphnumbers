"""Small exact algebra, approximation, reconstruction and heat calculations."""
from fractions import Fraction

from graphlocal import (CutLineDefect, Finite, Line, cycle, exp, heat_return,
                       path, reconstruct, relative_heat)

H = Finite.from_graph(path(2), normalize=True)
assert (H * H).finite() == Finite.from_graph(cycle(4), normalize=True)

L = Line()
local = L.approximate(radius=2, k=1, epsilon="1e-8")
print("Line radius-two seminorm:", local.histogram.norm(1))

result = reconstruct(local.histogram, [path(4), path(5)])
print("Exact finite representative:", result.coefficients, "on [P4, P5]")
print("Certified minimum coefficient mass on this catalog:", result.cost)

series = exp(H / 10).approximate(radius=1, k=1, epsilon="1e-8")
print("Algebra exponential local error bound:", float(series.error))

heat = heat_return(L * L, time=Fraction(1, 2), epsilon="1e-8")
print("Square-lattice heat certificate midpoint:", float(heat.interval.midpoint))
print("Absolute error bound:", float(heat.interval.radius))
print("Required radius:", heat.radius)

# A signed limit beyond finite global measures: cutting one edge of the line.
cut = CutLineDefect()
assert cut.local(3).norm(0) == 12
correction = relative_heat(cut * L, time="1/2", epsilon="1e-8")
print("Planar-cut heat correction per transverse volume:",
      float(correction.interval.midpoint))
print("Absolute error bound:", float(correction.interval.radius))

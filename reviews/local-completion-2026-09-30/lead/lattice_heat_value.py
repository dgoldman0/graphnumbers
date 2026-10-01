"""Square-lattice heat return at t=1/2: closed form versus the library interval.

For the combinatorial Laplacian on Z^2 the return value is (e^{-1} I_0(1))^2.
Run from the repository root with PYTHONPATH=python/src plus mpmath.
"""
import mpmath

from graphlocal import Line, heat_return

mpmath.mp.dps = 30
exact = (mpmath.e ** -1 * mpmath.besseli(0, 1)) ** 2
certificate = heat_return(Line() * Line(), time="1/2", epsilon="1e-8")
lower, upper = certificate.interval.lower, certificate.interval.upper
midpoint = (lower + upper) / 2
print("closed form (e^-1 I0(1))^2 :", mpmath.nstr(exact, 15))
print("certified interval         :", float(lower), float(upper))
print("interval contains value    :", mpmath.mpf(lower.numerator) / lower.denominator <= exact
      <= mpmath.mpf(upper.numerator) / upper.denominator)
print("midpoint quoted in README  :", float(midpoint))
print("midpoint - closed form     :", mpmath.nstr(mpmath.mpf(midpoint.numerator) / midpoint.denominator - exact, 5))

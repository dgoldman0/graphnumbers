import sympy as sp
from fractions import Fraction as Q

t, w, u, z, W = sp.symbols("t w u z W")
# NONSPECTRAL_CALCULUS (24)-(25): reciprocal motif response for U_t=1+tX, X has mu_{c4}=delta_2-delta_0
F_U = 1 + t * (w**2 - 1)
Finv = 1 / F_U
# moments M_q = ((w d/dw)^q Finv)(1)
def moment(Fexpr, q):
    e = Fexpr
    for _ in range(q):
        e = w * sp.diff(e, w)
    return sp.simplify(e.subs(w, 1))
print("M1..M3 of U_t^{-1}:", [sp.expand(moment(Finv, q)) for q in (1, 2, 3)])
print("expected:", [-2*t, 8*t**2 - 4*t, -48*t**3 + 48*t**2 - 8*t])
# distribution coefficients vs claimed formula
ser = sp.series(Finv, w, 0, 9).removeO()
claimed = sum(1 / (1 - t) * (-t / (1 - t))**n * w**(2 * n) for n in range(5))
print("series matches claimed mu_{c4}(U_t^{-1})?", sp.simplify(sp.expand(ser - claimed)) == 0)
# check consistent with actual inverse sum_{a,b} C(a+b,a)(-t)^a t^b R^a S^b: c4 on R^aS^b is 2a
actual = 0
for a in range(5):
    actual += (-t)**a * w**(2*a) * sum(sp.binomial(a + b, a) * t**b for b in range(40))
diffc = sp.series(sp.expand(actual) - claimed, w, 0, 9).removeO()
print("actual inverse c4 law vs claimed (coefficient differences, as series in t to O(t^30)):",
      [sp.series(sp.expand(diffc).coeff(w, 2*n), t, 0, 30).removeO() for n in range(5)])
# cumulants of e^{tX}: log F = t(w^2-1) -> kappa_q = t * 2^q; verify via K(s) = log F(e^s)
s = sp.symbols("s")
K = t * (sp.exp(2 * s) - 1)
print("kappa_q(e^{tX}) q=1..4:", [sp.diff(K, s, q).subs(s, 0) for q in (1, 2, 3, 4)])
# (15) one-statistic reciprocal moments from jets
v, m1, m2 = sp.symbols("v m1 m2")
jet = v + m1 * s + m2 * s**2 / 2
inv = sp.series(1 / jet, s, 0, 3).removeO()
print("M1(X^-1) =", sp.simplify(inv.coeff(s, 1)), "  M2(X^-1) =", sp.simplify(2 * inv.coeff(s, 2)))
# (18) covariance cumulant check
t1, t2 = sp.symbols("t1 t2")
M = sp.symbols("M10 M01 M20 M11 M02")
V = sp.symbols("V")
mgf = 1 + (M[0]*t1 + M[1]*t2 + M[2]*t1**2/2 + M[3]*t1*t2 + M[4]*t2**2/2) / V
logm = sp.expand(sp.series(sp.log(mgf).subs({t1: s*t1, t2: s*t2}), s, 0, 3).removeO().subs(s, 1))
print("kappa_{11} =", sp.simplify(logm.coeff(t1, 1).coeff(t2, 1)))

# GEOMETRIC_ARITHMETIC (31): F_m(q) = sum n^m q^n = sum_j S(m,j) j! q^j/(1-q)^{j+1}
q = sp.symbols("q")
from sympy.functions.combinatorial.numbers import stirling
for m in range(0, 6):
    Fm = sum(stirling(m, j) * sp.factorial(j) * q**j / (1 - q)**(j + 1) for j in range(m + 1))
    direct = sum((n**m if n > 0 else (1 if m == 0 else 0)) * Q(1, 3)**n for n in range(400))
    print("F_%d(1/3): formula" % m, float(Fm.subs(q, sp.Rational(1, 3))), "direct", float(direct))
# (32) exact tail sum_{n>N} (1+nD)^m q^n
for (m, D, N, qq) in [(2, 3, 4, sp.Rational(1, 2)), (3, 6, 2, sp.Rational(1, 5)), (1, 1, 0, sp.Rational(2, 3))]:
    formula = qq**(N + 1) * sum(sp.binomial(m, j) * (1 + D * (N + 1))**(m - j) * D**j *
                               sum(stirling(j, i) * sp.factorial(i) * qq**i / (1 - qq)**(i + 1) for i in range(j + 1))
                               for j in range(m + 1))
    direct = sum((1 + n * D)**m * qq**n for n in range(N + 1, 600))
    print("tail (32) m=%d D=%d N=%d q=%s:" % (m, D, N, qq), sp.nsimplify(formula), float(formula), float(direct))

# GEOMETRIC_ARITHMETIC Section 4 unit example
x = 1 + z**2 / 4
y = 1 - z**2 / 2 + sp.Rational(3, 4) * W
print("D(x)=", sp.expand(x.subs(z, u)), " D(y)=", sp.expand(y.subs({z: u, W: u**2})))
print("y(1,-2/3) =", y.subs({z: 1, W: sp.Rational(-2, 3)}), "  x-y =", sp.factor(x - y))
# (19) decomposition example: f(z1,z2,z3) with d2=2,d3=3
z1, z2, z3, sv = sp.symbols("z1 z2 z3 sv")
f = (z2 - z1**2) * (3 + z1 * z3) + (z3 - z1**3) * z2**2 + z1 * z2 * z3 - z1**6
print("D(f) =", sp.expand(f.subs({z2: z1**2, z3: z1**3})))
g2 = sp.integrate(sp.diff(f, z2).subs(z2, z1**2 + sv * (z2 - z1**2)), (sv, 0, 1))
f1 = f.subs(z2, z1**2)
g3 = sp.integrate(sp.diff(f1, z3).subs(z3, z1**3 + sv * (z3 - z1**3)), (sv, 0, 1))
print("f == (z2-z1^2)g2 + (z3-z1^3)g3 ?", sp.expand(f - (z2 - z1**2) * g2 - (z3 - z1**3) * g3) == 0)

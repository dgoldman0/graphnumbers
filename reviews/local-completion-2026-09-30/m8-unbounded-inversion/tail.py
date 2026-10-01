import random
from fractions import Fraction as Fr
from math import factorial
from mpmath import mp, mpf, nsum, inf, factorial as mfac
mp.dps = 50
random.seed(3)
worst = 0; tested = 0; viol = 0
for trial in range(400):
    r = random.randint(1, 4); k = random.randint(1, 3); t = Fr(random.randint(1, 40), random.choice([4, 10, 50, 100]))
    a = 4 * r * t; m = r * k
    # smallest N with q_N < 1
    N = 0
    while not (N + 2 > a * (1 + Fr(2, 2*N + 3)) ** m): N += 1
    N += random.randint(0, 3)
    qN = a / (N + 2) * (1 + Fr(2, 2*N + 3)) ** m
    bound = Fr((r + 1) ** k) * a ** (N + 1) * (2*N + 3) ** m / (factorial(N + 1) * (1 - qN))
    A = mpf(a.numerator) / a.denominator
    true = (r + 1) ** k * nsum(lambda n: (1 + 2*n) ** m * A ** n / mfac(n), [N + 1, inf])
    tested += 1
    if true > mpf(bound.numerator) / bound.denominator * (1 + mpf(10) ** -30): viol += 1
    worst = max(worst, float(true / (mpf(bound.numerator) / bound.denominator)))
print('tested', tested, 'violations', viol, 'max true/bound', worst)

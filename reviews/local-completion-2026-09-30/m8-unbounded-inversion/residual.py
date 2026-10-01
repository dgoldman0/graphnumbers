# Random test of residual certificate (4), refinement (10) and comparison (11)
# in the weighted degree algebra l^1(N, (n+1)^k) (the r=1 star subalgebra).
import random
from mpmath import mp, mpf
mp.dps = 40
random.seed(2)
D = 400
def mul(a, b):
    c = [mpf(0)] * D
    for i, x in enumerate(a):
        if x == 0: continue
        for j, y in enumerate(b):
            if i + j >= D: break
            c[i + j] += x * y
    return c
def norm(a, k): return sum(abs(x) * (n + 1) ** k for n, x in enumerate(a))
def inv(a):  # formal reciprocal, truncated
    b = [mpf(0)] * D; b[0] = 1 / a[0]
    for n in range(1, D):
        b[n] = -sum(a[i] * b[n - i] for i in range(1, min(n, len(a) - 1) + 1)) / a[0]
    return b
def pad(a): return [mpf(x) for x in a] + [mpf(0)] * (D - len(a))
viol = {'4a': 0, '4b': 0, '10': 0, '11': 0}; cnt = 0
for trial in range(300):
    k = random.randint(1, 2)
    a = pad([1] + [random.uniform(-0.3, 0.3) / (n + 1) ** 2 for n in range(1, 4)])
    # check true invertibility: roots outside unit disk
    from mpmath import polyroots
    if min(abs(z) for z in polyroots(list(reversed(a[:4])))) <= 1.01: continue
    ainv = inv(a)
    if abs(ainv[-1]) * D ** k > 1e-20: continue
    a0 = a[:]; pert = [random.uniform(-1, 1) * 1e-3 for _ in range(4)]
    for i in range(4): a0[i] += pert[i]
    delta = norm([x - y for x, y in zip(a, a0)], k)
    b = pad([1] + [random.uniform(-0.4, 0.4) / (n + 1) ** 2 for n in range(1, 5)])
    B = norm(b, k); e0 = [-x for x in mul(a0, b)]; e0[0] += 1
    q = norm(e0, k) + delta * B
    if q >= 1: continue
    cnt += 1
    tinv = norm(ainv, k); terr = norm([x - y for x, y in zip(ainv, b)], k)
    if tinv > B / (1 - q) * (1 + 1e-25): viol['4a'] += 1
    if terr > B * q / (1 - q) * (1 + 1e-25): viol['4b'] += 1
    # (11) with y_N = b*sum_{j<=N} e0^j
    q0 = norm(e0, k)
    N = random.randint(0, 6); y = b[:]; term = b[:]
    for _ in range(N): term = mul(term, e0); y = [s + t for s, t in zip(y, term)]
    lhs = norm([x - z for x, z in zip(ainv, y)], k)
    rhs = delta * B * B / ((1 - q) * (1 - q0)) + B * q0 ** (N + 1) / (1 - q0)
    if lhs > rhs * (1 + 1e-25): viol['11'] += 1
    # (10): refined source a1 with delta1 per (6), b_N = b*sum e1^j, N chosen by M alpha^{N+1} <= eps/2
    eps = mpf(10) ** -random.randint(3, 8); M = B / (1 - q); alpha = (1 + q) / 2
    d1 = min((1 - q) / (4 * B), eps / (4 * M * M))
    a1 = a[:]; s = d1 / sum((n + 1) ** k for n in range(4)) * mpf('0.999')
    for i in range(4): a1[i] += random.choice([-1, 1]) * s
    e1 = [-x for x in mul(a1, b)]; e1[0] += 1
    NN = 0
    while M * alpha ** (NN + 1) > eps / 2: NN += 1
    y = b[:]; term = b[:]
    for _ in range(NN): term = mul(term, e1); y = [s_ + t for s_, t in zip(y, term)]
    if norm([x - z for x, z in zip(ainv, y)], k) > eps: viol['10'] += 1
print('instances', cnt, 'violations', viol)

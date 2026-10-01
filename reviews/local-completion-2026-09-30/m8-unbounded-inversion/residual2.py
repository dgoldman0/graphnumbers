# Random test of (4), (10), (11) of EFFECTIVE_LOCAL_INVERSION in l^1(N,(n+1)^k) (the r=1 star subalgebra).
import random
import numpy as np
random.seed(2); D = 300
def mul(a, b): return np.convolve(a, b)[:D]
def norm(a, k): return float(np.sum(np.abs(a) * (np.arange(D) + 1.0) ** k))
def inv(a):
    b = np.zeros(D); b[0] = 1 / a[0]
    for n in range(1, D):
        b[n] = -sum(a[i] * b[n - i] for i in range(1, min(n, 3) + 1)) / a[0]
    return b
def pad(x): v = np.zeros(D); v[:len(x)] = x; return v
viol = {'4a': 0, '4b': 0, '10': 0, '11': 0}; cnt = 0; tight = {'4b': 0, '11': 0, '10': 0}
for trial in range(4000):
    k = random.randint(1, 2)
    a = pad([1] + [random.uniform(-0.4, 0.4) / (n + 1) ** 2 for n in range(1, 4)])
    if min(abs(np.roots(a[:4][::-1]))) <= 1.05: continue
    ainv = inv(a)
    if abs(ainv[-1]) * D ** k > 1e-14: continue
    a0 = a.copy(); a0[:4] += [random.uniform(-1, 1) * 1e-3 for _ in range(4)]
    delta = norm(a - a0, k)
    b = pad([1] + [random.uniform(-0.5, 0.5) / (n + 1) ** 2 for n in range(1, 5)])
    B = norm(b, k); e0 = -mul(a0, b); e0[0] += 1; q0 = norm(e0, k); q = q0 + delta * B
    if q >= 1: continue
    cnt += 1
    if norm(ainv, k) > B / (1 - q) * (1 + 1e-9): viol['4a'] += 1
    lhs = norm(ainv - b, k); rhs = B * q / (1 - q)
    if lhs > rhs * (1 + 1e-9): viol['4b'] += 1
    tight['4b'] = max(tight['4b'], lhs / rhs)
    N = random.randint(0, 8); y = b.copy(); term = b.copy()
    for _ in range(N): term = mul(term, e0); y = y + term
    lhs = norm(ainv - y, k); rhs = delta * B * B / ((1 - q) * (1 - q0)) + B * q0 ** (N + 1) / (1 - q0)
    if lhs > rhs * (1 + 1e-9): viol['11'] += 1
    tight['11'] = max(tight['11'], lhs / rhs)
    eps = 10.0 ** -random.randint(2, 7); M = B / (1 - q); alpha = (1 + q) / 2
    d1 = min((1 - q) / (4 * B), eps / (4 * M * M))
    a1 = a.copy(); s = d1 / sum((n + 1) ** k for n in range(4)) * 0.999
    a1[:4] += [random.choice([-1, 1]) * s for _ in range(4)]
    e1 = -mul(a1, b); e1[0] += 1
    NN = 0
    while M * alpha ** (NN + 1) > eps / 2: NN += 1
    y = b.copy(); term = b.copy()
    for _ in range(NN): term = mul(term, e1); y = y + term
    lhs = norm(ainv - y, k)
    if lhs > eps * (1 + 1e-9): viol['10'] += 1
    tight['10'] = max(tight['10'], lhs / eps)
print('instances', cnt, 'violations', viol, 'max lhs/rhs', tight)

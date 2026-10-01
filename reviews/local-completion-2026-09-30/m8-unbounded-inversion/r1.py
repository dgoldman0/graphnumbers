from mpmath import mp, mpf, exp, polyroots
mp.dps = 40
# T_1 exp(tE) = exp(2t(X - X^2)) in the degree variable X (X^n = star with n leaves)
def l1_exp(t, D=400):
    t = mpf(t)
    # coefficients of exp(g) with g = 2t X - 2t X^2 via recurrence n c_n = sum k g_k c_{n-k}
    g = {1: 2*t, 2: -2*t}
    c = [mpf(1)]
    for n in range(1, D):
        c.append(sum(k*g[k]*c[n-k] for k in (1, 2) if n-k >= 0)/n)
    return sum(abs(x) for x in c)
for t in ('0.001', '0.25', '-0.25', '1', '-1', '3'):
    v = l1_exp(t); print('t', t, '||T_1 exp(tE)||_1 =', mp.nstr(v, 12), ' e^{4|t|} =', mp.nstr(exp(4*abs(mpf(t))), 12))
# r=1 resolvent: 1 - t T_1E = 1 - 2t X + 2t X^2 is invertible in l1(M_1) iff roots outside closed unit disk
for t in ('0.24', '0.26', '0.49', '0.5', '-0.24', '-0.25', '-0.26'):
    t = mpf(t); roots = polyroots([2*t, -2*t, 1]); print('t', mp.nstr(t, 3), 'min |root| =', mp.nstr(min(abs(z) for z in roots), 10))

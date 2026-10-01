import mpmath as mp
exec(open('sign_late2.py').read().split('print("roots:"')[0].replace('for t in [1, 3, 10, 12, 15, 20, 40]:\n    print(t, mp.nstr(H(mp.mpf(t)), 20))\n',''))
def bis(a, b, it=200):
    a, b = mp.mpf(a), mp.mpf(b); fa = H(a)
    for _ in range(it):
        m = (a+b)/2; fm = H(m)
        if mp.sign(fm) == mp.sign(fa): a, fa = m, fm
        else: b = m
    return a
print("root1", mp.nstr(bis(1, 3), 15), "root2", mp.nstr(bis(10, 12), 15))
sg_prev = None; changes = []
for i in range(1, 12001):
    t = mp.mpf(i)/200
    s = mp.sign(H(t))
    if sg_prev is not None and s != sg_prev: changes.append(float(t))
    sg_prev = s
print("sign changes on grid (0,60]:", changes)

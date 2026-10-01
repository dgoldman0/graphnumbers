import numpy as np
# f(z) = exp(-(1-z)^{-1/2}) on the unit circle, principal branch (positive for real z<1)
for M in (2**16, 2**20):
    th = 2 * np.pi * np.arange(M) / M
    zz = np.exp(1j * th)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 - zz) ** (-0.5)
        f = np.exp(-g)
    f[0] = 0.0
    c = np.fft.fft(f) / M          # c_n for n = 0..M-1 (aliased)
    n = np.arange(M)
    half = M // 2
    cn = c[:half]
    print(f"M={M}: c0={cn[0].real:.15f} (e^-1={np.exp(-1):.15f}), max|Im c_n|={np.max(np.abs(cn.imag)):.2e}")
    print("   |c_n| at n=10,100,1000,10000:", [f"{abs(cn[k]):.3e}" for k in (10, 100, 1000, 10000) if k < half])
    print("   aliased tail |c_n| for n in upper half (should be ~0):", f"{np.max(np.abs(c[half:half + 1000])):.2e}")
    for q in range(5):
        print(f"   sum n^{q} c_n = {np.sum((n[:half].astype(float) ** q) * cn.real):.3e}")

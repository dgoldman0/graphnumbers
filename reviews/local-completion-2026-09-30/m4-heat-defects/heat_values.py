import mpmath as mp
mp.mp.dps = 50
def h(t):  # combinatorial Laplacian on Z, unit rate per edge: e^{-2t} I_0(2t)
    t = mp.mpf(t); return mp.e**(-2*t)*mp.besseli(0, 2*t)
def h_int(t):
    t = mp.mpf(t); return mp.quad(lambda th: mp.e**(-2*t*(1-mp.cos(th))), [0, mp.pi])/mp.pi
for t in ['0.1','0.5','1','2']:
    print('t=',t,' line', mp.nstr(h(t),20), ' check integral', mp.nstr(h_int(t),20), ' grid', mp.nstr(h(t)**2,20), ' cubic', mp.nstr(h(t)**3,15))
# library certified interval for grid at t=1/2 (from heat_benchmark.json torus/limit)
mid = mp.mpf('0.2169320139861297'); rad = mp.mpf('1.937864073633431e-9')
true = h('0.5')**2
print('true grid t=1/2:', mp.nstr(true, 15), ' lib midpoint', mid, ' inside:', mid-rad <= true <= mid+rad, ' distance to lower endpoint', mp.nstr(true-(mid-rad),5))
# Normalization alternatives for comparison
t=mp.mpf('0.5')
print('if normalized Laplacian (rate 1/deg) on Z: e^{-t} I0(t) =', mp.nstr(mp.e**(-t)*mp.besseli(0,t),15), 'squared grid (rate 1/4):', mp.nstr((mp.e**(-t/2)*mp.besseli(0,t/2))**2,15))
# torus C24 x C24 exact
import itertools
def cyc(n,t):
    return mp.fsum(mp.e**(-t*(2-2*mp.cos(2*mp.pi*k/n))) for k in range(n))/n
print('C24 x C24 normalized trace at 1/2:', mp.nstr(cyc(24,t)**2, 20))
print('C64 normalized trace at 1/2 :', mp.nstr(cyc(64,t),20), ' line:', mp.nstr(h(t),20))

import numpy as np, networkx as nx, mpmath as mp, random, itertools
from fractions import Fraction as F
mp.mp.dps = 30
def L(G): return nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(float)
def trh(G,t): return float(np.exp(-t*np.linalg.eigvalsh(L(G))).sum())
# (B) path / doubled cycle identity
mx=0
for n in range(2, 30):
    for t in (0.1, 0.7, 2.0, 5.0):
        lhs = trh(nx.path_graph(n), t); rhs = (trh(nx.cycle_graph(2*n), t) + 1 - np.exp(-4*t))/2
        mx = max(mx, abs(lhs-rhs))
print("(B) max |tr e^{-tL_Pn} - [tr e^{-tL_C2n}+1-e^{-4t}]/2| =", mx)
# (F) d_j(E) under P=I-L/D, exact, for D=2,3,4 via large finite P_n - C_n
def trP(G, D, jmax):
    nodes=sorted(G.nodes()); A=nx.to_numpy_array(G,nodelist=nodes,dtype=int); deg=A.sum(1)
    Mx = (np.diag(D-deg)+A).astype(object); P=np.identity(len(nodes),dtype=object); out=[]
    for j in range(jmax+1):
        out.append(F(int(np.trace(P)), D**j)); P=P.dot(Mx)
    return out
ok=True
for D in (2,3,4,7):
    n=40; a=trP(nx.path_graph(n),D,14); c=trP(nx.cycle_graph(n),D,14)
    for j in range(15):
        if a[j]-c[j] != (1-(1-F(4,D))**j)/2: ok=False; print("d_j(E) mismatch",D,j,a[j]-c[j])
print("(F) d_j(E)=[1-(1-4/D)^j]/2 exact for D in 2,3,4,7, j<=14:", ok)
# (I) product moment identity d_j(XY)=sum binom a^i b^(j-i) d_i(X) m_(j-i)(Y), X=P_n-C_n (D1=2), Y=U(H)
def cart(G,H): return nx.cartesian_product(G,H)
ok=True
for H in (nx.path_graph(2), nx.path_graph(3), nx.complete_graph(4), nx.star_graph(3)):
    D1=2; D2=max(dict(H.degree()).values()); D=D1+D2; n=12; jmax=8
    dX=[x-y for x,y in zip(trP(nx.path_graph(n),D1,jmax), trP(nx.cycle_graph(n),D1,jmax))]
    mY=[x/H.number_of_nodes() for x in trP(H,D2,jmax)]
    dXY=[(x-y)/H.number_of_nodes() for x,y in zip(trP(cart(nx.path_graph(n),H),D,jmax), trP(cart(nx.cycle_graph(n),H),D,jmax))]
    a,b=F(D1,D),F(D2,D)
    for j in range(jmax+1):
        rhs=sum(F(int(np.math.comb(j,i)) if hasattr(np,'math') else __import__('math').comb(j,i))*a**i*b**(j-i)*dX[i]*mY[j-i] for i in range(j+1))
        if rhs!=dXY[j]: ok=False; print("product moment mismatch", j)
        if abs(dXY[j]) > F(2*j, D): ok=False; print("bound 2qCj/D violated", j, dXY[j])
print("(I) binomial product-moment identity and |d_j(XY)|<=2qCj/D:", ok)
# (D) trace-norm moment bound |tr(P'^j-P^j)|<=2mj/D, random graphs and mixed edits
rng=random.Random(3); worst=0
for trial in range(200):
    n=rng.randint(4,12); G=nx.gnp_random_graph(n,0.4,seed=rng.randint(0,10**6)); G2=G.copy()
    m=rng.randint(1,3)
    for _ in range(m):
        u,v=rng.sample(range(n),2)
        if G2.has_edge(u,v): G2.remove_edge(u,v)
        else: G2.add_edge(u,v)
    medits=sum(1 for u in range(n) for v in range(u+1,n) if G.has_edge(u,v)!=G2.has_edge(u,v))
    if medits==0: continue
    D=max(max(dict(G.degree()).values()), max(dict(G2.degree()).values()), 1)
    P=np.eye(n)-L(G)/D; P2=np.eye(n)-L(G2)/D
    for j in range(1,15):
        ratio=abs(np.trace(np.linalg.matrix_power(P2,j))-np.trace(np.linalg.matrix_power(P,j)))/(2*medits*j/D)
        worst=max(worst,ratio)
print("(D) max ratio |tr(P'^j-P^j)|/(2mj/D) over random cases:", worst)
# (C) Krylov trace exactness: degree 2m exact, 2m+1 not in general
rng=np.random.default_rng(5); exact_ok=True; fails_at_2m1=0; tot=0
for trial in range(40):
    n=int(rng.integers(12,30)); G=nx.gnp_random_graph(n,0.25,seed=int(rng.integers(10**6)))
    G2=G.copy(); edges=list(G.edges())
    if len(edges)<3: continue
    e1=edges[int(rng.integers(len(edges)))]; G2.remove_edge(*e1)
    u,v=[int(x) for x in rng.choice(n,2,replace=False)]
    if G2.has_edge(u,v) or {u,v}=={*e1}: continue
    G2.add_edge(u,v)
    A=L(G); A2=L(G2); Bm=np.zeros((n,2)); Bm[e1[0],0]=1;Bm[e1[1],0]=-1;Bm[u,1]=1;Bm[v,1]=-1
    for m in (1,2,3):
        K=[Bm]; 
        for _ in range(m-1): K.append(A@K[-1])
        Kmat=np.hstack(K); U,s,_=np.linalg.svd(Kmat,full_matrices=False); U=U[:,s>1e-10*s[0]]
        Ac=U.T@A@U; A2c=U.T@A2@U
        for j in range(0,2*m+2):
            full=np.trace(np.linalg.matrix_power(A2,j))-np.trace(np.linalg.matrix_power(A,j))
            comp=np.trace(np.linalg.matrix_power(A2c,j))-np.trace(np.linalg.matrix_power(Ac,j))
            scale=max(1,abs(full))
            if j<=2*m and abs(full-comp)>1e-7*scale: exact_ok=False; print("Krylov not exact",m,j,full,comp)
            if j==2*m+1:
                tot+=1
                if abs(full-comp)>1e-6*scale: fails_at_2m1+=1
print("(C) Krylov compressed relative traces exact through degree 2m:", exact_ok, "; differs at 2m+1 in", fails_at_2m1, "of", tot)
# (E) separation additivity: dist_W(S_i,S_j)>2R suffices; check also =2R can fail
def balltype_key(G,o,R):
    d=nx.single_source_shortest_path_length(G,o,cutoff=R); B=G.subgraph(d).copy(); nx.set_node_attributes(B,{v:(v==o) for v in B},'root'); return B
def same(B1,B2): return nx.is_isomorphic(B1,B2,node_match=lambda a,b:a['root']==b['root'])
def Tdiff(G1,G0,R):  # list of (ball, coeff) per root difference
    out=[]
    for v in G0.nodes():
        b1,b0=balltype_key(G1,v,R),balltype_key(G0,v,R)
        if not same(b1,b0): out += [(b1,1),(b0,-1)]
    return out
def canon(items):
    groups=[]
    for B,c in items:
        for g in groups:
            if same(g[0],B): g[1]+=c; break
        else: groups.append([B,c])
    return [g for g in groups if g[1]!=0]
def eq_hist(a,b):
    a=canon(a); b=canon(b)
    bb=[[B,-c] for B,c in b]
    return canon([(B,c) for B,c in a]+[(B,c) for B,c in bb])==[]
R=2; n=30; G0=nx.cycle_graph(n)
for sep in (2*R, 2*R+1):
    Ga=G0.copy(); Ga.remove_edge(0,1)          # group 1 S1={0,1}
    j=1+sep; Gb=G0.copy(); Gb.remove_edge(j,j+1)  # group 2 S2={j,j+1}, dist(S1,S2)=sep
    Gab=G0.copy(); Gab.remove_edge(0,1); Gab.remove_edge(j,j+1)
    W=G0
    dist=min(nx.shortest_path_length(W,a,b) for a in (0,1) for b in (j,j+1))
    print(f"(E) R={R}, dist_W={dist}: additive at radius R?", eq_hist(Tdiff(Gab,G0,R), Tdiff(Ga,G0,R)+Tdiff(Gb,G0,R)))
# (H) obstruction: H_t(U(C_n)-U(C_2n))>0 and T_R=0 for n>=2R+2
for n in (6,8,12):
    for t in (0.1,1,5):
        val = mp.fsum(mp.e**(-t*(2-2*mp.cos(2*mp.pi*k/n))) for k in range(n))/n - mp.fsum(mp.e**(-t*(2-2*mp.cos(2*mp.pi*k/(2*n)))) for k in range(2*n))/(2*n)
        print(f"(H) n={n} t={t}: H_t(U(C_n)-U(C_2n)) = {mp.nstr(val,6)}", end='; ')
    print()

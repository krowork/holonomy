"""Newton-Kantorovich graduado con constantes explícitas para n = 6, 9 (9.25)."""
import sys, pickle; sys.path.insert(0,'/home/alex/fisica/holonomy/scripts')
import numpy as np, mpmath as mp
mp.mp.dps=60
from k1_racional import base
import newton_graduado as NG
import os
S = os.environ.get('DATOS', '.') + '/'
B=base(); W=NG.W
sc=[mp.mpf(1) if B[w][0]=='d' else 1/mp.sqrt(2) for w in W]      # raw -> Hilbert-Schmidt
dobles={('re',2,6),('im',2,6)}
cur=[2 if B[w] in dobles else 1 for w in W[:0]]
def bloques():
    nom=[B[w] for w in W]
    duras=[('re',1,3),('re',5,7),('re',2,6),('re',4,6),('im',1,3),('im',5,7),('im',2,6),('im',4,6)]
    return [2 if k in dobles else 1 for k in duras]+[1,1]+[0]*4
CUR=bloques()
BF={6:(4.782e-2,306.),9:(3.188e-2,459.)}   # radio y cota M_n de chi (Z28, 9.23)
TAUH=mp.mpf('0.6')
def Jhat_min(n,eps,Kx,Kf):
    e=mp.mpf(eps)
    T=mp.matrix(14,14)
    for i,a in enumerate(W):
        for j,b in enumerate(W):
            cs=Kx.get((a,b)); v=mp.mpf(0)
            if cs:
                v+=sum(mp.mpf(c.numerator)/c.denominator*e**m for m,c in enumerate(cs) if c!=0)
                for m in range(5,9): v+=mp.mpf(float(Kf[a,b,m]))*e**m
            T[i,j]=v*sc[i]*sc[j]
    T=(T+T.T)/2
    Bm=mp.matrix(NG.base_explicita(float(eps)).tolist())
    # recomputar B con mpmath (los coeficientes eps*Delta/3*2sqrt2)
    Bm=mp.matrix(14,14)
    Bn=NG.base_explicita(1.0)-NG.base_explicita(0.0)    # parte proporcional a eps
    B0=NG.base_explicita(0.0)
    for i in range(14):
        for j in range(14): Bm[i,j]=mp.mpf(B0[i,j])+e*mp.mpf(Bn[i,j])
    s=[1]*8+[e]*2+[e**2]*4
    J=2*(Bm.T*T*Bm)
    for i in range(14):
        for j in range(14): J[i,j]=J[i,j]/(s[i]*s[j])
    lmin=min(mp.eigsy((J+J.T)/2)[0])
    normB=max(abs(x) for x in mp.eigsy(Bm.T*Bm)[0])**0.5
    return lmin, normB, Bm
def analiza(n, eps_a, Rh, Rm, Rs, verbose=False):
    Kx=pickle.load(open(S+f'chi3_exacto_n{n}.pkl','rb'))[1]
    Kf=np.load(S+f'chi3_{n}_8.npz')['Kc']
    # cota de F+1 en el polidisco (Bauer-Fike con la diagonal blanda en H0')
    Vn = 2*2**0.5*eps_a + 8*Rh/2**0.5 + 2*Rm/2**0.5 + eps_a*Rs*(2+2/3)
    if n*Vn >= 1-Rs: return None
    A=Vn                      # |F + 1| <= ||V||/n
    Gmax=121*A
    eaχ,Mχ=BF[n]
    res=[]
    for eps in np.geomspace(1e-22, eps_a*0.3, 90):
        r=eps/eps_a; e=eps
        lmin,normB,Bm=Jhat_min(n,eps,Kx,Kf)
        rc=eps/eaχ
        remχ=2*normB**2*Mχ*rc**9/(1-rc)/e**4
        # Theta: |Jhat(Theta)-Jhat(0)| por bloques (corrientes) -- cota conservadora por el peor bloque
        ftheta=4*A*max(r**(n-4)/Rh**2, r**(n-3)/(Rh*Rm*e), r**(n-2)/(Rh*Rs*e**2), r**(n-2)/(Rm**2*e**2), r**(n-1)/(Rm*Rs*e**3), r**n/(Rs**2*e**4))/(1-r)*14
        lam=lmin-remχ-ftheta
        if lam<=0: continue
        beta=1/lam
        rho=e**2*min(Rh/(2*eps_a**2), Rm/(2*eps_a), Rs/2)
        L=4.5*6*Gmax*r**4/rho**3
        # gradiente reescalado: d_i/s_i, d_i <= 2A/R_i r^{n-cur_i}/(1-r); mas (p-1) tau
        Rv=[Rh]*8+[Rm]*2+[Rs]*4; sv=[1]*8+[e]*2+[e**2]*4
        pE1=2*A*r**n/(1-r)/TAUH
        g=[ (2*A/Rv[i]*r**(n-CUR[i])/(1-r) + pE1*1.0)/sv[i] for i in range(14)]
        eta=beta*float(mp.sqrt(sum(x**2 for x in g)))
        h=float(beta)*L*eta
        ok = h<=0.5 and 2*eta<=float(rho)
        if ok:
            u=2*eta; gnorm=eta/float(beta)
            defE=0.5*u*gnorm+0.6*L*u**3
            res.append((float(eps),float(h),float(defE/TAUH),float(lmin)))
    return res
for n in (6,9):
    best=None
    for eps_a in (0.005,0.01,0.02,0.03):
        for Rs in (0.1,0.3,0.5):
            for Rh in (1e-3,3e-3,1e-2):
                for Rm in (1e-3,3e-3,1e-2):
                    r=analiza(n,eps_a,Rh,Rm,Rs)
                    if r:
                        emax=max(x[0] for x in r)
                        if best is None or emax>best[0]: best=(emax,(eps_a,Rh,Rm,Rs),r)
    if best:
        emax,par,r=best
        emin=min(x[0] for x in r)
        top=max(r)
        print(f"n={n}: Kantorovich graduado VALE para eps en [{emin:.2e}, {emax:.3e}] (parametros eps_a,Rh,Rm,Rs = {par}); en el borde h={top[1]:.3f}, deficit <= {top[2]:.2e}, lambda_min(Jhat)={top[3]:.4f}",flush=True)
    else:
        print(f"n={n}: no hay rango con esta rejilla",flush=True)

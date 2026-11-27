"""Radio de (H_chi) por Bauer-Fike para n fijo (celda_k_notas.md, 9.23). Lee las series de chi generadas con
k1_serie_rapida.py (modo chi3) y k1_racional (TRES = True), guardadas en el directorio DATOS."""
import pickle, numpy as np, mpmath as mp
mp.mp.dps=60
import os
S = os.environ.get('DATOS', '.') + '/'
s2=2**0.5
for n in (6,9,12,15,18):
    B,Kx,E=pickle.load(open(S+f'chi3_exacto_n{n}.pkl','rb'))
    Kf=np.load(S+f'chi3_{n}_8.npz')['Kc']
    re=[i for i,o in enumerate(B) if o[0]!='im']; im=[i for i,o in enumerate(B) if o[0]=='im']
    W=[0,4,7,8]+[i for i in re if B[i][0]=='re']+im
    g=[1 if B[i][0]=='d' else 2 for i in W]
    def lam(e,K):
        e=mp.mpf(e); M=mp.matrix(len(W),len(W))
        for i,a in enumerate(W):
            for j,b in enumerate(W):
                cs=Kx.get((a,b)); v=mp.mpf(0)
                if cs:
                    v+=sum(mp.mpf(c.numerator)/c.denominator*e**m for m,c in enumerate(cs) if c!=0)
                    for m in range(5,K+1): v+=mp.mpf(float(Kf[a,b,m]))*e**m
                M[i,j]=v/mp.sqrt(g[i]*g[j])
        return min(mp.eigsy((M+M.T)/2)[0])
    for K in (6,8):
        tab=[float(lam(e,K)/mp.mpf(e)**4) for e in [1e-12,1e-8,1e-5,1e-3,3e-3,1e-2,3e-2]]
        kap=0.9*min(tab)
        best=(0,None)
        emax=1/(2*s2*n)
        for ea in np.linspace(emax*0.02,emax*0.98,400):
            tp=(1-1e-9)/n-2*s2*ea            # 2 sqrt2 n eps_a + n t' < 1
            M=(4*s2*ea+tp)/tp**2
            e=(kap/2*ea**(K+1)/M)**(1/(K-3))
            for _ in range(60): e=(kap/2*ea**(K+1)*(1-e/ea)/M)**(1/(K-3))
            if e>best[0]: best=(e,(ea,tp,M))
        ea,tp,M=best[1]
        print(f"n={n} K={K}: lambda_T/eps^4 in [{min(tab):.6f},{max(tab):.6f}], kappa={kap:.6f}; eps_chi = {best[0]:.3e}  (eps_a={ea:.3e}, t'={tp:.3e}, M={M:.3g})",flush=True)

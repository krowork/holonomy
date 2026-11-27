"""Certificado de Z16/Z17 para H + tO (celda_k_notas.md, 9.22, Lema Z26): eps_ef = eps_a + sqrt(2) t/4 y paredes a coste 1 - sqrt(2) t."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
import numpy as np
from mpmath import iv
import certifica_continuo as C
W0=C.W_cert
DELTA=[0.0]
def W_d(bp, lam):  # coste de pared 1 - delta: lam -> lam + delta
    return W0(bp, lam + iv.mpf(DELTA[0]))
C.W_cert=W_d
ALPHA, D = 2**0.5, 2**0.5
c,a,kap,s,sig = 0.2, 0.01244, 1.0, 0.1, 0.1
res=[]
for ea in [3e-3,4e-3,5e-3,6e-3,6.5e-3,7e-3]:
    for tp in [1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,8e-3,1e-2,1.2e-2,1.5e-2]:
        ef=ea+ALPHA*tp/4
        if ef>8.198e-3: continue
        DELTA[0]=float(iv.mpf(D)*iv.mpf(tp)).__float__() if False else D*tp*(1+1e-12)
        r16=C.cond_cert(ef,a,kap,s,c)
        r17=C.z17_cert(ef,c,a,kap,s,sig)
        if r16 is None or not r16[2] or r17 is None:
            print(f"eps_a={ea:g} t'={tp:g}: NO certifica (eps_ef={ef:.4g}, delta={DELTA[0]:.3g})",flush=True); continue
        Wb=W_d(4*iv.mpf(ef)*iv.exp(3*iv.mpf(a)*kap), 3*iv.mpf(a))
        q=r17[0]
        M=(2*Wb+2*q/15)/tp**2
        print(f"eps_a={ea:g} t'={tp:g}: eps_ef={ef:.4g} delta={DELTA[0]:.3g}  W={Wb:.3e} q={q:.3e}  M_chi<={M:.4g}",flush=True)

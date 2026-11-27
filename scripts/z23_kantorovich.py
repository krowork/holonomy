"""Condiciones explícitas de Newton-Kantorovich para Z23 (9.24)."""
import numpy as np
s2=2**0.5
TAUH=0.6      # |tau_h| = e_R - ebar >= 2/3 - O(eps^2) >= 0.6
TAUN=1.0      # ||tau||_HS <= 1
EPSCHI={12:8.9e-5,15:5.7e-5,18:4.0e-5}
def analiza(n, ea, tp, Ae, kap, etiqueta):
    cap=EPSCHI.get(n, max(1.2e-5, 4.0e-5*(18/n)**2.2))
    L=216*Ae/tp**3
    Hmax=16*Ae/tp**2
    ok=[]
    for e in np.geomspace(1e-22, ea*0.5, 6000):
        r=e/ea
        D=(2*Ae/tp)*r**(n-2)/(1-r)
        dchi=2*(2*Ae/tp**2)*r**(n-4)/(1-r)          # |chi(Theta) - chi(0)|
        kk=kap*e**4-dchi
        if kk<=0: continue
        beta=1/(2*kk)
        pE1=2*Ae*r**n/(1-r)/TAUH
        d=D+pE1*TAUN
        for _ in range(3):                           # el deficit realimenta d (despreciable)
            x=2*beta*d
            defE=0.5*x*d+0.6*L*x**3
            d=D+(pE1+defE/TAUH)*TAUN
        x=2*beta*d
        if beta*L*beta*d<=0.5 and x<=tp/2 and e<cap:
            ok.append((e,defE/TAUH,pE1))
    if not ok:
        print(f"{etiqueta}: NO hay rango"); return
    emax=max(o[0] for o in ok); emin=min(o[0] for o in ok)
    # en el borde superior: deficit frente al termino principal
    e,dp,pE1=max(ok)
    print(f"{etiqueta}: Newton-Kantorovich vale en [{emin:.2e}, {emax:.3e}] (y por debajo); en eps={e:.3e}: deficit <= {dp:.2e}, cota de |p_E-1| = {pE1:.2e}")
# n = 12, 15, 18 por Bauer-Fike (A_e = 2 sqrt2 eps_a + 2 t'), kappa' de Z29 por n (kappa/2)
for n,kap in ((12,0.002674/2),(15,0.002674/2),(18,0.002674/2),(21,0.002674/2),(24,0.002674/2),(30,0.002674/2)):
    best=None
    for ea in np.linspace(0.05,0.95,19)/(2*s2*n):
        tp=(1-1e-9)/n-2*s2*ea
        Ae=2*s2*ea+2*tp
        import io,contextlib
        f=io.StringIO()
        with contextlib.redirect_stdout(f): analiza(n,ea,tp,Ae,kap,'x')
        out=f.getvalue()
        if 'vale en' in out:
            emax=float(out.split(', ')[1].split(']')[0])
            if best is None or emax>best[0]: best=(emax,ea,tp,Ae)
    if best: analiza(n,best[1],best[2],best[3],kap,f"n={n} (Bauer-Fike, eps_a={best[1]:.3e}, t'={best[2]:.3e})")
    else: print(f"n={n} (Bauer-Fike): NO")
# uniforme (Z26): eps_a=7e-3, t'=3e-3: A_e = W + q/n
for n in (18,21,24,30,60):
    analiza(n,7e-3,3e-3,4.131e-4+1.025e-2/n,0.002674/2,f"n={n} (Z26 uniforme)")

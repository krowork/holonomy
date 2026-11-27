import sys; sys.path.insert(0,'/home/alex/fisica/holonomy/scripts'); 
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, minres
from celda_k_flujo import build, sector
from k1_racional import base
from marg_k0 import marg_k0 as marg
B=base()
def mat9(o):
    M=np.zeros((9,9),complex)
    if o[0]=='d': M[o[1],o[1]]=1
    elif o[0]=='re': M[o[1],o[2]]=M[o[2],o[1]]=1/np.sqrt(2)
    else: M[o[1],o[2]]=1j/np.sqrt(2); M[o[2],o[1]]=-1j/np.sqrt(2)
    return M
re=[i for i,o in enumerate(B) if o[0]!='im']; im=[i for i,o in enumerate(B) if o[0]=='im']
W=[0,4,7,8]+[i for i in re if B[i][0]=='re']+im
def ops(n):
    codes,dig=sector(n,0); N=codes.size; pw=3**np.arange(n)
    out=[]
    for a in W:
        M=mat9(B[a]); rows=[];cols=[];vals=[]
        for j in range(n):
            k=(j+1)%n; col=dig[j]*3+dig[k]
            for row in range(9):
                coef=M[row,col]; m=np.abs(coef)>0
                if not m.any(): continue
                new=codes[m]+(row//3-dig[j][m])*pw[j]+(row%3-dig[k][m])*pw[k]
                idx=np.searchsorted(codes,new)
                rows.append(idx); cols.append(np.nonzero(m)[0]); vals.append(coef[m])
        out.append(sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(N,N)).tocsr())
    return out
def base_explicita(eps):
    """Base graduada explícita en coordenadas W (9.25): 8 duras (saltos desde el patrón, Re e Im), 2 intermedias
    (|02>-|11>, Re e Im) y 4 blandas o_d + eps*sum_y (Delta_y/3) * (salto de tipo y), con o_d en |00>,|11>,|21>,|22>.
    En la base de Hilbert-Schmidt, 2(|10><01| + h.c.) = 2 sqrt2 * re(01,10)."""
    nom=[B[w] for w in W]; ix={k:i for i,k in enumerate(nom)}
    duras=[('re',1,3),('re',5,7),('re',2,6),('re',4,6),('im',1,3),('im',5,7),('im',2,6),('im',4,6)]
    medias=[('re',2,4),('im',2,4)]
    cols=[]
    for k in duras+medias:
        v=np.zeros(len(W)); v[ix[k]]=1; cols.append(v)
    for dd in (0,4,7,8):
        o=np.zeros(9); o[dd]=1
        D1=o[7]+o[3]+o[2]          # (2,1),(1,0),(0,2): saltos (01)->(10) y (12)->(21)
        D2=3*o[4]                  # (1,1)x3: salto inverso (20)->(11)
        v=np.zeros(len(W)); v[ix[('d',dd)]]=1
        v[ix[('re',1,3)]]+=eps*D1/3*2*np.sqrt(2)
        v[ix[('re',5,7)]]+=eps*D1/3*2*np.sqrt(2)
        v[ix[('re',4,6)]]+=eps*D2/3*2*np.sqrt(2)
        cols.append(v)
    return np.array(cols).T

def analiza(n,eps,th=np.pi,explicita=False):
    N,diag,r,c=build(n,0)
    A=sp.coo_matrix((2*eps*np.ones(r.size),(r,c)),shape=(N,N)).tocsr(); H=(A+A.T+sp.diags(diag)).tocsr()
    w,v=eigsh(H,k=1,which='SA',tol=1e-15); E0=w[0]; psi=v[:,0]
    Os=ops(n); Hs=(H-E0*sp.identity(N)).tocsr()
    xs=[];phis=[]
    for O in Os:
        ph=O@psi; ph=ph-psi*(psi.conj()@ph); phis.append(ph)
        xr,_=minres(Hs,ph.real,rtol=1e-14,maxiter=50000); xi,_=minres(Hs,ph.imag,rtol=1e-14,maxiter=50000)
        x=xr+1j*xi; x-=psi*(psi.conj()@x); xs.append(x)
    m=len(W)
    chi=np.array([[np.real(np.vdot(phis[a],xs[b])) for b in range(m)] for a in range(m)])/n; chi=(chi+chi.T)/2
    expO=[np.real(psi.conj()@(O@psi)) for O in Os]
    Ox=[[(Os[b]@xs[c]-expO[b]*xs[c]) for c in range(m)] for b in range(m)]
    T=np.zeros((m,m,m))
    for a in range(m):
        for b in range(m):
            for c in range(m):
                T[a,b,c]=np.real(np.vdot(xs[a],Ox[b][c]))
    T=2*(T+T.transpose(1,0,2)+T.transpose(0,2,1))/n   # d^3 e (sin el signo global)
    # D (sin giro) en Theta
    rho0=marg(n,eps,0); rt=marg(n,eps,th); Q=np.array([-1,0,1.])
    best=None
    for sg in (1,-1):
        V=np.diag(np.exp(-1j*sg*th*Q/n)); U=np.kron(np.eye(3),V); Dm=U.conj().T@rt@U-rho0
        if best is None or np.abs(Dm).max()<np.abs(best).max(): best=Dm
    D=np.array([np.real(np.trace(mat9(B[a])@best)) for a in W])
    # base graduada
    ev,U=np.linalg.eigh(chi)
    orden=np.argsort(ev)[::-1]; ev=ev[orden]; U=U[:,orden]
    s=np.ones(m); s[8:10]=eps; s[10:]=eps**2
    Jh=np.diag(1/s)@np.diag(2*ev)@np.diag(1/s)
    if explicita:
        U=base_explicita(eps)
        Jh=np.diag(1/s)@(2*U.T@chi@U)@np.diag(1/s)
        evj=np.linalg.eigvalsh((Jh+Jh.T)/2)
        print(f"   base explícita: J graduado, autovalores en [{evj.min():.3g}, {evj.max():.3g}]")
    Dg=U.T@D; G0=Dg/s   # D es un covector: en la base B sus componentes son B^T D
    Tg=np.einsum('abc,ai,bj,ck->ijk',T,U,U,U)/np.einsum('i,j,k->ijk',s,s,s)
    Jinv=np.linalg.inv((Jh+Jh.T)/2)
    beta=np.linalg.norm(Jinv,2); eta=np.linalg.norm(Jinv@G0)

    Lh=np.sqrt((Tg**2).sum())
    beta0=1/(2*ev.min()); eta0=np.linalg.norm(D)*beta0; L0=np.sqrt((T**2).sum())
    blk=lambda x:(np.linalg.norm(x[:8]),np.linalg.norm(x[8:10]),np.linalg.norm(x[10:]))
    print(f"n={n} eps={eps}: chi/s^2 bloques: {2*ev[0]:.3g}..{2*ev[7]:.3g} | {2*ev[8]/eps**2:.3g},{2*ev[9]/eps**2:.3g} | {2*ev[10]/eps**4:.3g}..{2*ev[13]/eps**4:.3g}")
    print(f"   |D| por bloques (duro, eps2, blando) = {['%.2e'%x for x in blk(Dg)]};  eps^(n-2)={eps**(n-2):.1e}")
    grupos={'h':slice(0,8),'m':slice(8,10),'s':slice(10,14)}
    import itertools
    blq={''.join(sorted(k)): np.sqrt((Tg[grupos[k[0]],grupos[k[1]],grupos[k[2]]]**2).sum()) for k in itertools.product('hms',repeat=3)}
    print("   bloques de T graduado:", ', '.join(f"{k}={v:.2e}" for k,v in sorted(blq.items())))
    print(f"   T graduado: norma Frobenius = {Lh:.3e}  (sin graduar {L0:.3e});  bloques sss={np.sqrt((Tg[10:,10:,10:]**2).sum()):.2e}, hhh={np.sqrt((Tg[:8,:8,:8]**2).sum()):.2e}")
    print(f"   Kantorovich graduado h = beta*L*eta = {beta*Lh*eta:.3e} (beta={beta:.3g}, eta={eta:.3e});  sin graduar h0 = {beta0*L0*eta0:.3e}")
if __name__ == '__main__':
  for n in (6,9):
    for eps in ((0.005,0.01,0.02,0.04) if n==6 else (0.04,0.06)):
        analiza(n,eps,explicita=True)

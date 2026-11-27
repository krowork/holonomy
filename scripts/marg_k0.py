import sys; sys.path.insert(0,'/home/alex/fisica/holonomy/scripts')
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from celda_k_flujo import build, sector
def gs_k0(n,eps,th):
    """Fundamental de momento 0 de H(th) (fases uniformes) en el sector Sz = 0."""
    N,diag,r,c=build(n,0); ph=np.exp(1j*th/n)
    A=sp.coo_matrix((2*eps*ph*np.ones(r.size),(r,c)),shape=(N,N)).tocsr(); H=(A+A.getH()+sp.diags(diag)).tocsr()
    codes,dig=sector(n,0); pw=3**np.arange(n)
    Tc=np.dot(np.roll(dig,1,axis=0).T,pw)          # traslación s_i -> s_{i-1}
    perm=np.searchsorted(codes,Tc)
    Tm=sp.coo_matrix((np.ones(N),(perm,np.arange(N))),shape=(N,N)).tocsr()
    # proyector de momento 0: P = (1/n) sum_k T^k ; se aplica a H restringido con un desplazamiento
    w,v=eigsh(H,k=3,which='SA',tol=1e-15)
    # dentro del multiplete, diagonalizar T
    Tv=np.array([[np.vdot(v[:,i],Tm@v[:,j]) for j in range(3)] for i in range(3)])
    tw,tv=np.linalg.eig(Tv)
    cand=[(abs(tw[i]-1),i) for i in range(3)]
    _,i=min(cand); vec=v@tv[:,i]; vec/=np.linalg.norm(vec)
    return codes,vec
def marg_k0(n,eps,th):
    codes,vec=gs_k0(n,eps,th)
    psi=np.zeros(3**n,complex); psi[codes]=vec
    X=psi.reshape(3**(n-2),3,3); return np.einsum('rba,rdc->abcd',X,X.conj()).reshape(9,9)

"""Corrección de tamaño finito de la marginal (sensibilidad_celda3.tex, problema abierto 3).

sigma_n(eps): marginal de arista del fundamental de momento 0 (Theta = 0) del anillo n, por ED.
Imprime |sigma_n - sigma_nref|_max y e_n - e_nref, con nref el mayor n dado. Resultado: O(eps^{2n/3-1}).
Uso: python scripts/marginal_tamano.py 0.05,0.1,0.2 6,9,12
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from celda_k_flujo import build, sector

def marg(n, eps):
    N, diag, r, c = build(n, 0)
    A = sp.coo_matrix((2*eps*np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    H = (A + A.T + sp.diags(diag)).tocsr()
    w, v = eigsh(H, k=3, which="SA", tol=1e-13)
    codes, dig = sector(n, 0)
    # traslacion: s_j -> s_{j+1}
    pw = 3**np.arange(n, dtype=np.int64)
    sh = np.roll(dig, -1, axis=0)
    tcodes = (sh.astype(np.int64) * pw[:, None]).sum(0)
    perm = np.searchsorted(codes, tcodes)
    P = []
    for i in range(3):
        x = v[:, i].copy(); acc = x.copy()
        for _ in range(n-1):
            x = x[perm]; acc += x
        P.append(acc / n)
    P = np.array(P).T
    # combinacion de momento 0 en el espacio de los 3 vectores
    U, s, _ = np.linalg.svd(v.T @ P)
    psi = v @ U[:, 0]; psi /= np.linalg.norm(psi)
    E = psi @ (H @ psi)
    # marginal sitios 0,1 (traslacion invariante)
    idx = dig[0]*3 + dig[1]
    rest = codes // 9
    rho = np.zeros((9, 9))
    order = np.lexsort((idx, rest))
    # agrupar por rest
    rs, ix, ps = rest[order], idx[order], psi[order]
    bounds = np.flatnonzero(np.diff(rs)) + 1
    for blk_i, blk_p in zip(np.split(ix, bounds), np.split(ps, bounds)):
        vec = np.zeros(9); vec[blk_i] = blk_p
        rho += np.outer(vec, vec)
    return E/n, rho, w

if __name__ == "__main__":
    eps_list = [float(x) for x in sys.argv[1].split(",")]
    ns = [int(x) for x in sys.argv[2].split(",")]
    out = {}
    for eps in eps_list:
        for n in ns:
            e, rho, w = marg(n, eps)
            out[(eps, n)] = (e, rho)
            print(f"eps={eps} n={n} e={e:.15f} gaps={w[1]-w[0]:.3e},{w[2]-w[0]:.3e} tr={np.trace(rho):.12f}", flush=True)
    pass
    nref = max(ns)
    for eps in eps_list:
        for n in ns[:-1]:
            d = out[(eps, n)][1] - out[(eps, nref)][1]
            print(f"eps={eps} n={n}: |sigma_n - sigma_{nref}|_max = {np.abs(d).max():.3e}, "
                  f"e_n - e_{nref} = {out[(eps,n)][0]-out[(eps,nref)][0]:.3e}")

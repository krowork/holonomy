"""Serie en eps de los datos sigma_N (marginal de arista del fundamental de momento 0 del anillo N, Theta = 0).

Rayleigh-Schrödinger desde la suma de los tres patrones en el sector Sz = 0, como serie_respuesta.py, pero sin
vectores de respuesta: solo la energía por sitio E/N y el gradiente grad_a = tr(o_a rho_01) en las coordenadas W de
newton_graduado (base de Hilbert-Schmidt). rho_01 se obtiene de las series de psi por traslación (psi es invariante
por traslación orden a orden), así que no hace falta construir los O_a del anillo: cabe N = 15.
Los coeficientes coinciden con grad[0] y E[0] de serie_respuesta.py para el mismo anillo (comprobación con N = 9).
Uso: python scripts/serie_datos.py N K salida.npz
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from celda_k_flujo import build, sector  # noqa: E402
_argv, sys.argv = sys.argv, sys.argv[:1]   # k1_racional lee sys.argv al importarse
from k1_racional import base  # noqa: E402
import newton_graduado as NG  # noqa: E402
sys.argv = _argv


def serie_datos(N, K):
    nN, diag, r, c = build(N, 0)
    A = sp.coo_matrix((2.0 * np.ones(r.size), (r, c)), shape=(nN, nN)).tocsr()
    T = (A + A.T).tocsr()
    codes, dig = sector(N, 0)
    E0 = -float(N)
    pat = diag == E0
    assert pat.sum() == 3
    den = E0 - diag
    den[pat] = 1.0
    psi = [pat.astype(float)]
    E = [E0]
    for m in range(1, K + 1):
        rhs = T @ psi[m - 1]
        if np.ptp(rhs[pat]) > 1e-9 * (1 + np.abs(rhs).max()):
            raise RuntimeError("patrones no simétricos")
        E.append(rhs[pat][0])
        for k in range(1, m + 1):
            rhs = rhs - E[k] * psi[m - k]
        v = rhs / den
        v[pat] = 0.0
        psi.append(v)
    Psi = np.array(psi)                                   # (K+1) x dim, real
    # rho_01 por órdenes: rho^{(m)}[c, r] = sum_i sum_rest psi_i(c, rest) psi_{m-i}(r, rest)
    idx = (dig[0].astype(np.int64) * 3 + dig[1])
    rest = codes // 9
    _, gidx = np.unique(rest, return_inverse=True)
    X = np.zeros((gidx.max() + 1, 9, K + 1))
    X[gidx, idx, :] = Psi.T
    # G[i, j, a, b] = sum_rest psi_i(a, rest) psi_j(b, rest), para todos los pares de órdenes
    G = np.einsum('gai,gbj->ijab', X, X, optimize=True)
    rho = np.array([sum(G[i, m - i] for i in range(m + 1)) for m in range(K + 1)])   # (K+1) x 9 x 9
    nrm = np.array([np.trace(rho[m]) for m in range(K + 1)])
    # inversa de la serie de la norma
    inv = np.zeros(K + 1); inv[0] = 1 / nrm[0]
    for m in range(1, K + 1):
        inv[m] = -sum(nrm[i] * inv[m - i] for i in range(1, m + 1)) / nrm[0]
    rhon = np.array([sum(rho[i] * inv[m - i] for i in range(m + 1)) for m in range(K + 1)])
    B = base()
    grad = np.zeros((len(NG.W), K + 1), complex)
    for a, w in enumerate(NG.W):
        M = NG.mat9(B[w])
        # <psi|O|psi>/N = tr(M rho), con rho[c, r] = psi(c) psi(r)^*
        grad[a] = np.einsum('rc,mcr->m', M, rhon)
    return dict(E=np.array(E) / N, grad=grad, rho=rhon)


if __name__ == "__main__":
    N, K = int(sys.argv[1]), int(sys.argv[2])
    r = serie_datos(N, K)
    np.savez(sys.argv[3], E=r['E'], grad=r['grad'], rho=r['rho'], N=N, K=K)
    print(f"N={N} K={K}: E/N = {r['E'][:6]}")

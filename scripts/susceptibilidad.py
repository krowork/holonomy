"""Susceptibilidad estática de las marginales de arista del fundamental (celda_k_notas.md, 9.18).

chi_ab = (1/n) Re <phi_a| (H - E0)^{-1} |phi_b>,  phi_a = (1 - P0) O_a psi0,  O_a = sum_j o_a(j, j+1),
con o_a la base real de las matrices 9x9 invariantes por U(1). El coste en energía por sitio de mover la marginal
(tangencialmente) en dm es (1/4) dm^T chi^{-1} dm: la curvatura del borde del conjunto de marginales en sigma_n.
Uso: python scripts/susceptibilidad.py eps n [n ...]
"""
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, minres

from celda_k_flujo import build, sector


def base():
    """Base real (ortonormal en Hilbert-Schmidt) de las 9x9 hermíticas que conmutan con Q(x)1 + 1(x)Q."""
    car = [a + b for a in range(3) for b in range(3)]
    B = []
    for r in range(9):
        M = np.zeros((9, 9), complex); M[r, r] = 1; B.append(M)
    for r in range(9):
        for c in range(r + 1, 9):
            if car[r] == car[c]:
                M = np.zeros((9, 9), complex); M[r, c] = M[c, r] = 1 / np.sqrt(2); B.append(M)
                M = np.zeros((9, 9), complex); M[r, c] = -1j / np.sqrt(2); M[c, r] = 1j / np.sqrt(2); B.append(M)
    return B


def chi(n, eps, matrices=False):
    """matrices=True: devuelve también V = X*X/n (varianza), D = X*(H-E0)X/n (doble conmutador) y
    D3 = X*(H-E0)^3 X/n (conmutador anidado), con X = [phi_a]."""
    N, diag, r, c = build(n, 0)
    A = sp.coo_matrix((2 * eps * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    H = (A + A.T + sp.diags(diag)).tocsr()
    w, v = eigsh(H, k=2, which="SA", tol=1e-14)
    E0, psi = w[0], v[:, 0]
    codes, dig = sector(n, 0)
    pos = {int(x): i for i, x in enumerate(codes)}
    pw = 3 ** np.arange(n)
    B = base()
    # O_a psi para cada a: sum_j sum_{(ab),(a'b')} o[(ab),(a'b')] |ab><a'b'|_{j,j+1}
    phis = []
    for o in B:
        out = np.zeros(N, complex)
        for j in range(n):
            k = (j + 1) % n
            sa, sb = dig[j], dig[k]
            col = sa * 3 + sb
            for row in range(9):
                coef = o[row, col]
                m = np.abs(coef) > 0
                if not m.any():
                    continue
                a2, b2 = row // 3, row % 3
                new = codes[m] + (a2 - sa[m]) * pw[j] + (b2 - sb[m]) * pw[k]
                idx = np.array([pos[int(x)] for x in new])
                np.add.at(out, idx, coef[m] * psi[m])
        out -= psi * (psi @ out)
        phis.append(out)
    Hs = H - E0 * sp.identity(N)
    xs = []
    for ph in phis:
        xr, _ = minres(Hs, ph.real, rtol=1e-12, maxiter=20000)
        xi, _ = minres(Hs, ph.imag, rtol=1e-12, maxiter=20000)
        x = xr + 1j * xi
        x -= psi * (psi @ x)
        xs.append(x)
    C = np.array([[np.real(np.vdot(a, b)) for b in xs] for a in phis]) / n
    if matrices:
        V = np.array([[np.real(np.vdot(a, b)) for b in phis] for a in phis]) / n
        Hp = [Hs @ ph for ph in phis]
        D = np.array([[np.real(np.vdot(a, b)) for b in Hp] for a in phis]) / n
        Hp2 = [Hs @ x for x in Hp]
        D3 = np.array([[np.real(np.vdot(a, b)) for b in Hp2] for a in Hp]) / n
        return (C + C.T) / 2, (V + V.T) / 2, (D + D.T) / 2, (D3 + D3.T) / 2
    return (C + C.T) / 2, w[1] - E0


if __name__ == "__main__":
    eps = float(sys.argv[1])
    for n in [int(x) for x in sys.argv[2:]]:
        C, gap = chi(n, eps)
        ev = np.linalg.eigvalsh(C)
        nz = ev[ev > 1e-9 * ev.max()]
        print(f"eps={eps} n={n}: gap={gap:.4f}  autovalores de chi: {len(ev) - len(nz)} nulos; "
              f"no nulos min={nz.min():.4e} max={nz.max():.4e}", flush=True)
        print("   ", " ".join(f"{x:.3e}" for x in ev), flush=True)

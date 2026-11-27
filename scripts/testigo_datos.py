"""Del testigo a los datos: sensibilidad real p(n, Theta) frente a la del testigo fijo (exploración).

Modelo de llenado 1/3 (celda_k_llenado.py): carga q = (1, 0, 0), término h0 = -P_asc + eps (salto de la partícula)
+ lam (intercambio neutro 1 <-> 2). Los datos son la marginal de dos sitios del fundamental de sum h0 en un anillo de
NREF sitios (3 | NREF) con Theta = 0, que aproxima la de la cadena infinita salvo un error exponencialmente pequeño.

Para cada n y cada Theta se calculan:
  - p(n, Theta), el dual exacto (dual_celda.dual), minimizando sobre todos los testigos U(1) de dos sitios;
  - el cociente del testigo fijo w0 = -h0, que es la cota superior de p(n, Theta) que da ese testigo solo.
La sensibilidad es max_Theta - min_Theta de cada uno.

Uso: OMP_NUM_THREADS=1 python scripts/testigo_datos.py eps lam n [n ...]   (NTHETA y NREF en el entorno)
"""
import os
import sys
import time

import numpy as np

from celda_k_llenado import build
from dual_celda import anillo, dual, lmax

Q = [1, 0, 0]


def termino(eps, lam):
    h = np.zeros((9, 9))
    for a in range(3):
        h[3 * a + (a + 1) % 3, 3 * a + (a + 1) % 3] = -1
    for x in (1, 2):  # (0, x) <-> (x, 0)
        h[3 * x + 0, 3 * 0 + x] = h[3 * 0 + x, 3 * x + 0] = eps
    h[3 * 2 + 1, 3 * 1 + 2] = h[3 * 1 + 2, 3 * 2 + 1] = lam
    return h


def datos(h, nref, lam):
    """Marginal de dos sitios del fundamental del anillo de nref sitios (Theta = 0, sector de nref/3 partículas)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    import celda_k_llenado as m
    m.LAM = lam
    N, diag, r, c, r2, c2 = build(nref, nref // 3)
    eps = h[1, 3]
    A = sp.coo_matrix((np.r_[eps * np.ones(r.size), lam * np.ones(r2.size)], (np.r_[r, r2], np.r_[c, c2])),
                      shape=(N, N)).tocsr()
    H = A + A.getH() + sp.diags(diag)
    w, v = eigsh(H, k=1, which="SA", tol=1e-13)
    codes = np.arange(3**nref, dtype=np.int64)
    dig = (codes[None, :] // 3 ** np.arange(nref)[:, None]) % 3
    codes = codes[(dig == 0).sum(0) == nref // 3]
    psi = np.zeros(3**nref)
    psi[codes] = v[:, 0]
    T = psi.reshape(3 ** (nref - 2), 3, 3)  # ejes: (resto, s1, s0)
    rho = np.einsum("rba,rdc->abcd", T, T.conj()).reshape(9, 9)  # (s0 s1),(s0' s1')
    return rho, w[0] / nref


def main():
    eps, lam = float(sys.argv[1]), float(sys.argv[2])
    ns = [int(x) for x in sys.argv[3:]]
    nth = int(os.environ.get("NTHETA", 16))
    nref = int(os.environ.get("NREF", 12))
    h = termino(eps, lam)
    sigma, e_ref = datos(h, nref, lam)
    assert abs(np.trace(sigma) - 1) < 1e-10 and np.allclose(sigma, sigma.conj().T)
    tau = sigma - np.eye(9) / 9
    w0 = -h
    g = np.real(np.trace(w0 @ tau))
    print(f"eps={eps} lam={lam} NREF={nref}: energía por enlace {e_ref:.10f}, g = tr(w0 tau) = {g:.6f}", flush=True)
    thetas = np.linspace(0, 2 * np.pi, nth + 1)[:-1]
    for n in ns:
        t0 = time.time()
        P, L, F = [], [], []
        for t in thetas:
            lam_max, _ = lmax(anillo(n, 3, Q, w0, t))
            F.append((lam_max - n * np.trace(w0).real / 9) / (n * g))
            sup, _, inf = dual(n, 3, Q, tau, t, w0=w0, metodo="suave")
            P.append(sup)
            L.append(inf)
        P, L, F = np.array(P), np.array(L), np.array(F)
        sP, sF = P.max() - P.min(), F.max() - F.min()
        # horquilla certificada de la sensibilidad: L <= p <= P en cada Theta
        s_inf, s_sup = max(L.max() - P.min(), 0.0), P.max() - L.min()
        print(f"n={n:2d} (n%3={n % 3}): p en [{P.min():.9f}, {P.max():.9f}]  s_datos={sP:.4e} "
              f"(certificada en [{s_inf:.3e}, {s_sup:.3e}])  "
              f"s_testigo_fijo={sF:.4e}  cociente={sP / sF:.3f}  n^3 s_datos={n**3 * sP:.4f}  "
              f"argmax p/pi={thetas[P.argmax()] / np.pi:.2f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

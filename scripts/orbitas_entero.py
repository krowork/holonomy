"""Lado k ∤ n con carga entera: las tres órbitas de defectos por su carga (celda_k_notas.md, Sec. 7.14).

Modelo S+S- de celda_k_flujo.py (q = s - 1). Con 3 ∤ n, las configuraciones de un defecto forman tres órbitas
de la traslación, de carga total Q = sum_{i<n} q(c+i) en {-1, 0, 1}: (S_d) se cumple con la carga U(1). Para cada
Q se da el fundamental del sector en una malla de Theta, Delta_def(Q) = min_Theta E_0(Q) - n e_ref,
R_n(Q) = max - min y n^2 R_n(Q). e_ref = E_0(NREF, Q = 0, Theta = 0)/NREF con 3 | NREF.
Uso: python scripts/orbitas_entero.py eps n [n ...]   (NTHETA, NREF en el entorno)
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from celda_k_flujo import build

NTHETA = int(os.environ.get("NTHETA", 12))
NREF = int(os.environ.get("NREF", 12))


def e0(n, Q, eps, theta, cache={}):
    if (n, Q) not in cache:
        cache[(n, Q)] = build(n, Q)
    N, diag, r, c = cache[(n, Q)]
    A = sp.coo_matrix((2 * eps * np.exp(1j * theta / n) * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    H = A + A.getH() + sp.diags(diag)
    if N < 1500:
        return np.linalg.eigvalsh(H.toarray())[0]
    return eigsh(H, k=1, which="SA", tol=1e-11)[0][0]


def main():
    eps = float(sys.argv[1])
    e_ref = e0(NREF, 0, eps, 0.0) / NREF
    thetas = np.linspace(0, 2 * np.pi, NTHETA + 1)[:-1]
    print(f"eps={eps}: e_ref = {e_ref:.12f}", flush=True)
    for n in [int(x) for x in sys.argv[2:]]:
        fila = []
        for Q in (-1, 0, 1):
            E = np.array([e0(n, Q, eps, t) for t in thetas])
            R = E.max() - E.min()
            fila.append(f"Q={Q:+d}: Delta={E.min() - n * e_ref:.5f} R={R:.2e} n2R={n * n * R:.3f}")
        print(f"n={n:2d} (n%3={n % 3}): " + "   ".join(fila), flush=True)


if __name__ == "__main__":
    main()

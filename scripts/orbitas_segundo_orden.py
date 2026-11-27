"""Banda de un defecto a segundo orden, por órbita de carga (celda_k_notas.md, 7.14 y 9.14). Modelo S+S-.

Teoría de perturbaciones degenerada en el subespacio P de un defecto (sector de carga Q):
  primer orden:  P V P (se comprueba que es 0: ningún salto mueve el defecto sin crear paredes);
  segundo orden: H2 = -P V (H0 - E0)^{-1} V P, con amplitud 2 por salto (como en celda_k_flujo.build).
Se resta n e2 con e2 = -4/3, la energía del vacío por enlace a segundo orden. Lo que queda es w_Q(kappa), en
omega_Q = 1 + eps^2 w_Q + O(eps^3), y no depende de n.
Uso: python scripts/orbitas_segundo_orden.py [n ...]
"""
import sys

import numpy as np
import scipy.sparse as sp

from celda_k_flujo import build

E2_VACIO = -4.0 / 3.0


def bandas(n, Q):
    N, diag, r, c = build(n, Q)
    A = sp.coo_matrix((2 * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    V = (A + A.T).tocsr()
    d0 = diag.min()
    P = np.where(np.abs(diag - d0) < 1e-9)[0]
    VP = V[:, P].toarray()
    primer = np.abs(VP[P]).max()
    den = np.where(np.abs(diag - d0) > 1e-9, diag - d0, np.inf)
    H2 = -(VP.T / den) @ VP
    w = np.linalg.eigvalsh(H2) - n * E2_VACIO
    return len(P), primer, w


if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]] or [10, 13, 16, 11, 14, 17]:
        fila = []
        for Q in (-1, 0, 1):
            nd, p1, w = bandas(n, Q)
            fila.append(f"Q={Q:+d}: |PVP|={p1:.0f}  w en [{w.min():.4f}, {w.max():.4f}]")
        print(f"n={n:2d} (n%3={n % 3}): " + "   ".join(fila), flush=True)


def tercer_orden(n, Q):
    """Autovalores de H3 = P V R V R V P (R = Q0/(E0 - H0)), con P V P = 0. Para las cargadas H2 es plana, así que
    H3 es el siguiente término del hamiltoniano efectivo en el subespacio de un defecto."""
    N, diag, r, c = build(n, Q)
    A = sp.coo_matrix((2 * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    V = (A + A.T).tocsr()
    d0 = diag.min()
    P = np.where(np.abs(diag - d0) < 1e-9)[0]
    Rd = np.where(np.abs(diag - d0) > 1e-9, 1 / (d0 - np.where(np.abs(diag - d0) > 1e-9, diag, d0 + 1)), 0.0)
    X = V[:, P].toarray()
    X = Rd[:, None] * X
    X = Rd[:, None] * (V @ X)
    H3 = (V @ X)[P]
    return np.linalg.eigvals(H3)

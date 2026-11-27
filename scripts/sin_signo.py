"""¿Es H_n(eps, Theta*) estocuástica tras un gauge diagonal? Y ¿está el fundamental en el sector del multiplete?
(celda_k_notas.md, Sec. 7.6; exploración).

Para Theta* en {0, pi}: se recorre por anchura el grafo de configuraciones del sector, se fijan fases D(s) con
D(s) H_{ss'} conj(D(s')) = -|H_{ss'}| en un árbol y se comprueban todas las aristas. Se dan también el número de
componentes conexas (irreducibilidad) y el fundamental de los sectores vecinos frente al del multiplete.

Uso: python scripts/sin_signo.py modelo eps n [n ...]   (MU en el entorno, solo llenado)
"""
import os
import sys
from collections import deque

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

import celda_k_flujo as ent
import celda_k_llenado as ll

MU = float(os.environ.get("MU", 1.0))


def ham(modelo, n, eps, theta, sector):
    ph = np.exp(1j * theta / n)
    if modelo == "entero":
        N, diag, r, c = ent.build(n, sector)
        A = sp.coo_matrix((2 * eps * ph * np.ones(r.size), (r, c)), shape=(N, N))
    else:
        N, diag, r, c, r2, c2 = ll.build(n, sector)
        A = sp.coo_matrix((np.r_[eps * ph * np.ones(r.size), MU * eps * np.ones(r2.size)],
                           (np.r_[r, r2], np.r_[c, c2])), shape=(N, N))
    A = A.tocsr()
    return (A + A.getH()).tocsr(), diag


def sin_signo(V):
    """Devuelve (estocuástica tras gauge, número de componentes)."""
    V = V.tocsr()
    N = V.shape[0]
    ncomp, lab = connected_components(V != 0, directed=False)
    D = np.zeros(N, complex)
    for raiz in np.unique(lab, return_index=True)[1]:
        D[raiz] = 1
        cola = deque([raiz])
        while cola:
            s = cola.popleft()
            for idx in range(V.indptr[s], V.indptr[s + 1]):
                t, h = V.indices[idx], V.data[idx]
                if D[t] == 0:
                    # queremos D(t) h_ts conj(D(s)) = -|h|, con h_ts = conj(h_st) = conj(h)
                    D[t] = -abs(h) * D[s] / np.conj(h)
                    cola.append(t)
    W = sp.diags(D) @ V @ sp.diags(D.conj())
    W = W.tocoo()
    return bool(np.all((W.data.real <= 1e-12) & (np.abs(W.data.imag) <= 1e-12))), ncomp


def e0(V, diag):
    H = V + sp.diags(diag)
    return np.linalg.eigvalsh(H.toarray())[0] if H.shape[0] < 3000 else eigsh(H, k=1, which="SA")[0][0]


def main():
    modelo, eps = sys.argv[1], float(sys.argv[2])
    for n in [int(x) for x in sys.argv[3:]]:
        s0 = 0 if modelo == "entero" else n // 3
        vecinos = [-1, 1] if modelo == "entero" else [n // 3 - 1, n // 3 + 1]
        for th in (0.0, np.pi):
            V, diag = ham(modelo, n, eps, th, s0)
            ok, nc = sin_signo(V)
            print(f"n={n} Theta*={th / np.pi:.0f}pi: estocuástica tras gauge: {ok}; componentes: {nc}")
        E = {s: e0(*ham(modelo, n, eps, 0.0, s)) for s in [s0] + vecinos}
        print(f"   fundamental por sector (Theta=0): " + "  ".join(f"{s}: {E[s]:.6f}" for s in E)
              + f"   hueco al sector vecino: {min(E[s] for s in vecinos) - E[s0]:.4f}", flush=True)


if __name__ == "__main__":
    main()

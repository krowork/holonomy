"""Teoría de un defecto para el intercambio SU(3) (celda_k_notas.md, Sec. 7.10; exploración).

H(Theta) = -P_asc + eps * sum_{a != b} |b a><a b| en el anillo de n sitios, carga q = (1, 0, 0), giro uniforme
(el paso que lleva la especie 0 de j a j+1 lleva e^{-i Theta/n}). El intercambio conserva el número de cada especie,
así que se trabaja por sectores de contenido c = (c_0, c_1, c_2).

Para cada n se da:
  - si 3 | n: E_{c*} y el hueco a los sectores con un cambio de especie (el continuo de dos defectos);
  - si 3 no divide a n: el fundamental E_0(Theta) sobre los sectores de contenido más cercanos a n/3 (un defecto),
    su energía medida desde el vacío vestido, Delta_def(Theta) = E_0(Theta) - n e_ref, la respuesta
    R_n = max - min sobre Theta (malla NTHETA) y n^2 R_n.
e_ref es E_{c*}(NREF)/NREF, con NREF múltiplo de 3 (por defecto 12): aproxima e_KT con error O(eps^{2 NREF/3}).

Uso: python scripts/defecto_su3.py eps n [n ...]   (NTHETA, NREF en el entorno)
"""
import os
import sys
from itertools import permutations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

CACHE = {}


def sector(n, c):
    """Configuraciones de contenido c, como códigos sum s_j 3^j ordenados, y sus dígitos."""
    key = (n, c)
    if key in CACHE:
        return CACHE[key]
    # generación recursiva de las palabras con el contenido dado
    palabras = []

    def rec(pref, resto):
        if len(pref) == n:
            palabras.append(pref)
            return
        for a in range(3):
            if resto[a]:
                r = list(resto)
                r[a] -= 1
                rec(pref + [a], r)
    rec([], list(c))
    dig = np.array(palabras, dtype=np.int8)
    codes = (dig.astype(np.int64) * 3 ** np.arange(n, dtype=np.int64)[None, :]).sum(1)
    o = np.argsort(codes)
    codes, dig = codes[o], dig[o]
    pw = 3 ** np.arange(n, dtype=np.int64)
    diag = np.zeros(codes.size)
    filas, cols, fases = [], [], []
    for j in range(n):
        i2 = (j + 1) % n
        a, b = dig[:, j].astype(np.int64), dig[:, i2].astype(np.int64)
        diag -= ((b - a) % 3 == 1)
        m = a != b
        nuevo = codes[m] + (b[m] - a[m]) * pw[j] + (a[m] - b[m]) * pw[i2]
        filas.append(np.searchsorted(codes, nuevo))
        cols.append(np.nonzero(m)[0])
        # carga que pasa de j a j+1: q(a) - q(b), con q = (1, 0, 0); el giro da e^{-i phi (q_b' - q_b)}
        dq = (a[m] == 0).astype(int) - (b[m] == 0).astype(int)
        fases.append(dq)
    CACHE[key] = (codes.size, diag, np.concatenate(filas), np.concatenate(cols), np.concatenate(fases))
    return CACHE[key]


def e0(n, c, eps, theta, k=1):
    N, diag, f, co, dq = sector(n, c)
    A = sp.coo_matrix((eps * np.exp(-1j * theta / n * dq), (f, co)), shape=(N, N)).tocsr()
    H = A + sp.diags(diag)  # A ya es hermítica: contiene los dos sentidos de cada intercambio
    if N <= 1500:
        return np.linalg.eigvalsh(H.toarray())[:k]
    return np.sort(eigsh(H, k=k, which="SA", tol=1e-11)[0])


def contenidos_cercanos(n):
    m, r = divmod(n, 3)
    base = [m, m, m]
    out = set()
    if r == 0:
        return [tuple(base)]
    for extra in permutations([1] * r + [0] * (3 - r)):
        out.add(tuple(b + e for b, e in zip(base, extra)))
    return sorted(out)


def main():
    eps = float(sys.argv[1])
    nth = int(os.environ.get("NTHETA", 16))
    nref = int(os.environ.get("NREF", 12))
    e_ref = e0(nref, (nref // 3,) * 3, eps, 0.0)[0] / nref
    print(f"eps={eps}: e_ref = E_c*({nref})/{nref} = {e_ref:.12f}", flush=True)
    thetas = np.linspace(0, 2 * np.pi, nth + 1)[:-1]
    for n in [int(x) for x in sys.argv[2:]]:
        if n % 3 == 0:
            cs = (n // 3,) * 3
            E = e0(n, cs, eps, 0.0)[0]
            vec = [tuple(np.array(cs) + d) for d in [(1, -1, 0), (-1, 1, 0), (1, 0, -1), (-1, 0, 1), (0, 1, -1), (0, -1, 1)]]
            hue = min(e0(n, v, eps, 0.0)[0] for v in vec) - E
            print(f"n={n:2d} (3|n): E_c* - n e_ref = {E - n * e_ref:+.3e}   hueco de dos defectos = {hue:.6f}", flush=True)
            continue
        best = None
        for c in contenidos_cercanos(n):
            E = np.array([e0(n, c, eps, t)[0] for t in thetas])
            if best is None or E.min() < best[1].min():
                best = (c, E)
        c, E = best
        D = E - n * e_ref
        R = E.max() - E.min()
        print(f"n={n:2d} (n%3={n % 3}) sector {c}: Delta_def en [{D.min():.6f}, {D.max():.6f}]  R_n={R:.4e}  "
              f"n^2 R_n={n * n * R:.4f}  argmin Theta/pi={thetas[E.argmin()] / np.pi:.2f}", flush=True)


if __name__ == "__main__":
    main()

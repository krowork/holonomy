"""Orden de celda 3 con llenado 1/3: carga fraccionaria por sitio (exploración).

Tres estados por sitio, s in {0,1,2}; la carga es q = 1 si s = 0 y 0 si no (una partícula por celda en el orden
...0 1 2 0 1 2...). H(Theta) = -sum P_asc + eps * sum_{x=1,2} (e^{i Theta/n} |x 0><0 x| + h.c.): la partícula salta.
Con 3 no divisor de n, el llenado 1/3 no cabe y el defecto lleva carga fraccionaria; se compara la respuesta al
flujo R_n = max_Theta E_0 - min_Theta E_0 con la del modelo de carga entera (celda_k_flujo.py).

Con LAM > 0 en el entorno se añade LAM * sum (|2 1><1 2| + h.c.), un intercambio neutro que rompe la conservación de
la secuencia de especies neutras (sin él, el defecto de carga 1/3 solo se mueve cuando una partícula da la vuelta).

Uso: python scripts/celda_k_llenado.py eps [eps ...]   (NMAX y LAM en el entorno; por defecto 11 y 0)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402


def build(n, Np):
    codes = np.arange(3**n, dtype=np.int64)
    dig = np.empty((n, codes.size), dtype=np.int8)
    c = codes.copy()
    for i in range(n):
        dig[i] = c % 3
        c //= 3
    keep = (dig == 0).sum(0) == Np
    codes, dig = codes[keep], dig[:, keep]
    pw = 3 ** np.arange(n, dtype=np.int64)
    diag = np.zeros(codes.size)
    rows, cols, rows2, cols2 = [], [], [], []
    for i in range(n):
        j = (i + 1) % n
        a, b = dig[i], dig[j]
        diag -= ((b - a) % 3 == 1)
        m = (a == 0) & (b > 0)  # partícula en i salta a j: (0, x) -> (x, 0)
        x = b[m].astype(np.int64)
        new = codes[m] + x * pw[i] - x * pw[j]
        rows.append(np.searchsorted(codes, new))
        cols.append(np.nonzero(m)[0])
        m2 = (a == 1) & (b == 2)  # intercambio neutro (1, 2) -> (2, 1), sin carga: no ve el flujo
        rows2.append(np.searchsorted(codes, codes[m2] + pw[i] - pw[j]))
        cols2.append(np.nonzero(m2)[0])
    return (codes.size, diag, np.concatenate(rows), np.concatenate(cols),
            np.concatenate(rows2), np.concatenate(cols2))


CACHE = {}
LAM = float(os.environ.get("LAM", 0))  # amplitud del intercambio neutro 1 <-> 2


def e0(n, eps, theta):
    best = np.inf
    for Np in {n // 3, (n + 2) // 3}:
        if (n, Np) not in CACHE:
            CACHE[(n, Np)] = build(n, Np)
        N, diag, r, c, r2, c2 = CACHE[(n, Np)]
        A = sp.coo_matrix((np.r_[eps * np.exp(1j * theta / n) * np.ones(r.size), LAM * np.ones(r2.size)],
                           (np.r_[r, r2], np.r_[c, c2])), shape=(N, N)).tocsr()
        H = A + A.getH() + sp.diags(diag)
        w = np.linalg.eigvalsh(H.toarray())[0] if N < 400 else eigsh(H, k=1, which="SA", tol=1e-12)[0][0]
        best = min(best, w)
    return best


def main():
    thetas = np.linspace(0, 2 * np.pi, 9)[:-1]
    for eps in [float(x) for x in sys.argv[1:]] or [0.2]:
        print(f"eps = {eps}, LAM = {LAM}", flush=True)
        for n in range(5, int(os.environ.get("NMAX", 11)) + 1):
            E = np.array([e0(n, eps, t) for t in thetas])
            R = E.max() - E.min()
            print(f"{n:>3} {n % 3:>3} R_n={R:11.4e} n^2R_n={n * n * R:9.5f} argmin/pi={thetas[E.argmin()] / np.pi:4.2f}  "
                  + " ".join(f"{e - E.min():.2e}" for e in E[::2]), flush=True)


if __name__ == "__main__":
    main()

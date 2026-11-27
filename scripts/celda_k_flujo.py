"""Respuesta al flujo de un testigo con orden clásico de celda k = 3 (exploración).

Espín 1 en el anillo Z_n, s_i in {0, 1, 2} con S^z = s - 1, y
    H(Theta) = -sum_i P_asc(i, i+1) + eps * sum_i (e^{i Theta/n} S+_i S-_{i+1} + h.c.),
con P_asc el proyector sobre (s, s+1 mod 3): el orden clásico ...0 1 2 0 1 2... tiene celda 3 y es
triplemente degenerado si 3 | n; si no, el anillo fuerza un defecto. La pregunta es cómo escala con n
la respuesta al flujo R_n = max_Theta E_0 - min_Theta E_0 según n mod 3:
  - 3 | n: se espera túnel entre los tres órdenes, exponencialmente pequeño;
  - 3 no divide a n: un defecto móvil de carga fraccionaria, R_n ~ c / n^2.

Uso: python scripts/celda_k_flujo.py [eps ...]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402


def sector(n, Sz):
    codes = np.arange(3**n, dtype=np.int64)
    dig = np.empty((n, codes.size), dtype=np.int8)
    c = codes.copy()
    for i in range(n):
        dig[i] = c % 3
        c //= 3
    keep = dig.sum(0) - n == Sz
    return codes[keep], dig[:, keep]


def build(n, Sz):
    codes, dig = sector(n, Sz)
    N = codes.size
    diag = np.zeros(N)
    rows, cols, bond = [], [], []
    pw = 3 ** np.arange(n, dtype=np.int64)
    for i in range(n):
        j = (i + 1) % n
        a, b = dig[i], dig[j]
        diag -= ((b - a) % 3 == 1)
        # S+_i S-_{i+1}: s_i -> s_i + 1, s_j -> s_j - 1, amplitud 2 en espín 1
        m = (a < 2) & (b > 0)
        new = codes[m] + pw[i] - pw[j]
        rows.append(np.searchsorted(codes, new))
        cols.append(np.nonzero(m)[0])
        bond.append(np.full(m.sum(), i))
    return N, diag, np.concatenate(rows), np.concatenate(cols)


def e0(n, eps, theta, cache={}):
    best = np.inf
    for Sz in range(-2, 3):
        key = (n, Sz)
        if key not in cache:
            cache[key] = build(n, Sz)
        N, diag, r, c = cache[key]
        if N == 0:
            continue
        ph = np.exp(1j * theta / n)
        A = sp.coo_matrix((2 * eps * ph * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
        H = A + A.getH() + sp.diags(diag)
        if N < 400:
            w = np.linalg.eigvalsh(H.toarray())[0]
        else:
            w = eigsh(H, k=1, which="SA", tol=1e-12)[0][0]
        best = min(best, w)
    return best


def main():
    epss = [float(x) for x in sys.argv[1:]] or [0.1, 0.2]
    thetas = np.linspace(0, 2 * np.pi, 9)[:-1]
    for eps in epss:
        print(f"eps = {eps}")
        print(f"{'n':>3} {'n%3':>3} {'R_n':>12} {'n^2 R_n':>10}  argmin/pi  E0(Theta)-E0min (Theta = 0, pi/2, pi, 3pi/2)")
        for n in range(5, int(os.environ.get("NMAX", 12)) + 1):
            E = np.array([e0(n, eps, t) for t in thetas])
            R = E.max() - E.min()
            sel = [0, 2, 4, 6]
            print(f"{n:>3} {n % 3:>3} {R:12.4e} {n * n * R:10.4f}  {thetas[E.argmin()] / np.pi:5.2f}   "
                  + " ".join(f"{E[s] - E.min():.2e}" for s in sel), flush=True)


if __name__ == "__main__":
    main()

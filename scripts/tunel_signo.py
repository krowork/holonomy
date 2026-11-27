"""Túnel del multiplete y signo de b_0 (sensibilidad_celda3.tex, Proposición del túnel y Corolario de b_0).

1. Túnel. Suma, con fracciones, los caminos de saltos de c_0 a c_1 sin pasar por patrones, orden a orden, hasta
   el primero no nulo. Comprueba que es 2n/3 y que el coeficiente
       T_n = sum_{caminos de 2n/3 saltos} prod (amplitud 2) prod_{intermedios} 1/(-n - E_cl)
   es negativo. Lo compara con la separación por ED: lambda_{±2pi/3} - lambda_0 = 3|T_n| eps^{2n/3} + ...,
   y comprueba que el fundamental del sector S^z = 0 tiene momento 0.
2. b_0. Con orden_theta.B, los B(c_0 -> c_d) de n saltos directos tienen todos el signo (-1)^{n-1}.

Uso: python scripts/tunel_signo.py [n_max_ed] [n_max_b0]   (por defecto 12 y 15)
"""
import os
import sys
from collections import defaultdict
from fractions import Fraction

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from celda_k_flujo import build, sector  # noqa: E402
from orden_theta import B, energia  # noqa: E402


def saltos(s):
    """Saltos en los dos sentidos (S+_j S-_{j+1} y su inverso), amplitud 2 a Theta = 0."""
    n = len(s)
    for j in range(n):
        k = (j + 1) % n
        for da in (1, -1):
            a, b = s[j] + da, s[k] - da
            if 0 <= a <= 2 and 0 <= b <= 2:
                t = list(s)
                t[j], t[k] = a, b
                yield tuple(t)


def patrones(n):
    return [tuple((c + j) % 3 for j in range(n)) for c in range(3)]


def tunel(n):
    """Primer orden m y coeficientes de c_0 -> c_{±1} (caminos sin pasar por patrones)."""
    pats = patrones(n)
    P = set(pats)
    frente = {pats[0]: Fraction(1)}
    m = 0
    while True:
        m += 1
        nuevo = defaultdict(Fraction)
        for s, w in frente.items():
            for t in saltos(s):
                nuevo[t] += 2 * w
        llega = {pats.index(t): v for t, v in nuevo.items() if t in P and t != pats[0]}
        if llega:
            return m, llega
        frente = {t: w / (-n - energia(t)) for t, w in nuevo.items() if t not in P}


def ed_separacion(n, eps):
    N, diag, r, c = build(n, 0)
    A = sp.coo_matrix((2 * eps * np.ones(r.size), (r, c)), shape=(N, N)).tocsr()
    H = A + A.T + sp.diags(diag)
    if N < 400:
        w, v = np.linalg.eigh(H.toarray())
    else:
        w, v = eigsh(H, k=3, which="SA", tol=1e-13)
    o = np.argsort(w)
    w, v = w[o][:3], v[:, o][:, :3]
    codes, dig = sector(n, 0)
    pw = 3 ** np.arange(n, dtype=np.int64)
    idx = np.searchsorted(codes, (np.roll(dig, 1, axis=0).astype(np.int64) * pw[:, None]).sum(0))
    Tpsi = np.zeros_like(v[:, 0])
    Tpsi[idx] = v[:, 0]
    return w[1] - w[0], float(v[:, 0] @ Tpsi)


if __name__ == "__main__":
    n_ed = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_b0 = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    print("1. Túnel: primer orden y coeficiente T_n (c_0 -> c_{±1})")
    for n in range(6, n_ed + 1, 3):
        m, llega = tunel(n)
        T = llega[1]
        print(f"  n={n}: orden {m} (2n/3 = {2 * n // 3}), T_n = {T} = {float(T):.6g}, "
              f"c_0 -> c_-1: {llega[2]}; 3|T_n| = {3 * abs(float(T)):.6g}")
        for eps in (0.01, 0.02, 0.05):
            gap, mom = ed_separacion(n, eps)
            print(f"     ED eps={eps}: (lambda_1 - lambda_0)/eps^{m} = {gap / eps**m:.6g}, <T> del fundamental = {mom:.4f}")
    print("2. b_0 = sum_d B(c_0 -> c_d), caminos de n saltos directos")
    for n in range(6, n_b0 + 1, 3):
        pats = patrones(n)
        res = {pats.index(t): v for t, v in B(n, pats[0], set(pats)).items()}
        b0 = sum(res.values())
        signos = {d: (1 if v > 0 else -1) for d, v in res.items()}
        print(f"  n={n}: B = {{{', '.join(f'{d}: {res[d]}' for d in sorted(res))}}}, signos {signos} "
              f"(esperado {(-1) ** (n - 1):+d}), 4|b_0| = {4 * abs(float(b0)):.6g}")

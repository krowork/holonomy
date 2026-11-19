"""Néel con n impar: v(n,0), v(n,pi), Delta*, cota del Corolario de paridad y
escalado de la energía de giro. Uso: python scripts/neel_impar.py kappa n1 n2 ...

Por el Lema de reducción, en Theta0 in {0, pi} basta comparar tres testigos:
alineado (flujo Theta0, denominador Delta + 2k), antialineado (flujo Theta0 + n pi,
denominador Delta - 2k) e Ising (v_I = (n-2)/n).
"""
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from holonomy import xxz


def E0(n, D, Phi):
    # n impar: los sectores M = n//2 y n//2+1 son degenerados por inversión de espines,
    # así que basta uno; se compara con el ferromagnético por si D cae cerca de -1.
    return min(xxz.sector_spectrum(n, n // 2, D, Phi)[0], n * D / 4)


def rama(n, k, Theta, sgn):
    Phi = Theta + (0 if sgn > 0 else n * np.pi)
    lo = -sgn * 2 * k + 1e-6
    f = lambda D: 4 * (-E0(n, D, Phi) / n) / (D + sgn * 2 * k)
    grid = np.concatenate([np.linspace(lo + 0.05, lo + 6, 20),
                           lo + np.array([9, 14, 25, 45, 80, 200])])
    vals = [f(D) for D in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-9})
    return (r.fun, r.x) if r.fun < vals[i] else (vals[i], grid[i])


k = float(sys.argv[1])
for n in map(int, sys.argv[2:]):
    vI = (n - 2) / n
    res = {}
    for Theta in (0.0, np.pi):
        al, an = rama(n, k, Theta, +1), rama(n, k, Theta, -1)
        res[Theta] = (min(al[0], an[0], vI), al, an)
    vpi, al, an = res[np.pi]
    v0 = res[0.0][0]
    D = al[1]
    dE = E0(n, D, 0.0) - E0(n, D, np.pi)
    b = 4 * dE / (n * (D + 2 * k))
    gana = "al" if vpi == al[0] else ("an" if vpi == an[0] else "Ising")
    print(f"n={n:2d} v(pi)={vpi:.8f} [{gana}] v(0)={v0:.8f} D*={D:.4f} "
          f"an-vI={an[0]-vI:+.4f} n2*dv={n*n*(vpi-v0):.4f} n2*cota={n*n*b:.4f} "
          f"n*dE={n*dE:.4f} n2*dE={n*n*dE:.3f}", flush=True)

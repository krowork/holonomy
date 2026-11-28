"""Optimización de los parámetros del polidisco (eps_a, R_h, R_m, R_s, w_m, w_s) de kantorovich_series.py
(datos autoconsistentes) con objetivo eps*: el mayor eps tal que, en [eps, 1,01 eps], valen h <= 1/2 y
cota de delta_n < principal/2, con principal = 4|b0| eps^n / (n (e_bar - e_R)).

Para el principal se usa e_bar - e_R <= 2/3 + 2 c2 eps^2 (c2 = |coef. de eps^2 de e_R|, de la serie), que lo
rebaja: la comparación queda del lado seguro. El borde se busca por bisección en log eps; al final se recorre
todo [1e-30, eps*] en intervalos de razón 1,01 (Kantorovich en todos; déficit < principal/2 en todos los que el
principal no desborda la coma flotante, 1e-25 en adelante; por debajo el cociente déficit/principal ~ eps^(n-8)
solo baja). Nelder-Mead en log de los parámetros, desde varios puntos de partida.
Uso: python scripts/optimiza_principal.py serie_n.npz [4|b0|]   (4|b0| por defecto: n=6, 9, 12 -> 62,314, 237,28, 1001,08)
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import kantorovich_series as KS  # noqa: E402
sys.argv = _argv

B4 = {6: 62.314, 9: 237.28, 12: 1001.08}
# intervalo de búsqueda del borde (log10 eps) y puntos de partida (los de la rejilla gruesa primero)
RANGO = {6: (-5.5, -2.5), 9: (-3.0, -1.0), 12: (-3.5, -1.8)}
STARTS = {6: [(0.015, 1e-4, 8e-3, 0.4, 1, 1), (0.015, 1e-4, 8e-3, 0.4, 0.4, 0.4), (0.01, 3e-4, 1e-2, 0.2, 1, 1)],
          9: [(0.03, 1e-4, 4e-3, 0.1, 1, 1), (0.03, 1e-4, 4e-3, 0.1, 0.4, 0.4), (0.04, 5e-5, 2e-3, 0.15, 1, 1)],
          12: [(0.02, 2e-5, 1e-3, 0.1, 1, 1), (0.02, 2e-5, 1e-3, 0.1, 0.3, 0.3), (0.015, 5e-5, 2e-3, 0.15, 1, 1)]}


def principal(n, b4, c2, e):
    return b4 * e ** n / (n * (2 / 3 + 2 * c2 * e ** 2))


def vale(S, P, par, el, er, b4, c2, deficit=True):
    c = KS.cotas(S, P, el, er, par[0], par[1:4], np.array([1, par[4], par[5]]))
    if c is None or c['h'] > 0.5:
        return False, c
    if deficit and c['deficit'] >= 0.5 * principal(S['n'], b4, c2, el):
        return False, c
    return True, c


def borde(S, P, par, b4, c2, lo=-3.5, hi=-1.8, it=30):
    ok = lambda x: vale(S, P, par, 10 ** x, 1.01 * 10 ** x, b4, c2)[0]
    if not ok(lo):
        return lo - 1
    if ok(hi):
        return hi
    for _ in range(it):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return lo


def rango_fino(S, P, par, b4, c2, e0=1e-30, paso=1.01):
    """Mayor eps tal que todos los intervalos de [e0, eps] cumplen h <= 1/2 y (desde 1e-25) déficit < principal/2."""
    el, ult = e0, None
    while el < 0.2:
        er = el * paso
        ok, c = vale(S, P, par, el, er, b4, c2, deficit=el >= 1e-25)
        if not ok:
            motivo = 'h' if c is None or c['h'] > 0.5 else 'deficit'
            return el, ult, motivo
        ult = c
        el = er
    return el, ult, '-'


if __name__ == "__main__":
    S = KS.carga(sys.argv[1])
    P = KS.preparar(S)
    n = S['n']
    b4 = float(sys.argv[2]) if len(sys.argv) > 2 else B4[n]
    c2 = abs(np.real(S['E'][0, 2]))
    print(f"n={n}: 4|b0| = {b4}, c2 = {c2:.4f}", flush=True)
    starts = STARTS[n]
    lo, hi = RANGO[n]
    f = lambda z: -borde(S, P, np.exp(z), b4, c2, lo, hi)
    best = None
    for p0 in starts:
        r = minimize(f, np.log(np.array(p0, float)), method='Nelder-Mead',
                     options=dict(xatol=1e-3, fatol=1e-3, maxfev=300))
        print(f"  desde {p0}: eps* ~ 10^{-r.fun:.3f} con {tuple(float(x) for x in np.round(np.exp(r.x), 6))}  "
              f"({r.nfev} evaluaciones)", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    par = tuple(float(x) for x in np.exp(best.x))
    print("PARAMS " + " ".join(f"{x:.6g}" for x in par), flush=True)
    emax, ult, motivo = rango_fino(S, P, par, b4, c2)
    print(f"  rango fino (razón 1,01 desde 1e-30): VALE en [1e-30, {emax:.3e}]; en el borde h = {ult['h']:.3f}, "
          f"déficit <= {ult['deficit']:.1e}, principal = {principal(n, b4, c2, emax):.1e}; falla por {motivo}")
    # Kantorovich solo, con los mismos parámetros
    ek, uk, _ = KS.rango(S, P, par[0], par[1:4], np.array([1, par[4], par[5]]), e0=emax, paso=1.01)
    print(f"  Kantorovich solo (mismos parámetros): VALE hasta {ek:.3e}")

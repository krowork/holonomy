"""Optimización de los parámetros del polidisco (eps_a, R_h, R_m, R_s, w_m, w_s) de kantorovich_datos.py.

Dos objetivos, por separado:
  kant  el borde de Kantorovich (mayor eps con h <= 1/2 en el intervalo [eps, 1,01 eps]);
  sens  el borde de la sensibilidad (mayor eps con |Delta delta| < principal/2 en ese intervalo).
El borde superior se busca por bisección en log eps (Kantorovich vale en [0, borde]; la sensibilidad, en una ventana);
el rango completo desde 1e-30 se comprueba después con kantorovich_datos.rango. Nelder-Mead en log de los
parámetros, desde varios puntos de partida (los de la rejilla y los del .tex), y al final el rango fino.
Uso: python scripts/optimiza_parametros.py serie_n.npz datos.npz [inf] kant|sens
"""
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv, sys.argv = sys.argv, sys.argv[:1]
import kantorovich_series as KS  # noqa: E402
import kantorovich_datos as KD  # noqa: E402
sys.argv = _argv


def vale(S, P, D, par, e, obj):
    c = KD.cotas(S, P, D, e, e * 1.01, par[0], par[1:4], np.array([1, par[4], par[5]]))
    if c is None or c['h'] > 0.5:
        return False
    if obj == 'sens':
        return 'principal' in c and c['ddelta'] < 0.5 * c['principal']
    return True


def borde(S, P, D, par, obj, lo=-30.0, hi=-1.0, it=40):
    """log10 del borde superior por bisección. Para 'kant' parte de 1e-30; para 'sens', que vale en una ventana,
    del punto más alto de una rejilla gruesa (paso 0,5 en log eps, de 10^-1,5 a 10^-25,5) en que vale."""
    if obj == 'sens':
        # de arriba abajo: el primer punto en que vale es el más alto
        lo = next((x for x in np.arange(-1.5, -26.0, -0.5) if vale(S, P, D, par, 10 ** x, obj)), None)
        if lo is None:
            return -31.0
        hi = lo + 0.5
    elif not vale(S, P, D, par, 10 ** lo, obj):
        return lo - 1
    if vale(S, P, D, par, 10 ** hi, obj):
        return hi
    for _ in range(it):
        mid = (lo + hi) / 2
        if vale(S, P, D, par, 10 ** mid, obj):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return lo


if __name__ == "__main__":
    S = KS.carga(sys.argv[1])
    P = KS.preparar(S)
    inf = 'inf' in sys.argv[3:]
    obj = sys.argv[-1]
    D = KD.datos(sys.argv[2], S['K'], S['n'], inf=inf)
    n = S['n']
    starts = {9: [(0.03, 1e-4, 4e-3, 0.1, 1, 1), (0.0225, 1e-4, 4e-3, 0.15, 1, 1), (0.015, 5e-5, 2e-3, 0.15, 1, 1)],
              12: [(0.02, 2e-5, 1e-3, 0.1, 0.3, 0.3), (0.015, 4e-5, 2e-3, 0.15, 0.9, 0.9)]}[n]
    f = lambda z: -borde(S, P, D, np.exp(z), obj)
    best = None
    for p0 in starts:
        z0 = np.log(np.array(p0, float))
        r = minimize(f, z0, method='Nelder-Mead',
                     options=dict(xatol=1e-3, fatol=1e-3, maxfev=400, initial_simplex=None))
        print(f"  desde {p0}: borde 10^{-r.fun:.3f} con {tuple(np.round(np.exp(r.x), 6))}  ({r.nfev} evaluaciones)",
              flush=True)
        if best is None or r.fun < best.fun:
            best = r
    par = tuple(float(x) for x in np.exp(best.x))
    print(f"n={n}, datos {'inf' if inf else D['N']}, objetivo {obj}: óptimo {par}, borde ~ {10 ** -best.fun:.3e}")
    emax, ult, fallo, sens = KD.rango(S, P, D, par)
    print(f"  rango fino: Kantorovich VALE en [1e-30, {emax:.3e}]; déficit en el borde <= {ult['deficit']:.1e}; "
          f"sensibilidad en {'[%.3e, %.3e]' % sens if sens else 'ningún intervalo'}")
    print("PARAMS " + " ".join(f"{x:.6g}" for x in par))

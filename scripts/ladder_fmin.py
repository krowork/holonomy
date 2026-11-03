"""Mínimo de v(f, psi) en la escalera periódica 2x4 y 2x5 (exploración).

scripts/ladder_flux.py afina el mínimo en f con psi = 0 pero no lo guarda, y
los mapas de scripts/ladder_psi.py tienen paso pi/20 en f. Aquí, desde el
menor punto de la malla de data/ladder_flux.csv (paso pi/32, psi = 0), se
minimiza por coordenadas: Brent en f con psi fijo y Brent en psi con f fijo
(más los extremos 0 y pi y una malla de 21 valores de psi, porque el mínimo
en psi puede estar en el borde), hasta que v cambie menos de 1e-10. Todo con
el dual reducido como SDP (LadderDual.v_dual_sdp), como en ladder_psi.py.
Comprobaciones:
  1. el mínimo no supera el menor punto de la malla de ladder_flux.csv;
  2. el mínimo no supera el menor punto del mapa en psi de ese L.

Escribe data/ladder_fmin.csv.

Con el argumento perfil, solo el perfil de la 2x4 alrededor de (2pi/3, 0), que
fija el mínimo por debajo del ruido del solver (~5e-9, en el que Brent se mueve):
  3. la asimetría v(2pi/3 + d) - v(2pi/3 - d) escala como d^3 (de d = 0,003 pi a
     0,01 pi crece entre 15 y 100 veces; con pendiente no nula crecería ~3,3):
     el mínimo en f está en 2pi/3;
  4. en f = 2pi/3, v(psi) - v(0) crece como psi^4 (de psi = 0,05 pi a 0,1 pi,
     entre 8 y 32 veces; ~4 si fuera cuadrático): el mínimo en psi está en 0.
Escribe data/ladder_fmin_perfil_L4.csv.

Uso: (ulimit -v 12000000; python scripts/ladder_fmin.py [L ...])   por defecto 4 5
     (ulimit -v 12000000; python scripts/ladder_fmin.py perfil)     ~2 min
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import csv
import sys

import numpy as np
from scipy.optimize import minimize_scalar

from holonomy.report import DATA, write_csv
from ladder_psi import solve

MAPS = {4: "ladder_psi_L4.csv", 5: "ladder_psi_L5_11x41.csv"}
STEP = np.pi / 32


def grid_min(name, L=None):
    rows = list(csv.DictReader(open(DATA / name)))
    if L is not None:
        rows = [r for r in rows if int(r["L"]) == L]
    r = min(rows, key=lambda r: float(r["v_max"]))
    return float(r["f"]), float(r.get("psi", 0.0)), float(r["v_max"])


def profile():
    chk = Checker("ladder_fmin perfil")
    f0 = 2 * np.pi / 3
    v0 = solve(4, f0, 0.0)[0]
    rows = [("f", 0.0, v0)]
    asym = {}
    for d in (0.003, 0.01):
        vp, vm = solve(4, f0 + d * np.pi, 0.0)[0], solve(4, f0 - d * np.pi, 0.0)[0]
        rows += [("f", d, vp), ("f", -d, vm)]
        asym[d] = vp - vm
        chk.info(f"f/pi = 2/3 +- {d}: v - v0 = {vp - v0:.3e}, {vm - v0:.3e}; asimetría {vp - vm:+.2e}")
    r = asym[0.01] / asym[0.003]
    chk.check(15 < r < 100, f"asimetría en f: cociente {r:.1f} (cúbico ~37, lineal ~3,3)")
    rise = {}
    for p in (0.02, 0.05, 0.1):
        vp = solve(4, f0, p * np.pi)[0]
        rows.append(("psi", p, vp))
        rise[p] = vp - v0
        chk.info(f"f = 2pi/3, psi/pi = {p}: v - v0 = {vp - v0:.3e}")
    r = rise[0.1] / rise[0.05]
    chk.check(8 < r < 32, f"subida en psi: cociente {r:.1f} (cuártico ~16, cuadrático ~4)")
    chk.info(f"v(4, 2pi/3, 0) = {v0:.10f}")
    write_csv("ladder_fmin_perfil_L4.csv", ["eje", "desplazamiento_sobre_pi", "v_max"], rows)
    chk.finish()


def main():
    if sys.argv[1:] == ["perfil"]:
        return profile()
    Ls = [int(a) for a in sys.argv[1:]] or [4, 5]
    chk = Checker("ladder_fmin")
    out = []
    for L in Ls:
        f, _, vg = grid_min("ladder_flux.csv", L)
        fm, pm, vm = grid_min(MAPS[L])
        psi, v = 0.0, solve(L, f, 0.0)[0]
        for it in range(20):
            res = minimize_scalar(lambda x: solve(L, x, psi)[0], bounds=(max(f - STEP, 0), min(f + STEP, np.pi)),
                                  method="bounded", options=dict(xatol=1e-7))
            if res.fun < v:
                f, v = res.x, res.fun
            grid = [(solve(L, f, p)[0], p) for p in np.linspace(0, np.pi, 21)]
            vb, pb = min(grid)
            h = np.pi / 20
            res = minimize_scalar(lambda x: solve(L, f, x)[0], bounds=(max(pb - h, 0), min(pb + h, np.pi)),
                                  method="bounded", options=dict(xatol=1e-7))
            vn, pn = min((vb, pb), (res.fun, res.x))
            change = v - vn
            if vn < v:
                v, psi = vn, pn
            chk.info(f"2x{L} vuelta {it}: f/pi = {f / np.pi:.6f}, psi/pi = {psi / np.pi:.6f}, v = {v:.10f}")
            if change < 1e-10:
                break
        chk.check(v <= vg + 1e-9, f"2x{L}: mínimo {v:.10f} <= malla de ladder_flux.csv {vg:.10f} "
                  f"(f/pi = {f / np.pi:.5f})")
        chk.check(v <= vm + 1e-9, f"2x{L}: mínimo {v:.10f} <= mapa en psi {vm:.10f} "
                  f"(f/pi = {fm / np.pi:.3f}, psi/pi = {pm / np.pi:.3f})")
        out.append((L, f, psi, v))
    write_csv("ladder_fmin.csv", ["L", "f", "psi", "v_min"], out)
    chk.finish()


if __name__ == "__main__":
    main()

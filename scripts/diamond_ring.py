"""Cadena cerrada de k diamantes: mapas v(f, psi) (exploración).

Sitios y aristas de holonomy.diamond.chain_edges(k, periodic=True): 3k
sitios, 5k aristas. Datos abelianos uniformes en el gauge simétrico
(holonomy.diamond.abelian_phases): los mismos flujos de triángulo fA, fB en
todas las celdas y psi, el ciclo extra de la vuelta completa. Las holonomías
de las dos vueltas, por los u y por los w, son psi +- k (fA + fB) / 2.
Dos modos: "igual" (fA = fB = f) y "opuesto" (fA = -fB = f; las vueltas
tienen holonomía psi). Malla nf x npsi en [0, pi] x [0, pi]: basta, porque
la reflexión u <-> w compuesta con la conjugación compleja da
v(f, psi) = v(f, -psi) y la conjugación sola v(f, psi) = v(-f, -psi).

Dual reducido holonomy.diamond.DiamondDual (U(1), traslación de una celda,
inversión temporal; 15 coeficientes), con half (solo los bloques k <= n//2)
y resuelto por generación de columnas (v_dual_cg con r0 = R0, en frío en
cada punto; cotas superior e inferior) en todos los k. Solo el estado
'optimal' es un resultado: cualquier otro es un FALLO y su v se escribe
como nan. En cada fila de f se refina el mínimo en psi
(el mayor déficit 1/3 - v) por Brent alrededor del menor punto de la fila.

Comprobaciones:
  1. v(f, psi) = v(f, -psi) a 1e-7 en 3 puntos de cada modo;
  2. v <= 1/3 + 1e-7 en todos los puntos (cota del diamante simple; 1e-7 y
     no 1e-8 por el ruido de v_dual_cg en la meseta v = 1/3, ver TOL_BOUND);
  3. v = 1/3 a 1e-7 sin holonomía (f = 0, psi = 0). Falla con k = 3:
     v(0, psi) = 0.3077932 para todo psi, confirmado con un primal reducido
     por U(1) y traslación (cota inferior; coincide con el dual a 1e-8). Se
     deja como FALLO: es un resultado, no un error del cálculo;
  4. todas las evaluaciones con estado 'optimal'.

Cada evaluación se guarda en data/diamond_ring_L<k>_evals.csv al terminarla
y no se repite al relanzar (reanudable). Escribe
data/diamond_ring_L<k>_<modo>.csv (malla) y
data/diamond_ring_min_L<k>_<modo>.csv (mínimo en psi de cada fila).

Uso: (ulimit -v 12000000; /usr/bin/time -f "%M KB %e s" python scripts/diamond_ring.py k nf npsi)
     por defecto k = 2, nf = npsi = 13.
Medido (13 x 13, 2026-11-03, /usr/bin/time, OMP_NUM_THREADS = 2):
  k = 2: 846 evaluaciones, 130 s, pico 144 MB;
  k = 3: 5081 s (~85 min, ~6 s por evaluación), pico 673 MB;
  k = 4: en segundo plano (ver el informe de la tarea 6).
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import csv
import sys

import numpy as np
from scipy.optimize import minimize_scalar

from holonomy import diamond
from holonomy.report import DATA, write_csv

THIRD = 1 / 3
TOL_SYM = 1e-7
# Cota v <= 1/3: el informe de la tarea pedía 1e-8, pero en la meseta v = 1/3
# el v de v_dual_cg con estado 'optimal' sale hasta 3,1e-8 por encima (ruido
# de Clarabel en los SDP restringidos, ver R0; k = 2, 13 x 13). Se usa el
# ruido tolerado por la propia generación de columnas, PeriodicDual.GAP_NEG.
TOL_BOUND = 1e-7
GAP_TOL = 1e-6
# Subespacio inicial de v_dual_cg, siempre en frío (desde heisenberg_y0). En
# la meseta v = 1/3 de k = 2 los SDP restringidos acaban 'optimal_inaccurate'
# y la cota inferior (y con ella v) sale por encima de 1/3 con estado
# 'optimal': hasta 4e-8 con r0 = 6 en frío y hasta 1,7e-8 con r0 = 12 desde
# el testigo del punto anterior; con r0 = 12 en frío, ~3e-9 (medido en
# (f, psi) = (pi, -pi/2) opuesto y (5 pi/6, pi/3) igual).
R0 = 12
MODES = {"igual": 1.0, "opuesto": -1.0}  # fB = signo * fA

_D = {}
_cache = {}


def wrap(a):
    """Ángulo reducido a [0, pi] (el ángulo de una rotación R_z)."""
    return abs(float(np.angle(np.exp(1j * a))))


def dual(k):
    if k not in _D:
        _D[k] = diamond.DiamondDual(k, half=True)
    return _D[k]


def _key(mode, f, psi):
    return (mode, round(float(f), 12), round(float(psi), 12))


def _evals_path(k):
    return DATA / f"diamond_ring_L{k}_evals.csv"


def load_cache(k):
    path = _evals_path(k)
    if path.exists():
        for r in csv.DictReader(open(path)):
            y = np.array([float(x) for x in r["y"].split()]) if r["y"] else None
            _cache[_key(r["modo"], r["f"], r["psi"])] = (float(r["v_max"]), y, r["estado"])


def solve(k, mode, f, psi):
    """(v, y, estado) en (fA, fB) = (f, signo * f) y psi; v = nan si el estado
    no es 'optimal'. Cada evaluación se guarda en la caché en disco."""
    key = _key(mode, f, psi)
    if key not in _cache:
        ph = diamond.abelian_phases(k, f, MODES[mode] * f, periodic=True, psi=psi)
        v, y, st, _ = dual(k).v_dual_cg(ph, r0=R0, **CLARABEL_OPTS)
        if st != "optimal":
            v = np.nan
        _cache[key] = (float(v), np.asarray(y), st)
        path = _evals_path(k)
        new = not path.exists()
        DATA.mkdir(exist_ok=True)
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["modo", "f", "psi", "v_max", "estado", "y"])
            w.writerow([mode, repr(float(f)), repr(float(psi)), repr(_cache[key][0]), st,
                        " ".join(repr(float(c)) for c in y)])
    return _cache[key]


def row(k, mode, f, psis):
    """Una fila de f: puntos de la malla en psi y mínimo refinado en psi."""
    D = dual(k)
    sgn = MODES[mode]
    out = []
    for psi in psis:
        v, yn, st = solve(k, mode, f, psi)
        top = ""
        if st == "optimal":
            gaps = D.block_gaps(yn, D.labels)
            top = " ".join(f"{a}:{b}" for (a, b), g in sorted(zip(D.labels, gaps)) if g < GAP_TOL)
        S = k * (f + sgn * f) / 2
        out.append((f, psi, f, sgn * f, v, THIRD - v, wrap(psi + S), wrap(psi - S), top, st))
    vals = np.array([r[4] for r in out])
    if np.all(np.isnan(vals)):
        return out, (f, np.nan, np.nan, np.nan, "sin_puntos")
    j = int(np.nanargmin(vals))
    lo, hi = psis[max(j - 1, 0)], psis[min(j + 1, len(psis) - 1)]
    fails = []

    def g(p):
        v, _, st = solve(k, mode, f, p)
        if st != "optimal":
            fails.append(st)
            return np.inf
        return v

    res = minimize_scalar(g, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-5))
    # Brent no evalúa los extremos: el mínimo de la malla gana si es menor.
    pm, vm = (res.x, res.fun) if res.fun < vals[j] else (psis[j], vals[j])
    return out, (f, pm, vm, THIRD - vm, "optimal" if not fails else "fallos:" + ",".join(fails))


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    npsi = int(sys.argv[3]) if len(sys.argv) > 3 else nf
    chk = Checker(f"diamond_ring k={k}")
    load_cache(k)
    D = dual(k)
    chk.info(f"{D.n} sitios, {len(D.blocks)} bloques (half), dimensión máxima "
             f"{max(D._dim(r) for r in D.blocks)}; {len(_cache)} evaluaciones en caché")
    fs, psis = np.linspace(0, np.pi, nf), np.linspace(0, np.pi, npsi)

    print("== Simetría psi -> -psi")
    for mode in MODES:
        for f, psi in ((0.3 * np.pi, 0.4 * np.pi), (0.7 * np.pi, 0.9 * np.pi), (1.0 * np.pi, 0.5 * np.pi)):
            (v1, _, s1), (v2, _, s2) = solve(k, mode, f, psi), solve(k, mode, f, -psi)
            ok = s1 == s2 == "optimal" and abs(v1 - v2) < TOL_SYM
            chk.check(ok, f"{mode}: f/pi={f / np.pi:.2f} psi/pi={psi / np.pi:.2f}: v = {v1:.9f}, "
                      f"|dif| = {abs(v1 - v2):.1e} [estados {s1}, {s2}]")

    for mode in MODES:
        print(f"== Malla, modo {mode} (fA = f, fB = {'' if MODES[mode] > 0 else '-'}f)")
        rows, mins = [], []
        for f in fs:
            out, mn = row(k, mode, f, psis)
            rows += out
            mins.append(mn)
            chk.info(f"f/pi={f / np.pi:.3f}: v(psi) = " + " ".join(f"{r[4]:.5f}" for r in out))
            chk.info(f"   mínimo en psi/pi = {mn[1] / np.pi:.5f}, v = {mn[2]:.9f}, "
                     f"déficit = {mn[3]:.3e} [{mn[4]}]")
        bad = [r for r in rows if r[9] != "optimal"]
        for r in bad:
            chk.check(False, f"f/pi={r[0] / np.pi:.3f} psi/pi={r[1] / np.pi:.3f}: estado {r[9]}")
        chk.check(not bad, f"{len(rows)} puntos con estado 'optimal' ({len(bad)} fallos)")
        badmin = [m for m in mins if m[4] != "optimal"]
        chk.check(not badmin, f"{len(mins)} refinamientos de Brent sin fallos ({len(badmin)} con fallos)")
        good = [r for r in rows if r[9] == "optimal"]
        worst = max(r[4] for r in good)
        chk.check(worst <= THIRD + TOL_BOUND, f"v <= 1/3 + {TOL_BOUND:.0e} en los {len(good)} puntos: máx v - 1/3 = "
                  f"{worst - THIRD:.1e}")
        v00 = rows[0][4]
        chk.check(abs(v00 - THIRD) < 1e-7, f"sin holonomía (f = psi = 0): v = {v00:.10f}, "
                  f"|v - 1/3| = {abs(v00 - THIRD):.1e}")
        write_csv(f"diamond_ring_L{k}_{mode}.csv",
                  ["f", "psi", "fA", "fB", "v_max", "deficit", "vuelta_u", "vuelta_w", "bloques", "estado"],
                  rows)
        write_csv(f"diamond_ring_min_L{k}_{mode}.csv", ["f", "psi_min", "v_min", "deficit", "estado"], mins)
        gm = [m for m in mins if m[4] == "optimal"]
        if gm:
            m = max(gm, key=lambda t: t[3])
            chk.info(f"{mode}: mayor déficit 1/3 - v = {m[3]:.9e} (v = {m[2]:.12f}) en f/pi = "
                     f"{m[0] / np.pi:.5f}, psi/pi = {m[1] / np.pi:.5f}")
    chk.finish()


if __name__ == "__main__":
    main()

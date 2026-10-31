"""Límite L -> infinito de la escalera periódica en f = pi (exploración).

El dual es min_y lambda_max(W_L(y)) / N_L(y), con W_L un hamiltoniano de
escalera invariante por traslación y lambda_max y N extensivos: v(L) es una
densidad de energía y debería tener un límite independiente del contorno. En
la escalera infinita las patas no son ciclos, así que psi (un twist de
contorno) no debería contar en el límite.

Para cada L, en f = pi y psi = 0, pi/4, pi/2, 3pi/4, pi, el dual reducido por
generación de columnas (LadderDual.v_dual_cg) con bloques dispersos de
dimensión > DENSE_MAX y Lanczos (eigsh): con L >= 8 los bloques densos no caben
en memoria. Comprobaciones:
  1. en cada punto las cotas inferior y superior difieren menos de 1e-8;
  2. con L = 4..7, coincide con los mapas ya calculados (SDP completo en L = 4,
     5; generación de columnas densa en L = 6, 7) a <= 1e-8;
  3. v(psi) = v(-psi) en psi = pi/4.
Se dan el mínimo en psi y la variación de v en psi para cada L.

Escribe data/ladder_limit_L<L>.csv para cada L, al terminarlo (un fallo en un
L no pierde los anteriores).

Uso: (ulimit -v 12000000; python scripts/ladder_limit.py [L ...])   por defecto 4..10
Cada punto se resuelve en frío en un proceso hijo nuevo (Pool con
maxtasksperchild=1): encadenados en un mismo proceso, la memoria llegó a ~10 GB
en L = 10, frente a ~4 GB de un punto aislado.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from holonomy import ladder
from holonomy.report import write_csv

TOL = 1e-8
DENSE_MAX = 100
GAP_TOL = 1e-5
PSIS = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * np.pi
MAPS = {4: "ladder_psi_L4.csv", 5: "ladder_psi_L5_11x41.csv",
        6: "ladder_psi_L6_21x41.csv", 7: "ladder_psi_L7_21x41.csv"}
DATA = Path(__file__).resolve().parent.parent / "data"


def from_map(L, psi):
    """v(f = pi, psi) del mapa ya calculado, o None."""
    if L not in MAPS or not (DATA / MAPS[L]).exists():
        return None
    for r in csv.DictReader(open(DATA / MAPS[L])):
        if abs(float(r["f"]) - np.pi) < 1e-9 and abs(float(r["psi"]) - psi) < 1e-9:
            return float(r["v_max"])
    return None


def point(args):
    """v en (f = pi, psi), en frío, con los bloques degenerados del testigo."""
    L, psi = args
    t0 = time.time()
    D = ladder.LadderDual(L, dense_max=DENSE_MAX)
    ph = ladder.abelian_phases(L, np.pi, periodic=True, psi=psi)
    v, y, st, gap = D.v_dual_cg(ph, **CLARABEL_OPTS)
    gaps = D.block_gaps(y, D.labels)
    top = " ".join(f"{k}:{q}" for (k, q), g in zip(D.labels, gaps) if g < GAP_TOL)
    dmax = max(r[("C", "b")].shape[0] for r in D.blocks)
    return v, st, gap, top, time.time() - t0, len(D.blocks), dmax


def main():
    Ls = [int(a) for a in sys.argv[1:]] or list(range(4, 11))
    chk = Checker("ladder_limit")
    pool = Pool(1, maxtasksperchild=1)
    for L in Ls:
        rows, vs = [], []
        for psi in PSIS:
            v, st, gap, top, dt, nb, dmax = pool.apply(point, ((L, psi),))
            if psi == 0:
                chk.info(f"2x{L}: {nb} bloques (máx. {dmax})")
            chk.check(st == "optimal" and gap < TOL, f"2x{L} psi/pi={psi / np.pi:.2f}: v = {v:.10f}, "
                      f"cotas a {gap:.1e} ({st}, {dt:.0f} s)")
            ref = from_map(L, psi)
            if ref is not None:
                chk.check(abs(v - ref) < TOL, f"   = mapa {MAPS[L]}: {ref:.10f}, |dif| = {abs(v - ref):.1e}")
            vs.append(v)
            rows.append((L, psi, v, gap, top))
        vm = pool.apply(point, ((L, -PSIS[1]),))[0]
        chk.check(abs(vm - vs[1]) < TOL, f"2x{L}: simetría psi -> -psi en pi/4, |dif| = {abs(vm - vs[1]):.1e}")
        j = int(np.argmin(vs))
        chk.info(f"2x{L}: mínimo en psi/pi = {PSIS[j] / np.pi:.2f}, v = {vs[j]:.10f}, "
                 f"variación en psi = {max(vs) - min(vs):.1e}")
        write_csv(f"ladder_limit_L{L}.csv", ["L", "psi", "v_max", "dif_cotas", "bloques"], rows)
    pool.close()
    chk.finish()


if __name__ == "__main__":
    main()

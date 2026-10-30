"""Pesos del estado óptimo simétrico de la escalera periódica 2x4 (exploración).

Con el primal reducido por U(1) y traslación (ladder.primal_symmetric), el
estado óptimo simétrico es una suma de bloques (k giros, momento q). Para
f/pi en [0,70, 0,85] se dan su rango y los pesos de cada bloque. Comprueba:
  - rango 2, con soporte en (4,0) y (4,2) (los bloques en lambda_max del
    testigo, por holgura complementaria); todo lo demás <= 1e-6;
  - el peso de (4,0) tiene un mínimo suave > 0 cerca de f = 0,775 pi: el
    bloque no llega a salir, y el cambio de pendiente del flujo del testigo
    alrededor de 0,76-0,77 pi (scripts/ladder_regimes.py) es un cruce suave,
    no un cambio de régimen.

Escribe data/ladder_weights_L4.csv.

Uso: (ulimit -v 12000000; python scripts/ladder_weights.py)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import numpy as np

from holonomy import ladder
from holonomy.report import write_csv

L = 4


def main():
    chk = Checker("ladder_weights")
    rows = []
    for f in np.arange(0.70, 0.8501, 0.0025):
        ph = ladder.abelian_phases(L, f * np.pi, periodic=True)
        v, st, blocks = ladder.primal_symmetric(L, ph, return_blocks=True, **CLARABEL_OPTS)
        w = {}
        rest = 0.0
        for lab, X in blocks:
            ev = np.linalg.eigvalsh((X + X.conj().T) / 2)
            if lab in ((L, 0), (L, 2)):
                w[lab] = ev[-1]
                rest = max(rest, ev[-2] if len(ev) > 1 else 0.0)
            else:
                rest = max(rest, ev[-1])
        rank = sum(x > 1e-6 for x in w.values())
        chk.check(rank == 2 and rest < 1e-6, f"f/pi={f:.4f}: v = {v:.9f}, peso (4,0) = {w[(L, 0)]:.5f}, "
                  f"(4,2) = {w[(L, 2)]:.5f}, resto <= {rest:.1e}")
        rows.append((f, v, w[(L, 0)], w[(L, 2)], rest))
    p0 = np.array([r[2] for r in rows])
    j = int(np.argmin(p0))
    chk.check(0 < j < len(rows) - 1 and p0[j] > 0.01,
              f"mínimo interior del peso de (4,0): {p0[j]:.5f} en f/pi = {rows[j][0]:.4f}")
    write_csv("ladder_weights_L4.csv", ["f_pi", "v_max", "peso_q0", "peso_q2", "resto_max"], rows)
    chk.finish()


if __name__ == "__main__":
    main()

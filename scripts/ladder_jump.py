"""Cambio de régimen del testigo en la escalera periódica 2x5 (exploración).

En el barrido de scripts/ladder_flux.py, el flujo de plaqueta del testigo
óptimo pasa de ~0,69 pi a ~0,84 pi entre f = 0,5 pi y 0,53 pi. Una bisección
con el flujo del testigo como indicador muestra que no es un salto: es
continuo, pero su pendiente cambia. El cambio coincide con el de los bloques
(k giros, momento q) que dan lambda_max del testigo óptimo
(LadderDual.top_blocks): a la izquierda, (5, 0), (5, 2) y (5, 3); a la
derecha, solo (5, 2) y (5, 3). Aquí:
  1. en una malla fina de f se guarda la distancia de cada bloque (5, q) a
     lambda_max del testigo óptimo; el ruido del solver es ~3e-8, así que los
     bordes de los regímenes solo se fijan a la resolución de la malla;
  2. se estima la pendiente del flujo del testigo en cada régimen.
Escribe data/ladder_jump.csv.
Todo con el dual reducido como SDP (LadderDual.v_dual_sdp).

Uso: (ulimit -v 12000000; python scripts/ladder_jump.py)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import numpy as np

from holonomy import ladder
from holonomy.report import write_csv

L = 5


def main():
    chk = Checker("ladder_jump")
    D = ladder.LadderDual(L)
    cache = {}
    cache_y = {}

    def at(f):
        if f not in cache:
            ph = ladder.abelian_phases(L, f, periodic=True)
            v, y, st = D.v_dual_sdp(ph, **CLARABEL_OPTS)
            wf, wl = ladder.witness_fluxes(y, L)
            tops = D.top_blocks(y)
            cache[f] = (v, wf, tops)
            cache_y[f] = y
            chk.info(f"f/pi={f / np.pi:.9f}: v = {v:.10f}, flujo del testigo/pi = {wf / np.pi:+.5f}, "
                     f"bloques en lambda_max = {[(lab, round(w - tops[0][1], 9)) for lab, w in tops]}"
                     f"{chk.status_note(st)}")
        return cache[f]

    # Malla fina: distancia de cada bloque (L, q) a lambda_max en el óptimo.
    # La bisección con una tolerancia fija no sirve: la separación del bloque
    # (5, 0) crece de forma continua desde el ruido del solver (~3e-8), y el
    # punto en que cruza la tolerancia depende de ella.
    rows = []
    for f in np.arange(0.490, 0.5501, 0.0025) * np.pi:
        v, wf, _ = at(f)
        y = cache_y[f]
        gaps = D.block_gaps(y, [(L, q) for q in range(L)])
        rows.append([f, v, wf] + gaps)
        chk.info(f"   f/pi={f / np.pi:.4f}: lambda_max - lambda_(5,q), q=0..4: "
                 + " ".join(f"{g:.1e}" for g in gaps))
    write_csv("ladder_jump.csv", ["f", "v_max", "flujo_testigo"] + [f"gap_q{q}" for q in range(L)], rows)
    for h in (1e-2,):
        for fc in (0.495 * np.pi, 0.545 * np.pi):
            hp = h * np.pi
            chk.info(f"f/pi={fc / np.pi:.3f}: d(flujo testigo)/df = {(at(fc + hp)[1] - at(fc - hp)[1]) / (2 * hp):.4f}")
    chk.finish()


if __name__ == "__main__":
    main()

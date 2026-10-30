"""Unicidad del testigo óptimo en la escalera periódica 2x4 (exploración).

En scripts/ladder_regimes.py, el flujo del testigo de la 2x4 cambia de
pendiente hacia f ~ 0,76 pi sin ningún cruce de niveles. Una explicación
posible es que el testigo óptimo deje de ser único y el solver elija uno de
varios. Para cada f se calcula v* con el dual-SDP y después el rango de cada
coeficiente del testigo en {N(y) = 1, lambda_max(W(y)) <= v* + eps}
(LadderDual.witness_spread) para eps = 1e-4, 1e-5 y 1e-6 (con 1e-7, cerca de la precisión de v*, el
solver falla). Si el testigo es
único, la anchura máxima tiende a 0 con eps (como eps si el mínimo es una
arista de lambda_max, como sqrt(eps) si es suave); si hay una cara de
óptimos, se estanca.

Resultado (2026-10-30): de 1e-4 a 1e-5 la anchura baja ~3,15 veces en todos
los f >= 0,5 pi, también en 0,76-0,77 pi: el testigo óptimo es único y el
cambio de pendiente no es un artefacto del solver.

Uso: (ulimit -v 12000000; python scripts/ladder_unique.py)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import numpy as np

from holonomy import ladder

L = 4
FS = (0.10, 0.50, 0.70, 0.75, 0.76, 0.765, 0.77, 0.78, 0.85, 1.00)
EPS = (1e-4, 1e-5, 1e-6)


def main():
    chk = Checker("ladder_unique")
    D = ladder.LadderDual(L)
    for f in FS:
        ph = ladder.abelian_phases(L, f * np.pi, periodic=True)
        v, y, st = D.v_dual_sdp(ph, **CLARABEL_OPTS)
        widths = []
        for eps in EPS:
            sp = D.witness_spread(ph, v, eps, **CLARABEL_OPTS)
            fails = sum(np.isnan(lo) or np.isnan(hi) for lo, hi in sp)
            if fails:
                chk.info(f"f/pi={f:.3f} eps={eps:.0e}: {fails} de 8 coeficientes sin resolver")
            widths.append(np.nanmax([hi - lo for lo, hi in sp]))
        ratios = [widths[i] / widths[i + 1] for i in range(len(EPS) - 1)]
        chk.info(f"f/pi={f:.3f}: v* = {v:.10f}; anchura máxima de y para eps = "
                 + ", ".join(f"{e:.0e}: {w:.2e}" for e, w in zip(EPS, widths))
                 + f"; cocientes {', '.join(f'{r:.2f}' for r in ratios)} "
                 f"(10 si va como eps, 3,16 si va como sqrt(eps), 1 si hay una cara)")
        if f >= 0.5:
            # Entre 1e-4 y 1e-5 el solver es fiable en todos los puntos; con
            # 1e-6 falla en varios coeficientes (anotado arriba).
            chk.check(ratios[0] > 2.5, f"f/pi={f:.3f}: testigo único (la anchura baja {ratios[0]:.2f} veces "
                      f"de eps = 1e-4 a 1e-5; una cara daría ~1)")
    chk.finish()


if __name__ == "__main__":
    main()

"""Mapa de regímenes del testigo óptimo en la escalera periódica (exploración).

Para la escalera 2xL con flujo uniforme f (gauge simétrico, psi = 0), en una
malla de f se resuelve el dual reducido como SDP (LadderDual.v_dual_sdp) y se
anotan:
  - los bloques (k giros, momento q) cuyo autovalor máximo está a menos de
    TOL de lambda_max del testigo óptimo (el ruido del solver es ~1e-7; una
    separación que crece linealmente desde un borde vale ~1e-3 a un paso de
    malla);
  - el flujo de plaqueta del testigo (holonomy.ladder.witness_fluxes).
Un régimen es un intervalo de f con el mismo conjunto de bloques. Cada borde
se estima extrapolando a cero, linealmente, la separación del bloque que entra
o sale, con los dos puntos de malla del lado en que está fuera; se da también
la pendiente media del flujo del testigo en cada régimen.

Escribe data/ladder_regimes_L<L>.csv (con rango o paso no por defecto,
data/ladder_regimes_L<L>_<f0>_<f1>_<paso>.csv).

Uso: (ulimit -v 12000000; python scripts/ladder_regimes.py L [f0/pi f1/pi paso/pi])
     por defecto L = 4 en [0, 1] con paso 0,0025.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import sys

import numpy as np

from holonomy import ladder
from holonomy.report import write_csv

TOL = 1e-5


def main():
    args = sys.argv[1:]
    L = int(args[0]) if args else 4
    f0, f1, step = (float(a) for a in args[1:4]) if len(args) >= 4 else (0.0, 1.0, 0.0025)
    chk = Checker(f"ladder_regimes L={L}")
    D = ladder.LadderDual(L)
    labels = D.labels
    fs = np.arange(f0, f1 + step / 2, step) * np.pi
    data = []
    for f in fs:
        ph = ladder.abelian_phases(L, f, periodic=True)
        v, y, st = D.v_dual_sdp(ph, **CLARABEL_OPTS)
        gaps = D.block_gaps(y, labels)
        wf, _ = ladder.witness_fluxes(y, L)
        top = frozenset(lab for lab, g in zip(labels, gaps) if g < TOL)
        data.append((f, v, wf, gaps, top, st))
    default = (f0, f1, step) == (0.0, 1.0, 0.0025)
    name = f"ladder_regimes_L{L}.csv" if default else f"ladder_regimes_L{L}_{f0:g}_{f1:g}_{step:g}.csv"
    write_csv(name, ["f", "v_max", "flujo_testigo"] + [f"gap_k{k}_q{q}" for k, q in labels],
              [[f, v, wf] + gaps for f, v, wf, gaps, _, _ in data])
    bad = sum(st not in ("optimal", "optimal_inaccurate") for *_, st in data)
    chk.check(bad == 0, f"{len(data)} puntos resueltos ({bad} con estado de fallo del solver)")

    # Regímenes: intervalos consecutivos con el mismo conjunto de bloques.
    regimes, start = [], 0
    for i in range(1, len(data) + 1):
        if i == len(data) or data[i][4] != data[start][4]:
            regimes.append((start, i - 1))
            start = i
    idx = {lab: m for m, lab in enumerate(labels)}

    def edge(i):
        """Borde entre los puntos i e i+1 por extrapolación lineal."""
        before, after = data[i][4], data[i + 1][4]
        ests = []
        for lab in before ^ after:
            m = idx[lab]
            if lab in before:  # sale: fuera en i+1, i+2
                a, b = i + 1, min(i + 2, len(data) - 1)
            else:  # entra: fuera en i, i-1
                a, b = i, max(i - 1, 0)
            ga, gb = data[a][3][m], data[b][3][m]
            fa, fb = data[a][0], data[b][0]
            if a != b and gb != ga:
                ests.append(fa - ga * (fb - fa) / (gb - ga))
        return np.median(ests) if ests else (data[i][0] + data[i + 1][0]) / 2

    print(f"== Regímenes de la escalera 2x{L}")
    for r, (a, b) in enumerate(regimes):
        lo = data[a][0] if r == 0 else edge(a - 1)
        hi = data[b][0] if r == len(regimes) - 1 else edge(b)
        blocks = sorted(data[a][4])
        slope = ((data[b][2] - data[a][2]) / (data[b][0] - data[a][0])) if b > a else float("nan")
        chk.info(f"f/pi en [{lo / np.pi:.4f}, {hi / np.pi:.4f}] ({b - a + 1} puntos): bloques {blocks}; "
                 f"flujo del testigo/pi {data[a][2] / np.pi:+.4f} -> {data[b][2] / np.pi:+.4f}, "
                 f"pendiente {slope:.3f}")
    j = int(np.argmin([d[1] for d in data]))
    chk.info(f"mínimo de la malla: f/pi = {data[j][0] / np.pi:.4f}, v = {data[j][1]:.9f}, "
             f"bloques {sorted(data[j][4])}")
    chk.finish()


if __name__ == "__main__":
    main()

"""Barrido fino del flujo de plaqueta en la escalera periódica (exploración).

Para cada L y cada flujo uniforme f en [0, pi] (gauge simétrico, psi = 0):
  - v_max por el dual reducido (holonomy.ladder.LadderDual): resuelto como SDP
    (v_dual_sdp, exacto aunque lambda_max sea degenerado) para L <= 5, y por
    Nelder-Mead con continuación en f para L = 6 (el SDP necesita del orden de
    10 GB en L = 6);
  - flujo de plaqueta y holonomía de pata del testigo óptimo
    (holonomy.ladder.witness_fluxes);
  - el mínimo en f, refinado por el método de Brent.
Comprobaciones: el dual-SDP y Nelder-Mead coinciden en L = 4 y 5 (<= 1e-6) y
lambda_max del testigo devuelto por el SDP es su valor (<= 1e-7).

Escribe data/ladder_flux.csv.

Uso: python scripts/ladder_flux.py [L ...]   (por defecto 4 5 6)
Conviene limitar la memoria: (ulimit -v 12000000; python scripts/ladder_flux.py)
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import sys

import numpy as np
from scipy.optimize import minimize_scalar

from holonomy import ladder
from holonomy.report import write_csv

NPTS = 33
SDP_MAX_L = 5


def main():
    Ls = [int(a) for a in sys.argv[1:]] or [4, 5, 6]
    chk = Checker("ladder_flux")
    rows = []
    for L in Ls:
        print(f"== 2x{L} periódica")
        D = ladder.LadderDual(L)
        y = None

        def solve(f):
            nonlocal y
            ph = ladder.abelian_phases(L, f, periodic=True)
            if L <= SDP_MAX_L:
                v, yy, st = D.v_dual_sdp(ph, **CLARABEL_OPTS)
                lam = D.lam_max(yy)
                if abs(lam - v) > 1e-7:
                    chk.check(False, f"2x{L} f={f:.6f}: lambda_max(y) - v = {lam - v:.1e}")
                return v, yy
            v, yy = D.v_dual(ph, y0=y, restarts=0 if y is not None else 3)
            y = yy
            return v, yy

        for f in np.linspace(0, np.pi, NPTS):
            v, yy = solve(f)
            wf, wl = ladder.witness_fluxes(yy, L)
            chk.info(f"2x{L} f/pi={f / np.pi:.4f}: v = {v:.9f}, flujo del testigo/pi = {wf / np.pi:+.4f}, "
                     f"pata del testigo/pi = {wl / np.pi:+.4f}")
            rows.append((L, f, v, wf, wl))
        # Mínimo en f: Brent alrededor del menor punto de la malla.
        mine = [r for r in rows if r[0] == L]
        j = int(np.argmin([r[2] for r in mine]))
        lo, hi = mine[max(j - 1, 0)][1], mine[min(j + 1, len(mine) - 1)][1]
        if hi > lo:
            res = minimize_scalar(lambda f: solve(f)[0], bounds=(lo, hi), method="bounded",
                                  options=dict(xatol=1e-6))
            chk.info(f"2x{L}: mínimo en f/pi = {res.x / np.pi:.5f}, v = {res.fun:.9f}")
        if L <= SDP_MAX_L:
            for f in (np.pi / 2, np.pi):
                ph = ladder.abelian_phases(L, f, periodic=True)
                vs = D.v_dual_sdp(ph, **CLARABEL_OPTS)[0]
                vn = D.v_dual(ph)[0]
                chk.check(abs(vs - vn) < 1e-6, f"2x{L} f/pi={f / np.pi:.2f}: dual-SDP = {vs:.9f}, "
                          f"Nelder-Mead = {vn:.9f}, |dif| = {abs(vs - vn):.1e}")
    write_csv("ladder_flux.csv", ["L", "f", "v_max", "flujo_testigo", "pata_testigo"], rows)
    chk.finish()


if __name__ == "__main__":
    main()

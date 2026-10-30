"""Holonomía de las patas en la escalera periódica (exploración).

La escalera periódica 2xL tiene L+1 ciclos independientes: las L plaquetas y
una pata. Con flujo uniforme f por plaqueta, en el gauge simétrico
(ladder.abelian_phases con psi), las holonomías de las patas son
psi_b = L f/2 + psi (inferior) y psi_t = -L f/2 + psi (superior), con
psi_b - psi_t = L f. Simetrías: (f, psi) ~ (-f, psi) (intercambio de patas)
~ (f, -psi) (intercambio compuesto con R_x(pi)); basta f, psi en [0, pi].

El dual reducido (ladder.LadderDual) solo usa U(1), la traslación y la
reflexión izquierda-derecha compuesta con R_x(pi), que siguen siendo
simetrías con psi != 0. Comprobaciones:
  1. dual-SDP = primal U(1) sin reducir por traslación en dos puntos con
     psi != 0 (solo L = 4; <= 1e-6);
  2. v(f, psi) = v(f, -psi) en varios puntos (<= 1e-6);
  3. en cada punto, v <= cota de estrella 5/9, cota de plaqueta
     2/(3 + sin(f/2)) y cota de cada pata como anillo de L sitios,
     v_anillo(L, psi_b) y v_anillo(L, psi_t) (dual del anillo, sección 5).
En cada punto se guardan v, los bloques (k, q) en lambda_max del testigo
óptimo, su flujo de plaqueta y sus holonomías de pata.

Escribe data/ladder_psi_L<L>.csv.

Uso: (ulimit -v 12000000; python scripts/ladder_psi.py L N)   malla N x N
     por defecto L = 4, N = 21.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import sys

import numpy as np

from holonomy import dual, ladder, xxz
from holonomy.report import write_csv
from holonomy.sdp import primal_u1_graph

TOL = 1e-6
STAR = 5 / 9
GAP_TOL = 1e-5


def wrap(a):
    """Ángulo reducido a [0, pi] (el ángulo de una rotación R_z)."""
    return abs(float(np.angle(np.exp(1j * a))))


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    chk = Checker(f"ladder_psi L={L}")
    D = ladder.LadderDual(L)
    ring = {}

    def v_ring(theta):
        key = round(wrap(theta), 12)
        if key not in ring:
            ring[key] = dual.v_free(xxz.eps, L, key)[0]
        return ring[key]

    def solve(f, psi):
        ph = ladder.abelian_phases(L, f, periodic=True, psi=psi)
        v, y, st = D.v_dual_sdp(ph, **CLARABEL_OPTS)
        return v, y, st, ph

    print("== Contrastes")
    if L == 4:
        for f, psi in ((0.5 * np.pi, 0.3 * np.pi), (0.8 * np.pi, 0.7 * np.pi)):
            v, _, _, ph = solve(f, psi)
            vp, _, st = primal_u1_graph(2 * L, D.edges, ph, return_blocks=True, **CLARABEL_OPTS)
            chk.check(abs(v - vp) < TOL, f"f/pi={f / np.pi:.2f} psi/pi={psi / np.pi:.2f}: dual = {v:.9f}, "
                      f"primal U(1) = {vp:.9f}, |dif| = {abs(v - vp):.1e}{chk.status_note(st)}")
    for f, psi in ((0.3 * np.pi, 0.4 * np.pi), (0.7 * np.pi, 0.9 * np.pi), (1.0 * np.pi, 0.5 * np.pi)):
        v1, v2 = solve(f, psi)[0], solve(f, -psi)[0]
        chk.check(abs(v1 - v2) < TOL, f"simetría psi -> -psi en f/pi={f / np.pi:.2f}, psi/pi={psi / np.pi:.2f}: "
                  f"|dif| = {abs(v1 - v2):.1e}")

    print("== Malla")
    rows, viol = [], 0
    grid = np.linspace(0, np.pi, N)
    for f in grid:
        for psi in grid:
            v, y, st, ph = solve(f, psi)
            psib, psit = L * f / 2 + psi, -L * f / 2 + psi
            bound = min(STAR, 2 / (3 + np.sin(f / 2)), v_ring(psib), v_ring(psit))
            if v > bound + TOL:
                viol += 1
                chk.check(False, f"f/pi={f / np.pi:.3f} psi/pi={psi / np.pi:.3f}: v = {v:.9f} > cota {bound:.9f}")
            gaps = D.block_gaps(y, D.labels)
            top = sorted(lab for lab, g in zip(D.labels, gaps) if g < GAP_TOL)
            wf = np.arctan2(y[1], y[0]) - np.arctan2(y[4], y[3])
            wb, wt = L * np.arctan2(y[1], y[0]), L * np.arctan2(y[4], y[3])
            rows.append((f, psi, v, bound, wrap(wf), wrap(wb), wrap(wt),
                         " ".join(f"{k}:{q}" for k, q in top)))
        chk.info(f"f/pi={f / np.pi:.3f}: v(psi) = " + " ".join(f"{r[2]:.4f}" for r in rows[-N:]))
    chk.check(viol == 0, f"{len(rows)} puntos: v <= min(estrella, plaqueta, patas como anillos) en todos")
    write_csv(f"ladder_psi_L{L}.csv", ["f", "psi", "v_max", "cota", "flujo_testigo", "pata_b_testigo",
                                        "pata_t_testigo", "bloques"], rows)
    j = int(np.argmin([r[2] for r in rows]))
    k = int(np.argmax([r[2] for r in rows]))
    chk.info(f"mínimo: v = {rows[j][2]:.9f} en f/pi = {rows[j][0] / np.pi:.3f}, psi/pi = {rows[j][1] / np.pi:.3f}")
    chk.info(f"máximo: v = {rows[k][2]:.9f} en f/pi = {rows[k][0] / np.pi:.3f}, psi/pi = {rows[k][1] / np.pi:.3f}")
    chk.finish()


if __name__ == "__main__":
    main()

"""¿Es la escalera abierta 2x3 con holonomías R_z(pi), R_x(pi) un mínimo local estricto? (exploración)

Los invariantes gauge de un par de holonomías de plaqueta son sus ángulos
theta_A, theta_B y el ángulo gamma entre sus ejes. Se perturba el punto
(theta_A, theta_B, gamma) = (pi, pi, pi/2) con theta_A = pi - a,
theta_B = pi - b, gamma = pi/2 + c (R_B = R_y(gamma) R_z(theta_B) R_y(gamma)^T)
y se resuelve el primal completo (holonomy.sdp.primal). Como R_n(pi) =
R_{-n}(pi), los cambios de signo de (a, b, c) dan datos equivalentes: si v es
suave, el punto es crítico. Comprobaciones:
  1. el punto reproduce v de R_z(pi), R_x(pi) (<= 1e-7);
  2. v(p + d) > v(p) en los tres ejes (eps = 0,1; 0,05; 0,025; ambos signos)
     y en 12 direcciones al azar (eps = 0,1; 0,05);
  3. ajuste de dv a una forma cuadrática más términos lineales y cúbicos:
     términos lineales < 1e-6 y forma cuadrática definida positiva.

Escribe data/ladder_perp_min.csv (a, b, c, dv).

Uso: (ulimit -v 8000000; python scripts/ladder_perp_min.py)   ~19 min
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import numpy as np

from holonomy import ladder
from holonomy.paulis import Rx, Rz
from holonomy.report import write_csv
from holonomy.sdp import primal
from ladder import tree_gauge_open

L, N = 3, 6
EDGES = ladder.ladder_edges(L)


def Ry(g):
    c, s = np.cos(g), np.sin(g)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def v_at(tA, tB, g):
    RB = Ry(g) @ Rz(tB) @ Ry(g).T
    return primal(N, tree_gauge_open(L, (Rz(tA), RB)), edges=EDGES, **CLARABEL_OPTS)


def feats(d):
    a, b, c = d
    return [a * a, b * b, c * c, 2 * a * b, 2 * a * c, 2 * b * c, a, b, c, a * b * c, a ** 3, b ** 3, c ** 3]


def main():
    chk = Checker("ladder_perp_min")
    pi = np.pi
    P = np.array([pi, pi, pi / 2])
    v0 = v_at(*P)
    vx = primal(N, tree_gauge_open(L, (Rz(pi), Rx(pi))), edges=EDGES, **CLARABEL_OPTS)
    chk.check(abs(v0 - vx) < 1e-7, f"punto: v = {v0:.10f}, con R_x(pi) directo {vx:.10f}, |dif| = {abs(v0 - vx):.1e}")

    dirs = [s * eps * e for eps in (0.1, 0.05, 0.025) for e in np.eye(3) for s in (1, -1)]
    rng = np.random.default_rng(0)
    for _ in range(12):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        dirs += [0.1 * u, 0.05 * u]
    rows = []
    for d in dirs:
        dv = v_at(*(P + d)) - v0
        rows.append((*d, dv))
        eps = np.linalg.norm(d)
        chk.info(f"(a, b, c) = ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}): dv = {dv:+.3e}, dv/eps^2 = {dv / eps ** 2:+.5f}")
    dvs = np.array([r[3] for r in rows])
    chk.check(dvs.min() > 0, f"{len(rows)} perturbaciones: dv > 0 en todas (mínimo {dvs.min():.2e})")

    D = np.array([r[:3] for r in rows])
    F = np.array([feats(d) for d in D])
    c, *_ = np.linalg.lstsq(F, dvs, rcond=None)
    H = np.array([[c[0], c[3], c[4]], [c[3], c[1], c[5]], [c[4], c[5], c[2]]])
    ev = np.linalg.eigvalsh(H)
    chk.check(np.abs(c[6:9]).max() < 1e-6, f"términos lineales: {np.array2string(c[6:9], precision=1)}")
    chk.check(ev.min() > 0, f"forma cuadrática: autovalores {np.array2string(ev, precision=5)}, "
              f"diagonal (a, b, c) = ({H[0, 0]:.5f}, {H[1, 1]:.5f}, {H[2, 2]:.5f}); "
              f"residuo del ajuste {np.abs(F @ c - dvs).max():.1e}")
    write_csv("ladder_perp_min.csv", ["a", "b", "c", "dv"], rows)
    chk.finish()


if __name__ == "__main__":
    main()

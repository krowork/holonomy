"""Mínimo global de v en la escalera abierta 2x3 sobre todas las holonomías (exploración).

Los invariantes gauge de un par de holonomías de plaqueta son sus ángulos
theta_A, theta_B en [0, pi] y el ángulo gamma en [0, pi] entre sus ejes
(R_A = R_z(theta_A), R_B = R_y(gamma) R_z(theta_B) R_y(gamma)^T, como en
scripts/ladder_perp_min.py), con el primal completo (holonomy.sdp.primal).
La reflexión de la escalera invierte las dos plaquetas, así que
v(theta_A, theta_B, gamma) = v(theta_B, theta_A, gamma): basta
theta_A <= theta_B. Pasos:
  1. malla theta en {0, pi/4, pi/2, 3pi/4, pi}, gamma en {0, pi/6, ..., pi}
     (con theta = 0 el eje no cuenta: un solo gamma);
  2. Nelder-Mead desde los tres menores puntos de la malla, con
     (theta_A, theta_B, gamma) reflejados a [0, pi].
Comprobaciones:
  1. simetría theta_A <-> theta_B en un punto genérico (<= 1e-7);
  2. la malla reproduce los casos de data/ladder.csv (abelianos y R_z, R_x);
Se da el menor de todos los puntos evaluados y se compara con el punto de
rotaciones perpendiculares (pi, pi, pi/2) de scripts/ladder_perp_min.py.

Cada evaluación se guarda en data/ladder_open3_evals.csv al terminarla y no
se repite al relanzar (reanudable). Escribe data/ladder_open3_min.csv.

Uso: (ulimit -v 8000000; python scripts/ladder_open3_min.py)   varias horas
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker

import csv

import numpy as np
from scipy.optimize import minimize

from holonomy.report import DATA, write_csv
from ladder_perp_min import v_at

EVALS = DATA / "ladder_open3_evals.csv"
pi = np.pi
_cache = {}


def _key(p):
    return tuple(round(float(x), 10) for x in p)


def load():
    if EVALS.exists():
        for r in csv.DictReader(open(EVALS)):
            _cache[_key((r["theta_A"], r["theta_B"], r["gamma"]))] = float(r["v_max"])


def v(p):
    """v en (theta_A, theta_B, gamma), con caché en disco."""
    k = _key(p)
    if k not in _cache:
        _cache[k] = v_at(*k)
        new = not EVALS.exists()
        with open(EVALS, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["theta_A", "theta_B", "gamma", "v_max"])
            w.writerow([*k, _cache[k]])
    return _cache[k]


def fold(p):
    """Refleja cada coordenada a [0, pi]."""
    return np.abs(np.mod(np.asarray(p) + np.pi, 2 * np.pi) - np.pi)


def main():
    chk = Checker("ladder_open3_min")
    load()
    p = (0.6 * pi, 0.85 * pi, 0.3 * pi)
    a, b = v(p), v((p[1], p[0], p[2]))
    chk.check(abs(a - b) < 1e-7, f"simetría theta_A <-> theta_B: |dif| = {abs(a - b):.1e}")

    thetas = np.array([0, 0.25, 0.5, 0.75, 1.0]) * pi
    gammas = np.arange(7) / 6 * pi
    grid = []
    for i, ta in enumerate(thetas):
        for tb in thetas[i:]:
            for g in (gammas if ta > 0 and tb > 0 else gammas[:1]):
                grid.append(((ta, tb, g), v((ta, tb, g))))
    ref = {"(0.50pi, 0.50pi)": (pi / 2, pi / 2, 0), "(1.00pi, 1.00pi)": (pi, pi, 0),
           "(0.50pi, 0)": (0, pi / 2, 0), "(1.00pi, 0)": (0, pi, 0),
           "(0.50pi, -0.50pi)": (pi / 2, pi / 2, pi),
           "Rz(1.00pi), Rx(1.00pi)": (pi, pi, pi / 2), "Rz(0.50pi), Rx(0.50pi)": (pi / 2, pi / 2, pi / 2)}
    old = {r["holonomia"]: float(r["v_max"]) for r in csv.DictReader(open(DATA / "ladder.csv"))
           if r["contorno"] == "abierta" and r["L"] == "3"}
    worst = max(abs(v(q) - old[name]) for name, q in ref.items())
    chk.check(worst < 1e-6, f"malla = casos de data/ladder.csv: máx |dif| = {worst:.1e}")
    grid.sort(key=lambda t: t[1])
    for q, val in grid[:5]:
        chk.info(f"malla: (theta_A, theta_B, gamma)/pi = ({q[0] / pi:.3f}, {q[1] / pi:.3f}, {q[2] / pi:.3f}), "
                 f"v = {val:.10f}")

    for q, _ in grid[:3]:
        res = minimize(lambda x: v(fold(x)), np.array(q), method="Nelder-Mead",
                       options=dict(xatol=1e-4, fatol=1e-10, maxfev=250, initial_simplex=np.array(q) + 0.1 * pi *
                                    np.vstack([np.zeros(3), -np.eye(3)])))
        x = fold(res.x)
        chk.info(f"Nelder-Mead desde ({q[0] / pi:.3f}, {q[1] / pi:.3f}, {q[2] / pi:.3f}) pi: "
                 f"({x[0] / pi:.5f}, {x[1] / pi:.5f}, {x[2] / pi:.5f}) pi, v = {res.fun:.10f}, {res.nfev} evaluaciones")
    k, vmin = min(_cache.items(), key=lambda t: t[1])
    vp = v((pi, pi, pi / 2))
    chk.info(f"{len(_cache)} puntos evaluados; el menor: ({k[0] / pi:.5f}, {k[1] / pi:.5f}, {k[2] / pi:.5f}) pi, "
             f"v = {vmin:.10f}; rotaciones perpendiculares (pi, pi, pi/2): {vp:.10f}, dif = {vmin - vp:+.1e}")
    write_csv("ladder_open3_min.csv", ["theta_A", "theta_B", "gamma", "v_min"], [(*k, vmin)])
    chk.finish()


if __name__ == "__main__":
    main()

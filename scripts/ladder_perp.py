"""Testigo óptimo de la escalera abierta 2x3 con holonomías R_z(pi), R_x(pi) (exploración).

En el gauge de árbol de scripts/ladder.py los datos de arista son diagonales:
la identidad en el árbol, diag(-1, -1, 1) y diag(1, -1, -1) en las aristas
superiores. Las tres holonomías de ciclo (dos plaquetas y el ciclo exterior,
R_y(pi)) son rotaciones de pi.

Dual del primal completo (holonomy.sdp.primal): v = min lambda_max(W) con
W = sum_e sum_ab Y_e[a,b] s_a^i s_b^j + sum_i h_i . s_i y
N(Y) = -sum_e tr(Y_e^T R_e) = 1 (81 coeficientes). Las rotaciones globales de
pi alrededor de x, y, z conservan los datos; promediar sobre ellas anula los
términos no diagonales y los campos, así que hay un testigo óptimo XYZ con
pesos por enlace. Comprobaciones:
  1. dual = primal (<= 1e-6);
  2. el testigo que da el solver es XYZ sin campos (<= 1e-6);
  3. unicidad: rango de cada coeficiente en {N = 1, lambda_max <= v + eps}
     para eps = 1e-4, 1e-5, 1e-6; si el testigo es único, tiende a 0.

Escribe data/ladder_perp_witness.csv (pesos XYZ por arista) y
data/ladder_perp_spread.csv (rango de cada coeficiente por eps).

Uso: (ulimit -v 8000000; python scripts/ladder_perp.py)
Reanudable: los eps ya guardados en data/ladder_perp_spread.csv no se recalculan.
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker

import csv
import time

import cvxpy as cp
import numpy as np

from holonomy import ladder
from holonomy.paulis import Rx, Rz, pauli_string
from holonomy.report import DATA, write_csv
from holonomy.sdp import primal
from ladder import tree_gauge_open

TOL = 1e-6
EPSS = (1e-4, 1e-5, 1e-6)


def main():
    chk = Checker("ladder_perp")
    L, n = 3, 6
    edges = ladder.ladder_edges(L)
    Rs = tree_gauge_open(L, (Rz(np.pi), Rx(np.pi)))
    vp = primal(n, Rs, edges=edges, **CLARABEL_OPTS)

    keys2 = [(e, a, b) for e in range(len(edges)) for a in range(3) for b in range(3)]
    keys1 = [(i, a) for i in range(n) for a in range(3)]
    ops = [pauli_string(n, ((edges[e][0], a), (edges[e][1], b))) for e, a, b in keys2]
    ops += [pauli_string(n, ((i, a),)) for i, a in keys1]
    m = len(ops)
    Y = cp.Variable(m)
    t = cp.Variable()
    W = sum(Y[k] * P for k, P in enumerate(ops))
    g = np.array([-np.asarray(Rs[e])[a, b] for e, a, b in keys2] + [0.0] * len(keys1))
    eye = np.eye(2 ** n)
    prob = cp.Problem(cp.Minimize(t), [g @ Y == 1, t * eye - W >> 0])
    prob.solve(solver="CLARABEL", **CLARABEL_OPTS)
    v, y = float(t.value), np.asarray(Y.value)
    chk.check(abs(v - vp) < TOL, f"dual = {v:.10f}, primal = {vp:.10f}, |dif| = {abs(v - vp):.1e}"
              f"{chk.status_note(prob.status)}")

    diag = [k for k, (e, a, b) in enumerate(keys2) if a == b]
    rest = [k for k in range(m) if k not in diag]
    chk.check(np.abs(y[rest]).max() < TOL, f"testigo del solver XYZ sin campos: máx |no diagonal, campo| = "
              f"{np.abs(y[rest]).max():.1e}")
    rows = []
    for e, (i, j) in enumerate(edges):
        J = [y[keys2.index((e, a, a))] for a in range(3)]
        chk.info(f"arista {i}-{j}: (Jx, Jy, Jz) = ({J[0]:+.6f}, {J[1]:+.6f}, {J[2]:+.6f}), "
                 f"dato diag(R) = {np.diag(np.asarray(Rs[e])).astype(int)}")
        rows.append((i, j, *J))
    write_csv("ladder_perp_witness.csv", ["i", "j", "Jx", "Jy", "Jz"], rows)

    # Reanudable: cada eps se guarda al terminar y no se recalcula.
    path = DATA / "ladder_perp_spread.csv"
    spread_rows = []
    if path.exists():
        spread_rows = [(float(r["eps"]), float(r["rango_xyz"]), float(r["rango_resto"]))
                       for r in csv.DictReader(open(path))]
    done = {r[0] for r in spread_rows}
    for r in spread_rows:
        chk.info(f"eps = {r[0]:.0e}: rango máx de los pesos XYZ = {r[1]:.2e}, "
                 f"de los no diagonales y campos = {r[2]:.2e} (guardado)")
    for eps in EPSS:
        if eps in done:
            continue
        t0 = time.time()
        cons = [g @ Y == 1, (v + eps) * eye - W >> 0]
        spread = np.zeros(m)
        for k in range(m):
            lo = cp.Problem(cp.Minimize(Y[k]), cons)
            lo.solve(solver="CLARABEL", **CLARABEL_OPTS)
            hi = cp.Problem(cp.Maximize(Y[k]), cons)
            hi.solve(solver="CLARABEL", **CLARABEL_OPTS)
            spread[k] = hi.value - lo.value
        chk.info(f"eps = {eps:.0e}: rango máx de los pesos XYZ = {spread[diag].max():.2e}, "
                 f"de los no diagonales y campos = {spread[rest].max():.2e} ({time.time() - t0:.0f} s)")
        spread_rows.append((eps, spread[diag].max(), spread[rest].max()))
        write_csv("ladder_perp_spread.csv", ["eps", "rango_xyz", "rango_resto"], spread_rows)
    chk.finish()


if __name__ == "__main__":
    main()

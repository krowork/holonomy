"""El diamante simple K4-e tiene v_max = 1/3 para toda holonomía.

Sitios j = 0, u = 1, w = 2, j' = 3; aristas orientadas (0,1), (0,2), (1,2),
(1,3), (2,3) (holonomy.diamond.chain_edges(1)). Triángulos A = (0,1,2) y
B = (1,3,2). Cota superior: cada triángulo es un anillo de 3, con v_max = 1/3
para toda holonomía, y la cota de un subgrafo vale para el grafo entero.

Puntos: (theta_A, theta_B, gamma) = (0,0,0), (pi,pi,pi/2) y 6 al azar en
[0,pi]^3, con H_A = R_z(theta_A), H_B = R_y(gamma) R_z(theta_B) R_y(gamma)^T en
el gauge de árbol (holonomy.diamond.chain_rotations); y 6 juegos de rotaciones
aleatorias (uniformes en SO(3)) en las 5 aristas. Semilla fija.

Estado explícito: las aristas se reparten en tres emparejamientos,
M1 = {(0,1),(2,3)}, M2 = {(0,2),(1,3)}, M3 = {(1,2)}. Para la arista e,
s_e = rho_e(1) = (I - sum R_ab s_a s_b)/4 es un estado puro (singlete rotado,
T = -R_e). Entonces
    rho = (1/3) [ s_01 (x) s_23 + s_02 (x) s_13 + s_12 (x) I_03/4 ]
tiene en cada arista T = -R_e/3 (la arista aparece en un solo emparejamiento;
en los otros dos sus qubits están en factores distintos con marginales
I/2) y marginales nulas, sea cual sea la holonomía.

Comprueba en cada punto:
  1. primal completo (holonomy.sdp.primal) = 1/3 (<= 1e-6);
  2. el estado óptimo tiene marginales nulas y las 15 correlaciones de cada
     arista de rho_e(v) a 1e-6;
  3. el estado explícito es un estado (lambda_min >= -1e-12, traza 1) y tiene
     las correlaciones de cada arista de rho_e(1/3) a 1e-12.

Escribe data/diamond_single.csv (punto, v, estado del solver, error del
óptimo, error del explícito).

Uso: python scripts/diamond_single.py   (2,4 s y 150 MB de pico, medidos con
/usr/bin/time -f "%M KB")
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker, edge_mismatch

import numpy as np

from holonomy.diamond import chain_edges, chain_rotations
from holonomy.paulis import Ry, Rz, werner_state
from holonomy.report import write_csv
from holonomy.sdp import primal

N = 4
EDGES = chain_edges(1)
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((1, 2),))
SEED = 20261103
# Clarabel acaba en optimal_inaccurate (cara óptima degenerada); la exactitud la da la comprobación 3.
V_TOL = 1e-6
OPT_TOL = 1e-6
EXACT_TOL = 1e-12


def random_rotation(rng):
    """Rotación uniforme en SO(3) (QR de una gaussiana con signos fijados)."""
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(Rr))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def place(pairs_states, n):
    """Producto tensorial de estados de 2 qubits en los pares dados (el resto
    de qubits en I/2), reordenado a qubits 0..n-1 (qubit 0 el más significativo)."""
    order = [q for pair, _ in pairs_states for q in pair]
    rest = [q for q in range(n) if q not in order]
    rho = np.ones((1, 1), dtype=complex)
    for _, s in pairs_states:
        rho = np.kron(rho, s)
    for _ in rest:
        rho = np.kron(rho, np.eye(2) / 2)
    order += rest
    perm = [order.index(q) for q in range(n)]
    t = rho.reshape([2] * (2 * n)).transpose(perm + [n + p for p in perm])
    return t.reshape(2 ** n, 2 ** n)


def explicit_state(Rs):
    """rho = (1/3) sum_M prod_{e en M} s_e, con s_e = rho_e(1) puro."""
    R = dict(zip(EDGES, Rs, strict=True))
    return sum(place([(e, werner_state(1.0, R[e])) for e in M], N) for M in MATCHINGS) / len(MATCHINGS)


def points():
    rng = np.random.default_rng(SEED)
    pts = [("(0,0,0)", chain_rotations(1, [(Rz(0.0), Rz(0.0))])),
           ("(pi,pi,pi/2)", chain_rotations(1, [(Rz(np.pi), Ry(np.pi / 2) @ Rz(np.pi) @ Ry(np.pi / 2).T)]))]
    for _ in range(6):
        tA, tB, g = rng.uniform(0, np.pi, 3)
        pts.append((f"({tA:.4f},{tB:.4f},{g:.4f})", chain_rotations(1, [(Rz(tA), Ry(g) @ Rz(tB) @ Ry(g).T)])))
    for k in range(6):
        pts.append((f"aleatorias #{k}", [random_rotation(rng) for _ in EDGES]))
    return pts


def main():
    chk = Checker("diamond_single")
    rows = []
    for lab, Rs in points():
        v, rho, status = primal(N, Rs, edges=EDGES, return_state=True, **CLARABEL_OPTS)
        note = chk.status_note(status)
        chk.check(abs(v - 1 / 3) < V_TOL, f"{lab:28s} primal v = {v:.12f}, |v-1/3| = {abs(v - 1 / 3):.1e}{note}")
        e_opt = edge_mismatch(rho, N, v, Rs, edges=EDGES)
        chk.check(e_opt < OPT_TOL, f"{lab:28s} óptimo: marginales y 15 correlaciones x 5 aristas, "
                  f"err = {e_opt:.1e}{note}")
        sigma = explicit_state(Rs)
        w = np.linalg.eigvalsh(sigma)
        e_exp = edge_mismatch(sigma, N, 1 / 3, Rs, edges=EDGES)
        tr_err = abs(np.trace(sigma) - 1)
        chk.check(w[0] > -EXACT_TOL and tr_err < EXACT_TOL and e_exp < EXACT_TOL,
                  f"{lab:28s} explícito: lambda_min = {w[0]:.1e}, |tr-1| = {tr_err:.1e}, "
                  f"correlaciones vs rho_e(1/3) err = {e_exp:.1e}")
        rows.append([lab, f"{v:.12f}", status, f"{e_opt:.3e}", f"{e_exp:.3e}"])
    write_csv("diamond_single.csv", ["punto", "v", "estado", "err_optimo", "err_explicito"], rows)
    chk.finish()


if __name__ == "__main__":
    main()

"""SDP primal independiente para n = 3..6 (secciones sec:anillo4 y sec:nsitios).

Implementación propia del primal, independiente de holonomy.sdp: la marginal
de dos qubits de cada arista se impone igual a rho_e(v) como funcional lineal
tr(P rho) = sum_kl P_lk rho_kl (sin productos matriciales, para que el caso
n = 6, rho de 64x64, quepa en memoria y tiempo). Solo se reutilizan los
operadores de Pauli de holonomy.paulis.

Comprueba:
  1. v_max(n, Theta) en el gauge uniforme, n = 3..6, Theta = 0, pi/4, pi/2,
     3pi/4, pi, frente a la tabla tab:valores del artículo (4 decimales);
  2. el caso fuera de dominio: n = 4 con ángulo por arista phi = 3pi/4
     (holonomía 3pi, equivalente a pi) da 1/2; la fórmula ingenua en phi,
     1/(1 + cos^2(pi/4 - phi)), daría 1.

Uso: python scripts/chk.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import ANGLE_LABELS, ANGLES, CLARABEL_OPTS, TABLE, TABLE_TOL, Checker

import itertools

import cvxpy as cp
import numpy as np

from holonomy.paulis import Rz, holonomy, pauli_string, rotation_angle


def trace_functional(P, rho):
    """Re tr(P rho) como funcional lineal elemento a elemento."""
    return cp.real(cp.sum(cp.multiply(P.T, rho)))


def independent_primal(n, Rs):
    """max v sujeto a rho >= 0, tr rho = 1, <s_a^i> = 0, <s_a^i s_b^j> = -v R_e[a, b]."""
    d = 2 ** n
    rho = cp.Variable((d, d), hermitian=True)
    v = cp.Variable()
    cons = [rho >> 0, cp.real(cp.trace(rho)) == 1]
    for i in range(n):
        for a in range(3):
            cons.append(trace_functional(pauli_string(n, ((i, a),)), rho) == 0)
    for i in range(n):
        j = (i + 1) % n
        T = -np.asarray(Rs[i])
        for a, b in itertools.product(range(3), repeat=2):
            P = pauli_string(n, ((i, a), (j, b)))
            cons.append(trace_functional(P, rho) == v * T[a, b])
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver="CLARABEL", **CLARABEL_OPTS)
    return float(v.value), prob.status


def main():
    chk = Checker("chk")
    print("== Primal independiente vs tabla tab:valores")
    for n in range(3, 7):
        for Theta, lab, ref in zip(ANGLES, ANGLE_LABELS, TABLE[n]):
            v, status = independent_primal(n, [Rz(Theta / n)] * n)
            err = abs(v - ref)
            chk.check(err < TABLE_TOL, f"n={n} Theta={lab:5s}: v={v:.10f} tabla={ref:.4f} "
                      f"|dif|={err:.1e}{chk.status_note(status)}")

    print("== Fuera de dominio: n=4, phi=3pi/4 por arista")
    phi = 3 * np.pi / 4
    Rs = [Rz(phi)] * 4
    theta = rotation_angle(holonomy(Rs))
    naive = 1 / (1 + np.cos(np.pi / 4 - phi) ** 2)
    v, status = independent_primal(4, Rs)
    chk.check(abs(theta - np.pi) < 1e-12,
              f"ángulo de la holonomía: theta/pi={theta / np.pi:.15f} (esperado 1)")
    chk.check(abs(v - 0.5) < 1e-7 and abs(naive - 1) < 1e-15,
              f"v={v:.10f} (esperado 1/2, err={abs(v - 0.5):.1e}); fórmula ingenua en phi "
              f"daría {naive:.6f}{chk.status_note(status)}")
    chk.finish()


if __name__ == "__main__":
    main()

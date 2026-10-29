"""Estado explícito del triángulo (Proposición prop:triangulo, sección sec:partida).

Con |j> = qubit j en |1> y los demás en |0> (qubit 0 el más significativo),
S|j> = |j+1 mod 3>, y en ese sector de dimensión 3
    rho_+ = (1/3) [I + (1/2)(e^{i alpha} S + e^{-i alpha} S^dag)],  alpha = pi - Theta/3,
    rho   = (1/2) (rho_+ + X^{x3} conj(rho_+) X^{x3}).
Comprueba, para Theta = 0, pi/4, pi/2, 3pi/4, pi:
  1. autovalores de rho_+ en el sector: (1 + cos(alpha + 2 pi k/3))/3;
  2. rho >= 0 y tr rho = 1;
  3. las 15 correlaciones de cada arista (i -> i+1, incluida 2 -> 0) coinciden
     con las de rho_e(1/3), T^e = -R_z(Theta/3) (artículo: <= 1e-16);
  4. el SDP primal completo (holonomy.sdp.primal) da v_max = 1/3.

Uso: python scripts/tri.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import ANGLE_LABELS, ANGLES, CLARABEL_OPTS, Checker, edge_mismatch, uniform_gauge

import numpy as np

from holonomy.paulis import X, op
from holonomy.sdp import primal

N = 3
EXACT_TOL = 1e-15


def triangle_state(Theta):
    """Devuelve (rho de 8x8, rho_+ en el sector de 1 giro de 3x3)."""
    alpha = np.pi - Theta / 3
    S = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        S[(j + 1) % 3, j] = 1.0          # S|j> = |j+1>
    rp = (np.eye(3) + 0.5 * (np.exp(1j * alpha) * S + np.exp(-1j * alpha) * S.conj().T)) / 3
    idx = [1 << (N - 1 - j) for j in range(3)]  # |j>: qubit j en |1>
    rho_plus = np.zeros((8, 8), dtype=complex)
    for a in range(3):
        for b in range(3):
            rho_plus[idx[a], idx[b]] = rp[a, b]
    XXX = op(N, {k: X for k in range(N)})
    rho = 0.5 * (rho_plus + XXX @ rho_plus.conj() @ XXX)
    return rho, rp


def main():
    chk = Checker("tri")
    for Theta, lab in zip(ANGLES, ANGLE_LABELS):
        alpha = np.pi - Theta / 3
        rho, rp = triangle_state(Theta)
        w_plus = np.sort(np.linalg.eigvalsh(rp))
        w_pred = np.sort([(1 + np.cos(alpha + 2 * np.pi * k / 3)) / 3 for k in range(3)])
        e1 = np.max(np.abs(w_plus - w_pred))
        chk.check(e1 < EXACT_TOL, f"Theta={lab:5s}: autovalores de rho_+ "
                  + " ".join(f"{x:.6f}" for x in w_plus) + f" vs (1+cos(alpha+2pi k/3))/3 err={e1:.1e}")
        w = np.linalg.eigvalsh(rho)
        chk.check(w[0] > -EXACT_TOL and abs(np.trace(rho) - 1) < EXACT_TOL
                  and np.max(np.abs(rho - rho.conj().T)) < EXACT_TOL,
                  f"Theta={lab:5s}: rho >= 0 (lambda_min={w[0]:.1e}), tr rho - 1={abs(np.trace(rho) - 1):.1e}; "
                  f"autovalores de rho " + " ".join(f"{x:.4f}" for x in w[::-1] if x > 1e-14))
        err = edge_mismatch(rho, N, 1 / 3, uniform_gauge(N, Theta))
        chk.check(err < 1e-15, f"Theta={lab:5s}: 15 correlaciones x 3 aristas vs rho_e(1/3) err={err:.1e}")
        v, _, status = primal(N, uniform_gauge(N, Theta), return_state=True, **CLARABEL_OPTS)
        chk.check(abs(v - 1 / 3) < 1e-8, f"Theta={lab:5s}: SDP primal v={v:.12f} |v-1/3|={abs(v - 1 / 3):.1e}"
                  f"{chk.status_note(status)}")
    chk.finish()


if __name__ == "__main__":
    main()

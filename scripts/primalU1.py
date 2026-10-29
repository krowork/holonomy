"""SDP primal reducido por U(1) hasta n = 7 (secciones sec:reduccion y sec:optimalidad).

Compara holonomy.sdp.primal_u1 (rho en bloques de número de giros fijo, gauge
uniforme) con la fórmula dual de la Proposición prop:xxz, el mínimo sobre
(Delta, chi) de 4 eps_n(Delta, n chi)/(Delta + 2 cos(chi - Theta/n)),
calculado con holonomy.dual.v_free y la diagonalización exacta holonomy.xxz.eps:
  1. n = 3..7 en Theta = 0, pi/4, pi/2, 3pi/4, pi;
  2. n = 7 dentro de la ventana de desanclaje, Theta/pi = 0.985, 0.99, 0.995
     (artículo: coincidencia ~1e-8).
Para cada óptimo se imprime el rango numérico y el peso de cada sector de
número de giros k (tr del bloque k).

Uso: python scripts/primalU1.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import ANGLE_LABELS, ANGLES, CLARABEL_OPTS, Checker, u1_eigs

import numpy as np

from holonomy import dual, xxz
from holonomy.sdp import primal_u1

TOL = 1e-6          # artículo: <= 1e-6 en general
WINDOW_TOL = 1e-7   # artículo: ~1e-8 dentro de la ventana
RANK_TOL = 1e-5


def run_case(chk, n, Theta, label, tol):
    v, blocks, status = primal_u1(n, Theta, return_blocks=True, **CLARABEL_OPTS)
    vd, Phi, Delta = dual.v_free(xxz.eps, n, Theta)
    err = abs(v - vd)
    chk.check(err < tol, f"n={n} Theta={label:9s}: primal_u1={v:.10f} dual={vd:.10f} "
              f"|dif|={err:.1e} (Phi*/pi={Phi / np.pi:.4f}, Delta*={Delta:.4f})"
              f"{chk.status_note(status)}")
    per, allw = u1_eigs(blocks)
    rank = int(np.sum(allw > RANK_TOL))
    weights = [float(np.real(np.trace(b))) for b in blocks]
    wtxt = " ".join(f"k{k}:{w:.4f}" for k, w in enumerate(weights) if w > RANK_TOL)
    etxt = " ".join(f"{x:.3e}" for x in allw[:rank + 1])
    chk.info(f"rango {rank}; pesos por sector {wtxt}; autovalores {etxt}")


def main():
    chk = Checker("primalU1")
    print("== primal_u1 vs dual.v_free (n = 3..7, cinco ángulos)")
    for n in range(3, 8):
        for Theta, lab in zip(ANGLES, ANGLE_LABELS):
            run_case(chk, n, Theta, lab, TOL)
    print("== n = 7 dentro de la ventana de desanclaje")
    for t in (0.985, 0.99, 0.995):
        run_case(chk, 7, t * np.pi, f"{t}pi", WINDOW_TOL)
    chk.finish()


if __name__ == "__main__":
    main()

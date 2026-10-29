"""Testigo óptimo de n = 7 como matriz explícita de 128x128 (sección sec:reduccion).

Para Theta/pi = 0.985, 0.99, 0.995 (dentro de la ventana de desanclaje) se toma
el óptimo (Phi*, Delta*) de holonomy.dual.v_free con holonomy.xxz.eps y se
construye, como en la prueba de la Proposición prop:xxz,
    W = alpha A + beta B + c C,
    A = -sum(XX+YY), B = sum(XY-YX), C = -sum ZZ (sumas sobre las 7 aristas),
    alpha = r cos(chi), beta = r sin(chi), c = r Delta, chi = Phi*/n,
    r = 1 / (n (Delta + 2 cos(chi - Theta/n))).
Comprueba que lambda_max(W), calculado en todo el espacio de 2^7 estados (no
solo en los sectores de magnetización mínima), coincide con el valor dual, y
que la normalización n(2 alpha cos(phi) + 2 beta sin(phi) + c) vale 1.

Uso: python scripts/witness_check.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import Checker, edge_operators

import numpy as np

from holonomy import dual, xxz

N = 7
TOL = 1e-12


def main():
    chk = Checker("witness_check")
    ops = [edge_operators(N, i) for i in range(N)]
    A = sum(o[0] for o in ops)
    B = sum(o[1] for o in ops)
    C = sum(o[2] for o in ops)
    herm = max(np.max(np.abs(M - M.conj().T)) for M in (A, B, C))
    chk.check(herm < 1e-15, f"A, B, C hermíticos de {A.shape[0]}x{A.shape[1]} (err {herm:.1e})")
    for t in (0.985, 0.99, 0.995):
        Theta = t * np.pi
        phi = Theta / N
        vd, Phi, Delta = dual.v_free(xxz.eps, N, Theta)
        if not np.isfinite(Delta):
            chk.check(False, f"Theta={t}pi: Delta* infinito (testigo de Ising), no esperado en n=7")
            continue
        chi = Phi / N
        r = 1.0 / (N * (Delta + 2 * np.cos(chi - phi)))
        alpha, beta, c = r * np.cos(chi), r * np.sin(chi), r * Delta
        W = alpha * A + beta * B + c * C
        w = np.linalg.eigvalsh(W)
        lmax = w[-1]
        norm = N * (2 * alpha * np.cos(phi) + 2 * beta * np.sin(phi) + c)
        err = abs(lmax - vd)
        chk.check(err < TOL, f"Theta={t}pi: Phi*/pi={Phi / np.pi:.6f} Delta*={Delta:.6f} "
                  f"lambda_max(W)={lmax:.15f} dual={vd:.15f} |dif|={err:.1e} "
                  f"(gap al 2º: {lmax - w[-2]:.2e})")
        chk.check(abs(norm - 1) < 1e-15, f"Theta={t}pi: normalización n(2a cos+2b sin+c)-1 = {norm - 1:.1e}")
        # Comprobación adicional: el flujo está fuera de la cúspide (Phi* > 0).
        chk.check(Phi > 1e-3, f"Theta={t}pi: flujo desanclado de la cúspide (Phi*/pi={Phi / np.pi:.6f} > 0)")
    chk.finish()


if __name__ == "__main__":
    main()

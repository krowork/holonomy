"""Optimalidad global de la cúspide (sección 5.3, Lema 6 y Teorema 7).

Comprueba numéricamente lo que el Teorema de la cúspide supone o predice:
  1. Hipótesis (H): en (Delta*, Phi_c) el estado fundamental en todo el espacio
     es un doblete (n par) o un cuadruplete (n impar, S^z = +-1/2), separado
     por un gap, y sus dos estados de un mismo sector S^z tienen momentos
     distintos. n = 6..13, en d = 0, pi/(2n) y pi/n.
  2. Estacionariedad en Delta: Delta* + 2 cos d = eps / eps_Delta.
  3. Pesos del óptimo primal: p_-/+ = 1/2 -/+ eps_Delta sin d / (n s_n), frente
     a los autovalores del SDP primal reducido por U(1) en n = 6 y 7.
  4. La ventana es un intervalo: Delta*(d) crece con d (se demuestra) y
     R(Delta*(d)) - sin d, con R = n s_n / (2 eps_Delta), cambia de signo una
     sola vez en [0, pi/n] (n = 6..14, 17 puntos).

Uso: python scripts/cusp.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS

import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag

from holonomy import dual, xxz
from holonomy.report import Checker
from holonomy.sdp import numerical_rank, primal_u1

eps = xxz.eps
H_DELTA = 1e-5  # paso de la derivada central en Delta


def translation(n, M):
    """Traslación s -> s desplazado un sitio, en el sector de M giros."""
    states = [s for s in range(2 ** n) if bin(s).count("1") == M]
    idx = {s: k for k, s in enumerate(states)}
    full = 2 ** n - 1
    cols = range(len(states))
    rows = [idx[((s << 1) | (s >> (n - 1))) & full] for s in states]
    return sp.csr_matrix((np.ones(len(states)), (rows, list(cols))), shape=(len(states),) * 2)


def cusp_data(n, d):
    """(Theta, Delta*, eps, eps_Delta, s_n) en la cúspide a distancia d."""
    Pc = dual.cusp_flux(n)
    Theta = abs(Pc - n * d)
    _, D = dual.v_cusp(eps, n, Theta, xatol=1e-12)
    e = eps(n, D, Pc)
    eD = (eps(n, D + H_DELTA, Pc) - eps(n, D - H_DELTA, Pc)) / (2 * H_DELTA)
    s = dual.cusp_slope_fd(eps, n, D)
    return Theta, D, e, eD, s


def hypothesis(ck):
    for n in range(6, 14):
        Pc = dual.cusp_flux(n)
        for d in (0.0, np.pi / (2 * n), np.pi / n):
            _, D, _, _, _ = cusp_data(n, d)
            levels = np.sort(np.concatenate(
                [np.linalg.eigvalsh(xxz.hamiltonian(n, M, D, Pc).toarray())[:4] for M in range(n + 1)]))
            E0 = levels[0]
            deg = int(np.sum(levels < E0 + 1e-9))
            gap = levels[deg] - E0
            M = n // 2
            w, V = np.linalg.eigh(xxz.hamiltonian(n, M, D, Pc).toarray())
            P = V[:, w < E0 + 1e-9]
            T = translation(n, M).toarray()
            k = np.sort(np.angle(np.linalg.eigvals(P.conj().T @ T @ P)))
            expected = 2 if n % 2 == 0 else 4
            ok = deg == expected and gap > 0.1 and P.shape[1] == 2 and abs(k[1] - k[0]) > 1e-6
            ck.check(ok, f"(H) n={n} d={d * n / np.pi:.2f} pi/n Delta*={D:.5f}: degeneración {deg} "
                         f"(esperada {expected}), gap {gap:.3f}, momentos/pi {np.round(k / np.pi, 3)}")


def weights(ck):
    for n, t in ((6, 1.0), (6, 0.5), (6, 0.25), (6, 0.05), (7, 0.0), (7, 0.5), (7, 0.9), (7, 0.98)):
        d = abs(dual.cusp_flux(n) - t * np.pi) / n
        _, D, e, eD, s = cusp_data(n, d)
        ck.close(D + 2 * np.cos(d), e / eD, 1e-6, f"n={n} Theta/pi={t}: Delta*+2cos d = eps/eps_Delta")
        p_minus = 0.5 - eD * np.sin(d) / (n * s)
        _, blocks, status = primal_u1(n, t * np.pi, return_blocks=True, **CLARABEL_OPTS)
        _, w = numerical_rank(block_diag(*blocks), tol=1e-6)
        # n par: dos autovalores (p+, p-); n impar: cuatro, por pares S^z = +-1/2.
        primal_minus = w[1] if n % 2 == 0 else w[2] + w[3]
        note = "" if status == "optimal" else f" [estado: {status}]"
        ck.close(p_minus, primal_minus, 1e-4, f"n={n} Theta/pi={t}: peso p_- predicho vs primal{note}")


def interval(ck):
    for n in range(6, 15):
        ds = np.linspace(0.0, np.pi / n, 17)
        Ds, G = [], []
        for d in ds:
            _, D, _, eD, s = cusp_data(n, d)
            Ds.append(D)
            G.append(n * s / (2 * eD) - np.sin(d))
        changes = int(np.sum(np.diff(np.sign(G)) != 0))
        ck.check(bool(np.all(np.diff(Ds) > 0)), f"n={n}: Delta*(d) creciente en [0, pi/n]")
        ck.check(changes == 1 and G[0] > 0 > G[-1],
                 f"n={n}: R(Delta*) - sin d cambia de signo {changes} vez (de {G[0]:.4f} a {G[-1]:.4f})")


def main():
    ck = Checker()
    hypothesis(ck)
    weights(ck)
    interval(ck)
    ck.finish()


if __name__ == "__main__":
    main()

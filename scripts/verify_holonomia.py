"""Verificación del teorema del anillo de 4 qubits (sección sec:anillo4).

Comprueba, con el SDP primal completo (rho de 16x16, todas las marginales
fijadas), que v_max(theta) = f(theta) = 2/(3 + sin(theta/2)):
  1. barrido theta = k pi/8, k = 0..8, en el gauge uniforme;
  2. 34 holonomías de Haar (R_e aleatorias por arista, semilla fija), con theta
     el ángulo de la holonomía H = R_01 R_12 R_23 R_30;
  3. rango numérico del óptimo: 1 en theta = 0 y 2 en el resto;
  4. testigo W del paso 1 (lambda_max(W) = f, multiplicidad 2, normalización)
     y estado óptimo rho_* del paso 3(e) (las 15 correlaciones de cada arista
     coinciden con las de rho_e(f)).

Uso: python scripts/verify_holonomia.py
"""
import _common  # noqa: F401  (fija OMP_NUM_THREADS antes de numpy)
from _common import CLARABEL_OPTS, Checker, edge_mismatch, edge_operators, uniform_gauge

import numpy as np
from scipy.spatial.transform import Rotation

from holonomy.paulis import holonomy, rotation_angle
from holonomy.sdp import numerical_rank, primal

N = 4
SDP_TOL = 1e-7      # tolerancia de los SDP (el artículo da 5e-9 y 2.3e-8)
EXACT_TOL = 1e-13   # tolerancia de las comprobaciones algebraicas
RANK_TOL = 1e-5     # autovalores por debajo cuentan como nulos
HAAR_SEED = 20240611
HAAR_CASES = 34


def f(theta):
    return 2.0 / (3.0 + np.sin(theta / 2))


def basis_state(bits):
    """|b_0 b_1 b_2 b_3> con el qubit 0 como bit más significativo."""
    vec = np.zeros(2 ** N, dtype=complex)
    vec[int(bits, 2)] = 1.0
    return vec


def witness(theta):
    """W = a sum(A_i + B_i) + c sum C_i del paso 1. Devuelve (W, a, c, kappa)."""
    phi = theta / N
    kappa = np.cos(np.pi / 4 - phi)
    J = kappa / (4 * (1 + kappa ** 2))
    a = J / np.sqrt(2)
    c = (1 - kappa ** 2) / (4 * (1 + kappa ** 2))
    W = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N):
        A, B, C = edge_operators(N, i)
        W += a * (A + B) + c * C
    return W, a, c, kappa


def optimal_state(theta):
    """rho_* = p_0 |K=0><K=0| + p_pi |K=pi><K=pi| del paso 3(e)."""
    phi = theta / N
    kappa = np.cos(np.pi / 4 - phi)
    Nn = 1 / np.sqrt(2 * (1 + kappa ** 2))
    N0, N1 = basis_state("1010"), basis_state("0101")
    A1, A2, A3, A4 = (basis_state(b) for b in ("1100", "0110", "0011", "1001"))
    k0 = Nn * (N0 + N1) - kappa * Nn / np.sqrt(2) * (A1 + A2 + A3 + A4)
    kpi = Nn * (-N0 + N1) + 1j * kappa * Nn / np.sqrt(2) * (A1 - A2 + A3 - A4)
    p0 = np.cos(phi) / (np.cos(phi) + np.sin(phi))
    ppi = np.sin(phi) / (np.cos(phi) + np.sin(phi))
    rho = p0 * np.outer(k0, k0.conj()) + ppi * np.outer(kpi, kpi.conj())
    return rho, k0, kpi


def main():
    chk = Checker("verify_holonomia")

    # 1. Barrido en el gauge uniforme y 3. rango numérico.
    print("== Barrido theta = k pi/8 (gauge uniforme)")
    sweep_err = 0.0
    for k in range(9):
        theta = k * np.pi / 8
        v, rho, status = primal(N, uniform_gauge(N, theta), return_state=True, **CLARABEL_OPTS)
        err = abs(v - f(theta))
        sweep_err = max(sweep_err, err)
        chk.check(err < SDP_TOL, f"barrido k={k}: v={v:.12f} f={f(theta):.12f} "
                  f"err={err:.1e}{chk.status_note(status)}")
        rank, w = numerical_rank(rho, tol=RANK_TOL)
        expected = 1 if k == 0 else 2
        chk.check(rank == expected,
                  f"rango k={k}: {rank} (esperado {expected}); autovalores "
                  + " ".join(f"{x:.2e}" for x in w[:4]))
    chk.info(f"error máximo del barrido: {sweep_err:.2e} (artículo: 5e-9)")

    # 2. Holonomías de Haar.
    print(f"== {HAAR_CASES} holonomías de Haar (semilla {HAAR_SEED})")
    rots = Rotation.random(N * HAAR_CASES, random_state=HAAR_SEED).as_matrix()
    haar_err = 0.0
    for m in range(HAAR_CASES):
        Rs = [rots[N * m + e] for e in range(N)]
        theta = rotation_angle(holonomy(Rs))
        v, _, status = primal(N, Rs, return_state=True, **CLARABEL_OPTS)
        err = abs(v - f(theta))
        haar_err = max(haar_err, err)
        chk.check(err < SDP_TOL, f"Haar {m:2d}: theta/pi={theta / np.pi:.6f} v={v:.12f} "
                  f"f={f(theta):.12f} err={err:.1e}{chk.status_note(status)}")
    chk.info(f"error máximo Haar: {haar_err:.2e} (artículo: 2.3e-8)")

    # 4. Testigo y estado óptimo explícitos.
    print("== Testigo W (paso 1) y estado rho_* (paso 3e), theta = k pi/5")
    for k in range(6):
        theta = k * np.pi / 5
        phi = theta / N
        W, a, c, kappa = witness(theta)
        w = np.linalg.eigvalsh(W)
        lmax = w[-1]
        mult = int(np.sum(np.abs(w - lmax) < 1e-10))
        norm = N * (2 * a * (np.cos(phi) + np.sin(phi)) + c)
        chk.check(abs(lmax - f(theta)) < EXACT_TOL and mult == 2 and abs(norm - 1) < EXACT_TOL,
                  f"testigo theta={k}pi/5: lambda_max={lmax:.16f} f={f(theta):.16f} "
                  f"err={abs(lmax - f(theta)):.1e}, multiplicidad {mult}, "
                  f"4(2a(cos+sin)+c)-1={norm - 1:.1e}")
        rho, k0, kpi = optimal_state(theta)
        eig_err = max(np.linalg.norm(W @ k0 - f(theta) * k0), np.linalg.norm(W @ kpi - f(theta) * kpi))
        orth = max(abs(np.vdot(k0, k0) - 1), abs(np.vdot(kpi, kpi) - 1), abs(np.vdot(k0, kpi)))
        corr_err = edge_mismatch(rho, N, f(theta), uniform_gauge(N, theta))
        chk.check(corr_err < EXACT_TOL and eig_err < EXACT_TOL and orth < EXACT_TOL
                  and abs(np.trace(rho) - 1) < EXACT_TOL,
                  f"rho_* theta={k}pi/5: 15 correlaciones x 4 aristas vs rho_e(f) err={corr_err:.1e}; "
                  f"|(W-f)K|={eig_err:.1e}; ortonormalidad {orth:.1e}; tr-1={abs(np.trace(rho) - 1):.1e}")

    chk.finish()


if __name__ == "__main__":
    main()

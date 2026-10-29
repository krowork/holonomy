"""Reducción del dual de n sitios al anillo XXZ retorcido (Proposición 4).

Construye el testigo simétrico W = alpha A + beta B + c C como matriz de 2^n x 2^n
con productos de Pauli y comprueba, para parámetros aleatorios, que

    lambda_max(W) = 4 r n eps_n(Delta, n chi),  alpha = r cos chi, beta = r sin chi, Delta = c/r,

con eps_n calculada por diagonalización exacta del anillo XXZ en todo el espacio.
"""
import numpy as np

from holonomy import xxz
from holonomy.paulis import pauli_string
from holonomy.report import Checker


def symmetric_terms(n):
    A = np.zeros((2 ** n, 2 ** n), dtype=complex)
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    for i in range(n):
        j = (i + 1) % n
        A -= pauli_string(n, ((i, 0), (j, 0))) + pauli_string(n, ((i, 1), (j, 1)))
        B += pauli_string(n, ((i, 0), (j, 1))) - pauli_string(n, ((i, 1), (j, 0)))
        C -= pauli_string(n, ((i, 2), (j, 2)))
    return A, B, C


def main():
    ck = Checker()
    rng = np.random.default_rng(20261028)
    for n in range(3, 9):
        A, B, C = symmetric_terms(n)
        worst = 0.0
        for _ in range(8):
            r = rng.uniform(0.2, 2.0)
            chi = rng.uniform(-np.pi, np.pi)
            Delta = rng.uniform(-0.9, 4.0)
            W = r * np.cos(chi) * A + r * np.sin(chi) * B + r * Delta * C
            lam = np.linalg.eigvalsh(W)[-1]
            Hfull = xxz.full_hamiltonian(n, Delta, n * chi)
            eps_full = -np.linalg.eigvalsh(Hfull)[0] / n
            worst = max(worst, abs(lam - 4 * r * n * eps_full))
        ck.check(worst < 1e-11, f"n={n}: lambda_max(W) = 4 r n eps_n en 8 puntos aleatorios (error máx {worst:.1e})")
    ck.finish()


if __name__ == "__main__":
    main()

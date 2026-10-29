"""Diagonalización por sectores de magnetización.

Comprueba que, para Delta > -1, el estado fundamental del anillo XXZ con flujo
está en los sectores de magnetización mínima o es el ferromagnético totalmente
polarizado (este último solo cerca de Delta = -1 y para n pequeño), de modo que
basta diagonalizar esos sectores (es lo que hace `holonomy.xxz.eps`). Se compara con el mínimo
sobre todos los sectores y, para n pequeño, con la matriz completa de 2^n.
"""
import numpy as np

from holonomy import xxz
from holonomy.report import Checker

DELTAS = (-0.99, -0.9, -0.7, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0, 10.0)
FLUXES = np.linspace(0.0, np.pi, 7)


def main():
    ck = Checker()
    for n in range(3, 13):
        worst = 0.0
        for D in DELTAS:
            for P in FLUXES:
                worst = max(worst, abs(xxz.eps(n, D, P) - xxz.eps_full(n, D, P)))
        ck.check(worst < 1e-12, f"n={n}: sectores mínimos = todos los sectores (error máx {worst:.1e})")
    for n in range(3, 9):
        worst = 0.0
        for D in DELTAS:
            for P in FLUXES:
                e_full = -np.linalg.eigvalsh(xxz.full_hamiltonian(n, D, P))[0] / n
                worst = max(worst, abs(xxz.eps(n, D, P) - e_full))
        ck.check(worst < 1e-12, f"n={n}: sectores = matriz completa de 2^n (error máx {worst:.1e})")
    ck.finish()


if __name__ == "__main__":
    main()

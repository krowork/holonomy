"""Ecuaciones de Bethe con flujo frente a diagonalización exacta (sección 5.4).

Para n = 4..14, cinco anisotropías en los tres regímenes (Delta < 1, = 1, > 1)
y siete flujos en [0, pi] (385 combinaciones):
  * la energía de la rama fundamental coincide con la diagonalización exacta;
  * para n impar, la otra rama del doblete es un autovalor del sector, y las
    dos ramas son degeneradas en Phi = 0;
  * la derivada en el flujo por diferenciación implícita coincide con
    diferencias centradas;
  * la pendiente en la cúspide s_n (implícita) coincide con la de diferencias
    laterales sobre la diagonalización exacta.
"""
import numpy as np

from holonomy import dual, xxz
from holonomy.bethe import BetheSolver
from holonomy.report import Checker

DELTAS = (0.3, 0.8, 1.0, 1.5, 3.0)
FLUXES = np.linspace(0.0, np.pi, 7)


def main():
    ck = Checker()
    count, worst_e, worst_other, worst_d = 0, 0.0, 0.0, 0.0
    for n in range(4, 15):
        bs = BetheSolver(n)
        for D in DELTAS:
            for P in FLUXES:
                count += 1
                worst_e = max(worst_e, abs(bs.eps(D, P) - xxz.eps(n, D, P)))
                if 0 < P < np.pi:  # en los extremos la derivada es lateral
                    b = bs.ground_branch()
                    _, dE = bs.energy_and_flux_derivative(D, P, b)
                    h = 1e-5
                    fd = (bs.energy(D, P + h, b) - bs.energy(D, P - h, b)) / (2 * h)
                    worst_d = max(worst_d, abs(dE - fd))
                if n % 2:
                    try:
                        Eo = bs.energy(D, P, +1)
                    except RuntimeError:
                        continue  # rapidez que escapa: la rama no existe aquí
                    spec = np.linalg.eigvalsh(xxz.hamiltonian(n, n // 2, D, P).toarray())
                    worst_other = max(worst_other, np.min(np.abs(spec - Eo)))
    ck.check(count == 385, f"{count} combinaciones (n = 4..14, 5 anisotropías, 7 flujos)")
    ck.check(worst_e < 1e-13, f"rama fundamental = diagonalización exacta (error máx {worst_e:.1e})")
    ck.check(worst_other < 1e-12, f"n impar: la otra rama del doblete es autovalor del sector (error máx {worst_other:.1e})")
    ck.check(worst_d < 1e-8, f"dE/dPhi implícita = diferencias centradas en flujos interiores (error máx {worst_d:.1e})")

    for n in (5, 7, 9, 11, 13):
        bs = BetheSolver(n)
        gap = abs(bs.energy(1.0, 0.0, +1) - bs.energy(1.0, 0.0, -1))
        ck.check(gap < 1e-12, f"n={n}: ramas degeneradas en la cúspide Phi = 0 (diferencia {gap:.1e})")

    worst_s = 0.0
    for n in range(6, 15):
        bs = BetheSolver(n)
        for D in (0.8, 1.0, 1.3):
            worst_s = max(worst_s, abs(bs.cusp_slope(D) - dual.cusp_slope_fd(xxz.eps, n, D)))
    ck.check(worst_s < 1e-8, f"pendiente en la cúspide: implícita = diferencias laterales (error máx {worst_s:.1e})")
    ck.finish()


if __name__ == "__main__":
    main()

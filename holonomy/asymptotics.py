"""Magnitudes de las secciones 5.2 y 6: anchura de la ventana, coeficiente de
1/n^2 y rigidez finita, para un n dado y un `eps` (diagonalización o Bethe).

    coef   = n^2 [v(n, 0) - v(n, pi)]
    D_n    = n^2 s_n / pi, con s_n la pendiente en la cúspide evaluada en el
             Delta* del borde de la ventana
    1 + r_n = |coef| / (2/3 pi^2 D_n)
    psi_pred/pi = 1 - 3 D_n / (2 eps_n)   (sección 6.3)
"""
import numpy as np

from . import dual

FIELDS = ("n", "psi_over_pi", "delta_edge", "coef", "D_n", "one_plus_r",
          "psi_pred_over_pi", "delta_star_cusp_extreme", "v0", "vpi")


def vmax_extreme(eps, n, Theta, xatol=1e-12):
    """v_max en Theta = 0 o pi: el menor entre cúspide y punto liso."""
    return min(dual.v_cusp(eps, n, Theta, xatol=xatol)[0],
               dual.v_smooth(eps, n, Theta, xatol=xatol)[0])


def row(eps, n, slope=None):
    """Una fila de la tabla 3 (o de la tabla 1 para n pequeño)."""
    slope_fn = slope if slope is not None else (lambda m, D: dual.cusp_slope_fd(eps, m, D))
    v0 = vmax_extreme(eps, n, 0.0)
    vpi = vmax_extreme(eps, n, np.pi)
    coef = n * n * (v0 - vpi)
    psi, D_edge = dual.window_edge(eps, n, slope=slope)
    s = slope_fn(n, D_edge)
    Dn = n * n * s / np.pi
    one_plus_r = abs(coef) / (2 / 3 * np.pi ** 2 * Dn)
    psi_pred = 1 - 3 * Dn / (2 * eps(n, D_edge, dual.cusp_flux(n)))
    # Delta* en la cúspide en el extremo opuesto al punto liso (Theta = pi para
    # n par, Theta = 0 para n impar): es el de la discusión de la sección 6.2.
    Theta_c = dual.cusp_flux(n)
    D_star = dual.v_cusp(eps, n, Theta_c, xatol=1e-12)[1]
    return dict(n=n, psi_over_pi=psi / np.pi, delta_edge=D_edge, coef=coef, D_n=Dn,
                one_plus_r=one_plus_r, psi_pred_over_pi=psi_pred,
                delta_star_cusp_extreme=D_star, v0=v0, vpi=vpi)


def psi_infinity_over_pi():
    """psi_inf/pi = 1 - 3/(8 (ln 2 - 1/4)), con D = 1/4 y eps -> ln 2 - 1/4."""
    return 1 - 3 / (8 * (np.log(2) - 0.25))

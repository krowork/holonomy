"""Objetivo dual del anillo de n sitios (Proposición 4) y la estructura de
cúspide y ventana de desanclaje (sección 5.3).

Todas las funciones reciben `eps(n, Delta, Phi)`: puede ser
`holonomy.xxz.eps` (diagonalización exacta) o `holonomy.bethe.eps`.

    F(Delta, Phi) = 4 eps_n(Delta, Phi) / (Delta + 2 cos((Phi - Theta)/n)),
    v_max(n, Theta) = inf F, con el flujo reducido a Phi en [0, pi].
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

DELTA_MIN = -0.99  # para Delta <= -1 el cociente es > 1 (Lema 5)
DELTA_MAX = 60.0


def cusp_flux(n):
    return np.pi if n % 2 == 0 else 0.0


def smooth_flux(n):
    return np.pi - cusp_flux(n)


def objective(eps, n, Theta, Delta, Phi):
    den = Delta + 2 * np.cos((Phi - Theta) / n)
    if den <= 0:
        return np.inf
    return 4 * eps(n, Delta, Phi) / den


def _bracket(f, x0, lo, hi, step=0.25, grow=2.0):
    """Encierra el mínimo de una función cuasiconvexa partiendo de x0."""
    fx0 = f(x0)
    x1 = min(x0 + step, hi)
    f1 = f(x1)
    if f1 < fx0:  # hacia arriba
        a, b, fb = x0, x1, f1
        while b < hi:
            c = min(b + step, hi)
            fc = f(c)
            if fc >= fb:
                return a, c
            a, b, fb = b, c, fc
            step *= grow
        return a, hi
    # hacia abajo: acercarse a lo sin cruzarlo
    a, b, fb = x1, x0, fx0
    while True:
        c = b - step if b - step > lo else lo + (b - lo) / 4
        fc = f(c)
        if fc >= fb:
            return c, a
        a, b, fb = b, c, fc
        step *= grow
        if b - lo < 1e-9:
            return lo, a


def v_at_flux(eps, n, Theta, Phi, xatol=1e-10, bounds=None):
    """min_Delta F(Delta, Phi). Devuelve (v, Delta*).

    eps es convexa en Delta y el denominador es afín, así que F es
    cuasiconvexa en Delta: se encierra el mínimo partiendo de Delta = 1 (donde
    vive el óptimo para n grande) y se refina con Brent acotado.
    """
    lo = max(DELTA_MIN, -2 * np.cos((Phi - Theta) / n) + 1e-9)

    def f(D):
        return objective(eps, n, Theta, D, Phi)

    a, b = bounds if bounds is not None else _bracket(f, 1.0, lo, DELTA_MAX)
    res = minimize_scalar(f, bounds=(a, b), method="bounded", options=dict(xatol=xatol))
    v, D = float(res.fun), float(res.x)
    if bounds is None and ising_limit(n) < v:
        return ising_limit(n), np.inf
    return v, D


def ising_limit(n):
    """Límite Delta -> infinito del objetivo (testigo de Ising), independiente
    del flujo: 4 eps/Delta -> 4 (n - 2)/(4n) para n impar (un enlace frustrado)
    y 1 para n par. Solo es el ínfimo en el triángulo, donde vale 1/3."""
    return (n - 2) / n if n % 2 else 1.0


def v_cusp(eps, n, Theta, **kw):
    """Fórmula de un parámetro: flujo en la cúspide."""
    return v_at_flux(eps, n, Theta, cusp_flux(n), **kw)


def v_smooth(eps, n, Theta, **kw):
    """Flujo en el punto liso."""
    return v_at_flux(eps, n, Theta, smooth_flux(n), **kw)


def v_free(eps, n, Theta, grid=33, xatol=1e-9):
    """Mínimo completo sobre (Delta, Phi): barrido del flujo en [0, pi] y
    refinamiento alrededor del mejor punto. Devuelve (v, Phi*, Delta*)."""
    Phis = np.linspace(0.0, np.pi, grid)
    vals = [v_at_flux(eps, n, Theta, P)[0] for P in Phis]
    k = int(np.argmin(vals))
    a, b = Phis[max(k - 1, 0)], Phis[min(k + 1, grid - 1)]
    res = minimize_scalar(lambda P: v_at_flux(eps, n, Theta, P)[0],
                          bounds=(a, b), method="bounded", options=dict(xatol=xatol))
    cands = [(vals[k], Phis[k]), (float(res.fun), float(res.x))]
    for P in (0.0, np.pi):
        cands.append((v_at_flux(eps, n, Theta, P)[0], P))
    v, P = min(cands)
    return v, P, v_at_flux(eps, n, Theta, P)[1]


# ---------------------------------------------------------------------------
# Pendiente en la cúspide y criterio de borde (Proposición 6)
# ---------------------------------------------------------------------------

def cusp_slope_fd(eps, n, Delta, h=2e-3):
    """s_n = d eps/d|Phi - Phi_c| en la cúspide, por diferencias laterales con
    extrapolación de Richardson (error O(h^3))."""
    Pc = cusp_flux(n)
    sgn = -1.0 if n % 2 == 0 else 1.0  # dirección hacia el interior de [0, pi]
    e0 = eps(n, Delta, Pc)

    def D(hh):
        return (eps(n, Delta, Pc + sgn * hh) - e0) / hh

    d1, d2, d4 = D(h), D(h / 2), D(h / 4)
    r1, r2 = 2 * d2 - d1, 2 * d4 - d2
    return (4 * r2 - r1) / 3


def edge_function(eps, n, Theta, slope=None):
    """g_n(Theta) de la Proposición 6. slope(n, Delta) por defecto: diferencias."""
    v, Ds = v_cusp(eps, n, Theta)
    d = abs(cusp_flux(n) - Theta) / n
    s = slope(n, Ds) if slope is not None else cusp_slope_fd(eps, n, Ds)
    return s * (Ds + 2 * np.cos(d)) - (2 / n) * eps(n, Ds, cusp_flux(n)) * np.sin(d), Ds


def window_edge(eps, n, slope=None, xtol=1e-12, scan=48, marginal_tol=1e-10):
    """Anchura psi(n) de la ventana de desanclaje, medida desde el punto liso.

    Devuelve (psi, Delta* en el borde). psi = 0 si g_n >= -marginal_tol en el
    punto liso (n = 4 y n = 5 son marginales: g_n = 0 exactamente allí).
    """
    smooth = smooth_flux(n)

    def g(psi):
        return edge_function(eps, n, abs(smooth - psi), slope)[0]

    g0 = g(0.0)
    if g0 >= -marginal_tol:
        return 0.0, edge_function(eps, n, smooth, slope)[1]
    # g < 0 junto al punto liso; buscar el primer cambio de signo.
    grid = np.linspace(0.0, np.pi, scan + 1)
    prev = 0.0
    for x in grid[1:]:
        if g(x) >= 0:
            psi = brentq(g, prev, x, xtol=xtol)
            return psi, edge_function(eps, n, abs(smooth - psi), slope)[1]
        prev = x
    raise RuntimeError(f"no se encontró el borde para n={n}")


def flux_derivative_fd(eps, n, Delta, Phi, h=1e-4):
    return (eps(n, Delta, Phi + h) - eps(n, Delta, Phi - h)) / (2 * h)

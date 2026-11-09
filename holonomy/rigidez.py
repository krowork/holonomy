"""Cota tipo Kohn: cuánto puede variar v_max al girar la holonomía.

Enunciado (grafo cualquiera, datos abelianos)
---------------------------------------------
Sea G un grafo con aristas orientadas e = (i, j) y datos R_e = R_z(phi_e).
La marginal de la arista es rho_e(v) = I/4 + v tau_e(phi), con
tau_e(phi) = (1 x U(phi_e)) tau (1 x U(phi_e))^+, tau la parte sin traza
del estado de Werner y U(a) la rotación de ángulo a alrededor de z (en el
qubit de llegada j). Un testigo es W = sum_e W_e, con W_e sin traza sobre los
dos qubits de e, y su normalización es N_phi(W) = sum_e tr(W_e tau_e(phi)).
Si rho tiene esas marginales, tr(rho W) = v N_phi(W) <= lambda_max(W), de
modo que v(phi) <= lambda_max(W) para todo W con N_phi(W) = 1 (dualidad
débil), y por dualidad fuerte (Slater, rho = I/2^n) el mínimo se alcanza:
v(phi) = lambda_max(W) para un W óptimo.

Para delta = phi' - phi se define W[delta] = sum_e (1 x U(delta_e)) W_e
(1 x U(delta_e))^+. En la forma de la Proposición 4 de \\cite{anillos},
W_e = alpha_e A_e + beta_e B_e + c_e C_e con alpha_e = r_e cos chi_e y
beta_e = r_e sin chi_e, es el mismo testigo con las mismas r_e, c_e y fases
chi_e + delta_e.

Proposición. Si W es óptimo para phi y phi' = phi + delta, entonces
N_phi'(W[delta]) = N_phi(W) = 1 y
    v(phi') - v(phi) <= lambda_max(W[delta]) - lambda_max(W),
y, simétricamente, v(phi) - v(phi') <= lambda_max(W'[-delta]) -
lambda_max(W') con W' óptimo para phi'. lambda_max(W[delta]) solo depende
de las holonomías de delta: si delta difiere en un gauge de un giro DeltaPsi
concentrado en una arista e0, es W con un flujo DeltaPsi insertado por los
ciclos que pasan por e0, y es 2 pi-periódico en DeltaPsi.

Demostración.
(a) Normalización. Las rotaciones alrededor de z conmutan (datos abelianos),
    así que U(delta_e)^+ U(phi'_e) = U(phi_e) y
    tr(W[delta]_e tau_e(phi')) = tr(W_e tau_e(phi)) arista a arista.
(b) Cota. W[delta] es admisible para phi', luego por dualidad débil
    v(phi') <= lambda_max(W[delta]); y v(phi) = lambda_max(W) por ser W
    óptimo. Restando se obtiene la desigualdad. La simétrica es la misma
    con los papeles de phi y phi' intercambiados.
(c) Invariancia gauge. Los datos y tau son invariantes bajo la rotación
    global alrededor de z, así que promediando cada W_e sobre ese grupo se
    obtiene otro testigo óptimo con la misma normalización; se puede suponer
    que cada W_e conmuta con Z_i + Z_j. Si delta_e = g_j - g_i + DeltaPsi
    [e = e0], con V = prod_k U_k(g_k) se tiene
    V W_e V^+ = (1 x U(g_j - g_i)) (U(g_i) x U(g_i)) W_e (...)^+
              = (1 x U(g_j - g_i)) W_e (...)^+,
    de modo que W[delta] = V W[DeltaPsi 1_e0] V^+ y ambos tienen el mismo
    espectro. Como U(2 pi) = -1, la conjugación por U(DeltaPsi + 2 pi) es la
    misma que por U(DeltaPsi): un giro de 2 pi es un gauge.  QED

Para el anillo de n sitios en el gauge uniforme, phi_e = Theta/n y el giro
Theta0 -> Theta1 es delta_e = (Theta1 - Theta0)/n: el testigo óptimo de
Theta0, de flujo Phi0 = n chi y anisotropía Delta0, pasa a flujo
Phi0 + Theta1 - Theta0 con el mismo denominador Delta0 + 2 cos((Phi0 -
Theta0)/n), y
    v(n, Theta1) <= 4 eps_n(Delta0, Phi0 + Theta1 - Theta0)
                    / (Delta0 + 2 cos((Phi0 - Theta0)/n)).
A segundo orden en el giro, v(Theta0 + d) - v(Theta0) <= -(2/den0)
(D_W / n^2) d^2 + o(d^2), con D_W = -n^2 d^2 eps_n/d Phi^2 la rigidez del
testigo óptimo. A diferencia de la cota de holonomía pequeña de
\\cite{escaleras} (Proposición 14), que usa |eps(Phi) - eps(Phi')| <= 1/n y es
de orden 1/n, esta usa la respuesta del propio testigo al giro, de orden
D_W/n^2.

Convención de signos: `dual.v_free` devuelve el flujo óptimo en [0, pi], que
para Theta en [0, pi] es un minimizador genuino del objetivo de la
Proposición 4 (la reflexión Phi -> -Phi y la traslación Phi -> 2 pi - Phi
nunca mejoran el denominador en ese rango), así que no hace falta cambiar
Phi0 de signo.
"""
import numpy as np

from holonomy import dual, xxz


def twist_response(eps, n, Delta, Phi, Psi):
    """eps_n(Delta, Phi + Psi) - eps_n(Delta, Phi): respuesta del testigo de
    anisotropía Delta y flujo Phi a un giro de holonomía Psi."""
    if Psi == 0:
        return 0.0
    return eps(n, Delta, Phi + Psi) - eps(n, Delta, Phi)


def ring_kohn_bound(n, Theta0, Theta1, eps=xxz.eps_full, grid=33):
    """Cota tipo Kohn del anillo de n sitios. Devuelve (v(n, Theta1), cota).

    (v0, Phi0, Delta0) = v_free(eps, n, Theta0) da el testigo óptimo en
    Theta0; se le inserta el giro Theta1 - Theta0 (flujo Phi0 + Theta1 -
    Theta0, mismo denominador). Theta0 y Theta1 en [0, pi]: v_free reduce el
    flujo a [0, pi], válido solo en ese rango (v es par y 2 pi-periódica en
    Theta; el llamante reduce antes).
    """
    for T in (Theta0, Theta1):
        if not -1e-12 <= T <= np.pi + 1e-12:
            raise ValueError(f"Theta = {T} fuera de [0, pi]")
    v1 = dual.v_free(eps, n, Theta1, grid=grid)[0]
    _, Phi0, Delta0 = dual.v_free(eps, n, Theta0, grid=grid)
    if not np.isfinite(Delta0):
        # Testigo de Ising (límite Delta -> infinito): sin parte XY, el giro no
        # lo cambia.
        return v1, dual.ising_limit(n)
    den = Delta0 + 2 * np.cos((Phi0 - Theta0) / n)
    return v1, 4 * eps(n, Delta0, Phi0 + Theta1 - Theta0) / den

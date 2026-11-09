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
lambda_max(W') con W' óptimo para phi'. Con W invariante bajo la rotación
global alrededor de z, que siempre puede elegirse así por promedio (paso (c)),
lambda_max(W[delta]) solo depende de las holonomías de delta: si delta
difiere en un gauge de un giro DeltaPsi concentrado en una arista e0, es W
con un flujo DeltaPsi insertado por los ciclos que pasan por e0, y es
2 pi-periódico en DeltaPsi.

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
Si Phi0 no está en la cúspide (eps derivable en Phi0), la estacionariedad en
Phi da 4 eps'(Phi0)/den0 = v'(Theta0), y a segundo orden en el giro
    v(Theta0 + d) - v(Theta0) - v'(Theta0) d <= -(2/den0)(D_W/n^2) d^2 + o(d^2),
con D_W = -n^2 d^2 eps_n/dPhi^2 (Delta0, Phi0) la rigidez del testigo óptimo.
En el punto liso Theta0 = Phi_s el término lineal se anula (Phi0 = Phi_s).

Comparación con cotas anteriores. La cota de holonomía pequeña de
\\cite{escaleras} (Proposición 14) acota la norma de lo que cambia en cada
arista, 2 r_e |e^{i phi_e} - 1| <= 2 r_e phi_max, y suma con el lema de pesos:
es de orden phi_max = O(1/n). La desigualdad |eps(Phi) - eps(Phi')| <= 1/n,
que da una cota del mismo orden en el anillo, es el Lema 5(i) de
\\cite{anillos}. La cota de aquí usa la respuesta del propio testigo óptimo
al giro, de orden D_W/n^2.

Convención de signos: `dual.v_free` devuelve el flujo óptimo en [0, pi], que
para Theta en [0, pi] es un minimizador genuino del objetivo de la
Proposición 4 (la reflexión Phi -> -Phi y la traslación Phi -> 2 pi - Phi
nunca mejoran el denominador en ese rango), así que no hace falta cambiar
Phi0 de signo.

Escalera periódica (ladder.LadderDual)
--------------------------------------
Es el caso escalera de la construcción anterior. En ladder.abelian_phases el
twist psi se reparte sumando psi/L a cada arista de las dos patas (peldaños
intactos), y el testigo de LadderDual tiene en cada pata la fase
chi = arg(alpha + i beta), con N = sum_e (2 r_e cos(chi_e - phi_e) + c_e).
El giro delta_e = Psi/L en las patas es, pues, rotar (alpha_b, beta_b) y
(alpha_t, beta_t) un ángulo +Psi/L (`ladder_twist_y`): conserva N entre los
datos de psi y los de psi + Psi (con el mismo signo; lo comprueba
test_ladder_giro_normalizacion) y deja los flujos de plaqueta del testigo; las
holonomías de las dos patas suben Psi. Con y óptimo en psi0,
    v(psi0 + Psi) <= lambda_max(W[Psi]),
y la respuesta R(Psi) = lambda_max(W[Psi]) - lambda_max(W) acota
v(psi0 + Psi) - v(psi0). El testigo de LadderDual solo tiene términos A, B y
C, invariantes bajo la rotación global alrededor de z, así que R es
2 pi-periódica en Psi (paso (c)).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from holonomy import dual, xxz


def twist_response(eps, n, Delta, Phi, Psi):
    """eps_n(Delta, Phi + Psi) - eps_n(Delta, Phi): respuesta del testigo de
    anisotropía Delta y flujo Phi a un giro de holonomía Psi."""
    return eps(n, Delta, Phi + Psi) - eps(n, Delta, Phi)


def kohn_bound_from_witness(eps, n, Theta0, Phi0, Delta0, Theta1):
    """Cota en Theta1 del testigo de flujo Phi0 y anisotropía Delta0,
    admisible en Theta0, tras insertarle el giro Theta1 - Theta0:
    4 eps_n(Delta0, Phi0 + Theta1 - Theta0)/(Delta0 + 2 cos((Phi0 - Theta0)/n)).
    Si Delta0 es infinito (testigo de Ising, sin parte XY), el giro no lo
    cambia y la cota es `dual.ising_limit(n)`."""
    if not np.isfinite(Delta0):
        return dual.ising_limit(n)
    den = Delta0 + 2 * np.cos((Phi0 - Theta0) / n)
    return 4 * eps(n, Delta0, Phi0 + Theta1 - Theta0) / den


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
    return v1, kohn_bound_from_witness(eps, n, Theta0, Phi0, Delta0, Theta1)


def ladder_twist_y(y, L, Psi):
    """Testigo de la escalera periódica (y en el orden de LadderDual.KEYS)
    girado Psi: (alpha_b, beta_b) y (alpha_t, beta_t) rotan un ángulo Psi/L
    (la fase chi de cada arista de pata sube Psi/L, como la de los datos en
    abelian_phases con psi -> psi + Psi); c_b, c_t y los peldaños no cambian."""
    y = np.array(y, dtype=float)
    a = Psi / L
    c, s = np.cos(a), np.sin(a)
    for i in (0, 3):  # (alpha_b, beta_b) y (alpha_t, beta_t)
        al, be = y[i], y[i + 1]
        y[i], y[i + 1] = c * al - s * be, s * al + c * be
    return y


def _block_W(dual, red, y):
    return sum(c * red[key] for key, c in zip(dual.KEYS, y))


def _top_eigs(W, r, rng=None, ncv=None):
    """Los r autovalores mayores de W (ascendentes): eigvalsh si es densa o
    pequeña; si no, Lanczos (eigsh) con arranque aleatorio de rng (o el de
    ARPACK si rng es None) y ncv vectores."""
    d = W.shape[0]
    r = min(r, d)
    if not sp.issparse(W) or d <= max(4 * r, 64):
        w = np.linalg.eigvalsh(W.toarray() if sp.issparse(W) else W)
        return w[-r:]
    v0 = None
    if rng is not None:
        v0 = rng.standard_normal(d)
        if np.iscomplexobj(W.data):
            v0 = v0 + 1j * rng.standard_normal(d)
    w = eigsh(W, k=r, which="LA", v0=v0, ncv=None if ncv is None else min(ncv, d - 1),
              tol=0, maxiter=100 * d, return_eigenvectors=False)
    return np.sort(w)


def witness_gap(dual, y, tol=1e-8, r=4, detalle=False):
    """lambda_max(W(y)) menos el primer autovalor distinto (a más de tol) de
    todos los bloques de `dual` (los r mayores de cada uno; si en un bloque
    los r están en el multiplete superior, se piden más).

    Con detalle, devuelve (gap, multiplete), con multiplete la lista de
    ((k, q), multiplicidad) de los bloques que tienen autovalores a menos de
    tol de lambda_max. Con half=True los bloques (k, q) y (2L - k, -q) tienen
    el mismo espectro y solo están los k <= L: el multiplete los cuenta una
    vez (y los (L, q) y (L, -q) dos), así que la multiplicidad es la de los
    bloques calculados, no la de todo el espacio; el gap no cambia."""
    tops = []
    for red in dual.blocks:
        W = _block_W(dual, red, y)
        d = W.shape[0]
        rr = min(r, d)
        while True:
            w = _top_eigs(W, rr)
            if rr >= d or w[0] < w[-1] - tol:
                break
            rr = min(2 * rr, d)
        tops.append(w)
    lam = max(w[-1] for w in tops)
    below = [w[w < lam - tol] for w in tops]
    nxt = max((b[-1] for b in below if b.size), default=-np.inf)
    gap = float(lam - nxt)
    if not detalle:
        return gap
    mult = [(lab, int(np.sum(w >= lam - tol))) for lab, w in zip(dual.labels, tops)]
    return gap, [(lab, m) for lab, m in mult if m]


def lam_max_aleatorio(dual, y, seed=0, r=4, ncv=40):
    """lambda_max(W(y)) reevaluado con Lanczos arrancado en un vector
    aleatorio (semilla seed) y ncv vectores de Lanczos, pidiendo los r mayores
    de cada bloque disperso (los densos, con eigvalsh): contraste de las
    cotas superiores finales (Review Focus 1 del plan; un arranque fijo puede
    quedarse en un sector de simetría y subestimar lambda_max)."""
    rng = np.random.default_rng(seed)
    return float(max(_top_eigs(_block_W(dual, red, y), r, rng, ncv)[-1]
                     for red in dual.blocks))

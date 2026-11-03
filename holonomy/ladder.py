"""Escaleras de dos patas: geometría, holonomías abelianas y dual.

Sitios: pata inferior 0..L-1 y pata superior L..2L-1; el peldaño i une i con
L+i. Aristas orientadas: patas i -> i+1 (y L+i -> L+i+1), peldaños i -> L+i;
con `periodic`, también L-1 -> 0 y 2L-1 -> L.

Datos abelianos: T^e = -R_z(phi_e) en cada arista. La holonomía de la plaqueta
p (recorrida p -> p+1 -> L+p+1 -> L+p -> p) es R_z del flujo
f_p = phi(p->p+1) + phi(p+1->L+p+1) - phi(L+p->L+p+1) - phi(p->L+p).
Gauge simétrico (de Landau): peldaños con fase 0, pata inferior con f_p/2 y
pata superior con -f_p/2. En la escalera periódica hay un ciclo independiente
más, la pata: `psi` se suma (repartido, psi/L por arista) a las dos patas, de
modo que la holonomía de la pata inferior es sum f_p/2 + psi y la de la
superior -sum f_p/2 + psi.

Dual (escalera periódica). Se minimiza lambda_max(W)/N sobre testigos
W = sum_e (alpha_e A_e + beta_e B_e + c_e C_e), con N = sum_e (2 alpha_e cos phi_e
+ 2 beta_e sin phi_e + c_e) > 0. Promediar un testigo óptimo sobre las
simetrías de los datos da otro testigo óptimo: U(1) alrededor de z (solo
quedan A, B, C y Z), la traslación a lo largo de las patas (coeficientes
iguales en todas las aristas de una misma pata y en todos los peldaños, si los
flujos son uniformes) y la reflexión izquierda-derecha compuesta con R_x(pi),
que anula los términos Z y la componente B de los peldaños. Quedan ocho
coeficientes: (alpha, beta, c) en cada pata y (alpha, c) en los peldaños. El
cociente es cuasiconvexo (lambda_max convexa, N lineal). Tanto el dual como
el primal se calculan por bloques de número de giros k y momento q.
"""
import numpy as np

from . import periodic
from .periodic import PeriodicDual


def ladder_edges(L, periodic=False):
    """Aristas en el orden: pata inferior, pata superior, peldaños."""
    m = L if periodic else L - 1
    bottom = [(i, (i + 1) % L) for i in range(m)]
    top = [(L + i, L + (i + 1) % L) for i in range(m)]
    rungs = [(i, L + i) for i in range(L)]
    return bottom + top + rungs


def edge_kinds(L, periodic=False):
    m = L if periodic else L - 1
    return ["b"] * m + ["t"] * m + ["r"] * L


def abelian_phases(L, fluxes, periodic=False, psi=0.0):
    """Fases phi_e en el gauge simétrico, con flujo fluxes[p] en la plaqueta p
    (L-1 plaquetas si es abierta, L si es periódica) y, si es periódica, psi."""
    m = L if periodic else L - 1
    fluxes = np.broadcast_to(np.asarray(fluxes, dtype=float), (m,))
    shift = psi / L if periodic else 0.0
    return ([f / 2 + shift for f in fluxes] + [-f / 2 + shift for f in fluxes]
            + [0.0] * L)


def plaquette_fluxes(L, phis, periodic=False):
    """Flujos de plaqueta y, si es periódica, holonomías de las dos patas."""
    m = L if periodic else L - 1
    b, t, r = phis[:m], phis[m:2 * m], phis[2 * m:]
    f = [b[p] + r[(p + 1) % L] - t[p] - r[p] for p in range(m)]
    if periodic:
        return f, (sum(b), sum(t))
    return f, None


def translation_perm(L):
    """Traslación de un sitio a lo largo de las patas: i -> i+1 (mód L) en
    cada pata."""
    return [(i + 1) % L + (L if i >= L else 0) for i in range(2 * L)]


def momentum_isometries(L, basis):
    """Isometrías V_q (q = 0..L-1) del sector `basis` sobre los estados de
    momento 2 pi q / L de la escalera (periodic.momentum_isometries con la
    traslación a lo largo de las patas)."""
    return periodic.momentum_isometries(L, basis, translation_perm(L))


def symmetric_blocks(L, dense_max=None, half=False, keys=None):
    """Bloques (k giros, momento q) de la escalera periódica 2xL, con los
    operadores sumados por tipo de arista: ("A"|"B"|"C", "b"|"t"|"r") y
    ("Z", "b"|"t") (Z total de cada pata), como matrices densas o, en los
    bloques de dimensión mayor que dense_max, dispersas (CSR). Con half, solo
    los sectores k <= L; con keys, solo esas claves."""
    return periodic.symmetric_blocks(2 * L, ladder_edges(L, periodic=True),
                                     edge_kinds(L, periodic=True), translation_perm(L), L,
                                     _z_groups(L), dense_max, half, keys)


def _z_groups(L):
    return {"b": range(L), "t": range(L, 2 * L)}


def primal_symmetric(L, phis, solver="CLARABEL", return_blocks=False, **kw):
    """Primal de la escalera periódica con flujo uniforme, reducido por U(1) y
    por traslación (rho = suma directa de bloques (k, q)). Las restricciones de
    las aristas de cada tipo se suman: son equivalentes porque los datos son
    invariantes bajo traslación y el estado se puede promediar."""
    import cvxpy as cp

    kinds = edge_kinds(L, periodic=True)
    phase = {kd: phis[kinds.index(kd)] for kd in "btr"}
    v = cp.Variable()
    cons, lin, tr, Xs = [], {}, 0, []
    for k, q, red in symmetric_blocks(L):
        d = red[("C", "b")].shape[0]
        X = cp.Variable((d, d), hermitian=True)
        Xs.append(((k, q), X))
        cons.append(X >> 0)
        tr = tr + cp.real(cp.trace(X))
        for key, M in red.items():
            lin[key] = lin.get(key, 0) + cp.real(cp.sum(cp.multiply(M.T, X)))
    cons.append(tr == 1)
    for kd in "btr":
        cons += [lin[("A", kd)] == 2 * L * v * np.cos(phase[kd]),
                 lin[("B", kd)] == 2 * L * v * np.sin(phase[kd]),
                 lin[("C", kd)] == L * v]
    cons += [lin[("Z", "b")] == 0, lin[("Z", "t")] == 0]
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver=solver, **kw)
    if return_blocks:
        return float(v.value), prob.status, [(lab, np.asarray(X.value)) for lab, X in Xs]
    return float(v.value), prob.status


class LadderDual(PeriodicDual):
    """lambda_max del testigo de la escalera periódica, por bloques (k, q)
    (periodic.PeriodicDual con la traslación a lo largo de las patas).

    Con dense_max, los bloques de dimensión mayor se guardan dispersos y sus
    autovalores superiores se calculan por Lanczos (eigsh): es lo que permite
    L >= 8 (los bloques densos no caben en memoria). Solo lam_max, top_blocks,
    block_gaps y v_dual_cg admiten bloques dispersos.

    Con half, solo los bloques k <= L y las claves de KEYS: el volteo global
    prod X compuesto con la conjugación compleja deja invariantes A, B y C y
    lleva el bloque (k, q) al (2L - k, -q), así que los dos tienen el mismo
    espectro para todo y y los bloques k > L no cambian lambda_max ni el
    dual. Ahorra ~40 % de memoria (lo que permite L = 11); labels y
    top_blocks solo dan entonces los bloques k <= L."""

    KEYS = (("A", "b"), ("B", "b"), ("C", "b"), ("A", "t"), ("B", "t"), ("C", "t"),
            ("A", "r"), ("C", "r"))

    def __init__(self, L, dense_max=None, half=False):
        self.L = L
        super().__init__(2 * L, ladder_edges(L, periodic=True), edge_kinds(L, periodic=True),
                         translation_perm(L), L, _z_groups(L), self.KEYS, dense_max, half)

    def witness_terms(self, y):
        """y = (alpha_b, beta_b, c_b, alpha_t, beta_t, c_t, alpha_r, c_r)."""
        ab, bb, cb, at, bt, ct, ar, cr = y
        return {("A", "b"): ab, ("B", "b"): bb, ("C", "b"): cb,
                ("A", "t"): at, ("B", "t"): bt, ("C", "t"): ct,
                ("A", "r"): ar, ("B", "r"): 0.0, ("C", "r"): cr}

    def heisenberg_y0(self, phis):
        """Heisenberg en todas las aristas, con la fase de los datos en las
        patas (peldaños con alpha = c = 1)."""
        pb, pt = phis[0], phis[self.L]
        return np.array([np.cos(pb), np.sin(pb), 1.0, np.cos(pt), np.sin(pt), 1.0, 1.0, 1.0])


def witness_fluxes(y, L):
    """Flujo de plaqueta del testigo, chi_b - chi_t (los peldaños, sin
    componente B, no aportan), y la holonomía de la pata inferior, L chi_b.
    La fase de una arista es la de alpha + i beta."""
    chi_b = np.arctan2(y[1], y[0])
    chi_t = np.arctan2(y[4], y[3])
    return float(np.angle(np.exp(1j * (chi_b - chi_t)))), float(np.angle(np.exp(1j * L * chi_b)))

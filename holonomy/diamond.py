"""Cadenas de diamantes K4-e: geometría, flujos de triángulo y gauge de árbol.

Sitios: la celda m (m = 0..k-1) tiene la unión j = 3m, los centrales u = 3m+1
y w = 3m+2, y la unión siguiente j' = 3m+3 (que es 0 en la cadena cerrada
para m = k-1). Hay 3k+1 sitios en la cadena abierta y 3k en la cerrada.
Aristas orientadas de cada celda, en este orden: (j,u), (j,w), (u,w), (u,j'),
(w,j'), de tipos "ju", "jw", "uw", "uj", "wj". La cadena abierta con k = 1 es
el diamante simple; la cerrada exige k >= 2.

Cada celda tiene dos triángulos: A = (j,u,w), recorrido j -> u -> w -> j, y
B = (u,j',w), recorrido u -> j' -> w -> u. La arista central (u,w) es común.

Datos abelianos: T^e = -R_z(phi_e). Los flujos de triángulo son
fA = phi(j,u) + phi(u,w) - phi(j,w) y fB = phi(u,j') - phi(w,j') - phi(u,w).
Gauge simétrico: (j,u) = fA/2 + s, (j,w) = -fA/2 + s, (u,w) = 0,
(u,j') = fB/2 + s, (w,j') = -fB/2 + s, con s = psi/(2k) en la cadena cerrada
(psi es el ciclo independiente extra, la vuelta completa) y s = 0 en la
abierta. Las holonomías de las dos vueltas de la cadena cerrada, por los
centrales u y por los w, son psi +- sum(fA+fB)/2.

Datos no abelianos (solo cadena abierta): gauge de árbol. R = identidad en
(j,u), (j,w), (u,j'); R(u,w) = H_A; R(w,j') = (H_B H_A)^T. Así la holonomía
de A, R_ju R_uw R_wj, es H_A y la de B, R_uj' R_j'w R_wu, es H_B, ambas en el
marco de j_m, y todos los marcos de unión coinciden por el árbol j -> u -> j'.

Dual reducido de la cadena cerrada (DiamondDual): ver su docstring.
"""
import numpy as np

from .paulis import Rz, Ry
from . import periodic
from .periodic import PeriodicDual

KINDS = ("ju", "jw", "uw", "uj", "wj")


def n_sites(k, periodic=False):
    return 3 * k if periodic else 3 * k + 1


def chain_edges(k, periodic=False):
    """Aristas orientadas, cinco por celda, en el orden de KINDS."""
    if periodic and k < 2:
        raise ValueError("la cadena cerrada exige k >= 2")
    edges = []
    for m in range(k):
        j, u, w = 3 * m, 3 * m + 1, 3 * m + 2
        jp = 0 if (periodic and m == k - 1) else 3 * m + 3
        edges += [(j, u), (j, w), (u, w), (u, jp), (w, jp)]
    return edges


def edge_kinds(k, periodic=False):
    chain_edges(k, periodic)  # valida k
    return list(KINDS) * k


def abelian_phases(k, fA, fB, periodic=False, psi=0.0):
    """Fases phi_e en el gauge simétrico; fA, fB escalares o secuencias de
    longitud k."""
    if periodic and k < 2:
        raise ValueError("la cadena cerrada exige k >= 2")
    fA = np.broadcast_to(np.asarray(fA, dtype=float), (k,))
    fB = np.broadcast_to(np.asarray(fB, dtype=float), (k,))
    s = psi / (2 * k) if periodic else 0.0
    phis = []
    for m in range(k):
        phis += [fA[m] / 2 + s, -fA[m] / 2 + s, 0.0, fB[m] / 2 + s, -fB[m] / 2 + s]
    return [float(x) for x in phis]


def triangle_fluxes(k, phis, periodic=False):
    """Flujos (fA, fB) de los triángulos y, si es cerrada, las holonomías de
    las dos vueltas (por los u y por los w)."""
    fA = [phis[5 * m] + phis[5 * m + 2] - phis[5 * m + 1] for m in range(k)]
    fB = [phis[5 * m + 3] - phis[5 * m + 4] - phis[5 * m + 2] for m in range(k)]
    if periodic:
        vu = sum(phis[5 * m] + phis[5 * m + 3] for m in range(k))
        vw = sum(phis[5 * m + 1] + phis[5 * m + 4] for m in range(k))
        return fA, fB, (vu, vw)
    return fA, fB, None


def chain_rotations(k, pairs, periodic=False):
    """Matrices R_e en el orden de chain_edges, con pairs = [(H_A, H_B), ...]."""
    if periodic:
        raise NotImplementedError("las cadenas cerradas no abelianas no son objetivo")
    if len(pairs) != k:
        raise ValueError("hace falta un par (H_A, H_B) por celda")
    I = np.eye(3)
    Rs = []
    for HA, HB in pairs:
        HA, HB = np.asarray(HA), np.asarray(HB)
        Rs += [I, I, HA, I, (HB @ HA).T]
    return Rs


def relative_pairs(tA, tB, g, Q):
    """Los dos pares de la cadena de 2 con los mismos invariantes:
    H_A = R_z(tA), H_B = R_y(g) R_z(tB) R_y(g)^T, y el segundo par conjugado
    por Q en SO(3)."""
    HA = Rz(tA)
    HB = Ry(g) @ Rz(tB) @ Ry(g).T
    Q = np.asarray(Q)
    return [(HA, HB), (Q @ HA @ Q.T, Q @ HB @ Q.T)]


def translation_perm(k):
    """Traslación de una celda en la cadena cerrada: i -> i + 3 (mód 3k)."""
    return [(i + 3) % (3 * k) for i in range(3 * k)]


def _z_groups(k):
    return {"j": range(0, 3 * k, 3), "u": range(1, 3 * k, 3), "w": range(2, 3 * k, 3)}


def ring_primal(k, phis, solver="CLARABEL", **kw):
    """Primal de la cadena cerrada reducido por U(1) y traslación de una
    celda (periodic.primal_reduced): cota inferior de v_max, exacta con datos
    uniformes en el gauge simétrico. Devuelve (v, estado). Cabe con k <= 3;
    con k = 4 los bloques de dimensión 236 no caben en Clarabel."""
    return periodic.primal_reduced(n_sites(k, True), chain_edges(k, True), edge_kinds(k, True),
                                   translation_perm(k), k, _z_groups(k), phis, solver, **kw)


class DiamondDual(PeriodicDual):
    """lambda_max del testigo de la cadena cerrada de k diamantes con datos
    abelianos uniformes (los mismos fA, fB en todas las celdas), por bloques
    (k giros, q momento) de periodic.PeriodicDual con la traslación de una
    celda, i -> i + 3.

    Reducciones (un testigo óptimo promediado sobre una simetría de los datos
    es otro testigo óptimo, porque lambda_max es convexa y N lineal):
      - U(1) alrededor de z: solo A, B, C por arista y Z por sitio;
      - traslación de una celda: en el gauge simétrico de abelian_phases la
        fase de cada arista depende solo de su tipo (s = psi/(2k) es la misma
        en todas), así que los coeficientes se igualan por tipo: 15 claves
        (A|B|C, tipo) y Z sumada por grupo j, u, w;
      - inversión temporal Theta = (iY)^{(x)n} K (antiunitaria): manda
        X -> -X, Y -> -Y, Z -> -Z en cada sitio, luego deja A_e, B_e y C_e
        invariantes (productos de dos Paulis) y cambia el signo de Z_i; W y
        Theta W Theta^-1 tienen el mismo espectro y la misma N, así que el
        promedio anula los términos Z. Por eso KEYS no tiene claves Z (y half
        vale).
    La reflexión u <-> w no se usa: lleva (fA, fB, psi) a (-fA, -fB, psi), y
    compuesta con la conjugación compleja (fases -> -fases) a (fA, fB, -psi);
    solo es simetría de los datos con psi = 0 o pi, no en el mapa (f, psi)
    (donde es la comprobación psi -> -psi del guion diamond_ring).

    dense_max y half como en PeriodicDual (half: solo los bloques k <= n // 2,
    válido porque KEYS no tiene claves Z)."""

    KEYS = tuple((name, kd) for kd in KINDS for name in "ABC")

    def __init__(self, k, dense_max=None, half=False):
        self.k = k
        n = n_sites(k, periodic=True)
        super().__init__(n, chain_edges(k, periodic=True), edge_kinds(k, periodic=True),
                         translation_perm(k), k, _z_groups(k), self.KEYS, dense_max, half)

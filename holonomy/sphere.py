"""Poliedros esféricos con datos abelianos: curvatura como flujo por cara y
dual reducido por simetría.

Un qubit por vértice y R_e = R_z(phi_e) en cada arista orientada e = (i -> j),
con i < j. v_max es el mayor v con un estado global de marginales de arista
tr(rho A_e) = 2v cos(phi_e), tr(rho B_e) = 2v sin(phi_e), tr(rho C_e) = v y de un
qubit I/2, con A_e = -(X_iX_j + Y_iY_j), B_e = X_iY_j - Y_iX_j y C_e = -Z_iZ_j.
Dual: v_max = min lambda_max(W), W = sum_e(alpha_e A_e + beta_e B_e + c_e C_e),
con N = sum_e(2 alpha_e cos(phi_e) + 2 beta_e sin(phi_e) + c_e) = 1.

En el marco de los datos el término de hopping es
a_e (cos(phi_e) A_e + sin(phi_e) B_e) + b_e (-sin(phi_e) A_e + cos(phi_e) B_e),
y N solo ve a_e y c_e: N = sum_e(2 a_e + c_e). Promediando un testigo sobre el
grupo de isometrías, cada órbita de aristas conserva un único (a, c) y a lo sumo
un b; `edge_orbits` calcula las órbitas y si b se anula en cada una.

Convenio de `paulis`: el qubit 0 es el más significativo y |0> tiene Z = +1. El
sector de k giros son los estados de n bits con k unos, en orden creciente (el
mismo de `sdp.u1_sector_operators`).
"""
import itertools
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from scipy.spatial import ConvexHull

TOL_GEOM = 1e-9
PHI_AUREO = (1 + np.sqrt(5)) / 2
_GOLD = (np.sqrt(5) - 1) / 2

# ---------------------------------------------------------------------------
# Poliedros
# ---------------------------------------------------------------------------


def _signed_perms(p, cyclic=True):
    """Todas las permutaciones (cíclicas o no) de p con todos los signos."""
    perms = [p[i:] + p[:i] for i in range(3)] if cyclic else list(itertools.permutations(p))
    out = set()
    for q in perms:
        for s in itertools.product((1, -1), repeat=3):
            out.add(tuple(float(a * b) for a, b in zip(q, s)))
    return sorted(out)


def _coords(name):
    f = PHI_AUREO
    if name == "tetraedro":
        return [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    if name == "octaedro":
        return _signed_perms((1, 0, 0))
    if name == "cubo":
        return _signed_perms((1, 1, 1))
    if name == "icosaedro":
        return _signed_perms((0, 1, f))
    if name == "dodecaedro":
        return _signed_perms((1, 1, 1)) + _signed_perms((0, 1 / f, f))
    if name == "cuboctaedro":
        return _signed_perms((1, 1, 0), cyclic=False)
    if name == "rombododecaedro":
        return _signed_perms((1, 1, 1)) + _signed_perms((2, 0, 0))
    raise ValueError(f"poliedro desconocido: {name}")


# En orden de tamaño (número de vértices y, a igualdad, de aristas).
NAMES = ("tetraedro", "octaedro", "cubo", "cuboctaedro", "icosaedro",
         "rombododecaedro", "dodecaedro")


@dataclass(frozen=True)
class Polyhedron:
    """verts: V x 3 centrados; edges: pares (i, j), i < j; faces: ciclos de
    vértices antihorarios vistos desde fuera."""
    name: str
    verts: np.ndarray
    edges: tuple
    faces: tuple

    @property
    def n(self):
        return len(self.verts)

    @property
    def E(self):
        return len(self.edges)

    @property
    def F(self):
        return len(self.faces)

    @property
    def max_degree(self):
        deg = np.zeros(self.n, dtype=int)
        for i, j in self.edges:
            deg[i] += 1
            deg[j] += 1
        return int(deg.max())


def polyhedron(name):
    """Poliedro desde coordenadas: envolvente convexa, fusionando las facetas
    coplanares que Qhull triangula. Comprueba Euler V - E + F = 2."""
    V = np.array(_coords(name), dtype=float)
    V -= V.mean(axis=0)
    hull = ConvexHull(V)
    planes = []
    for eq in hull.equations:
        if not any(np.allclose(eq, p, atol=TOL_GEOM) for p in planes):
            planes.append(eq)
    faces, edges = [], set()
    for eq in planes:
        idx = np.flatnonzero(np.abs(V @ eq[:3] + eq[3]) < TOL_GEOM)
        c = V[idx].mean(axis=0)
        u = V[idx[0]] - c
        u /= np.linalg.norm(u)
        w = np.cross(eq[:3], u)  # (u, w, normal exterior) es una base directa
        order = np.argsort(np.arctan2((V[idx] - c) @ w, (V[idx] - c) @ u))
        cyc = idx[order].tolist()
        k = cyc.index(min(cyc))
        cyc = cyc[k:] + cyc[:k]
        faces.append(tuple(cyc))
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            edges.add((min(a, b), max(a, b)))
    poly = Polyhedron(name, V, tuple(sorted(edges)), tuple(sorted(faces)))
    if poly.n - poly.E + poly.F != 2:
        raise ValueError(f"{name}: Euler falla: {poly.n} - {poly.E} + {poly.F}")
    return poly


def incidence(poly):
    """Matriz cara x arista: +1 si el ciclo de la cara recorre e = (i, j) de i a j
    y -1 si la recorre al revés."""
    eidx = {e: m for m, e in enumerate(poly.edges)}
    D = np.zeros((poly.F, poly.E))
    for p, cyc in enumerate(poly.faces):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            if a < b:
                D[p, eidx[a, b]] += 1
            else:
                D[p, eidx[b, a]] -= 1
    return D


def face_fluxes(poly):
    """Ángulo sólido de cada cara visto desde el centro: exceso esférico del
    polígono geodésico (suma de ángulos interiores - (m - 2) pi)."""
    U = poly.verts / np.linalg.norm(poly.verts, axis=1, keepdims=True)
    out = np.empty(poly.F)
    for p, cyc in enumerate(poly.faces):
        m = len(cyc)
        total = 0.0
        for q in range(m):
            x = U[cyc[q]]
            ta = U[cyc[q - 1]] - (U[cyc[q - 1]] @ x) * x
            tb = U[cyc[(q + 1) % m]] - (U[cyc[(q + 1) % m]] @ x) * x
            total += np.arccos(np.clip(ta @ tb / np.linalg.norm(ta) / np.linalg.norm(tb), -1, 1))
        out[p] = total - (m - 2) * np.pi
    return out


# ---------------------------------------------------------------------------
# Fases
# ---------------------------------------------------------------------------


def phases(poly, fluxes, m=None, tol=1e-9):
    """Fases phi_e con holonomía de cada cara = flujo - 2 pi m_p (mínimos
    cuadrados). La suma de las fases por cara es 0 y la de flujos 4 pi, luego
    sum m_p = 2; por defecto, m = 2 en la cara 0. Verifica el residuo."""
    D = incidence(poly)
    if m is None:
        m = np.zeros(poly.F)
        m[0] = 2
    m = np.asarray(m, dtype=float)
    rhs = np.asarray(fluxes) - 2 * np.pi * m
    phis = np.linalg.lstsq(D, rhs, rcond=None)[0]
    res = np.max(np.abs(D @ phis - rhs))
    if res > tol:
        raise ValueError(f"{poly.name}: residuo de fases {res:.2e} (sum m = {m.sum()})")
    return phis


def face_holonomies(poly, phis):
    """Holonomía de cada cara: suma con signo de phi_e a lo largo de su ciclo."""
    return incidence(poly) @ np.asarray(phis)


def m_placements(F):
    """Colocaciones de sum m_p = 2: dos caras distintas con 1, o una cara con 2."""
    for p in range(F):
        m = np.zeros(F)
        m[p] = 2
        yield m
    for p, q in itertools.combinations(range(F), 2):
        m = np.zeros(F)
        m[p] = m[q] = 1
        yield m


def min_l1_phases(poly, fluxes):
    """Mínimo de sum_e |phi_e| sobre gauges y colocaciones de m (programa lineal
    por colocación). Devuelve dict con min, mean y max de |phi_e| (de la solución
    que da el LP, que puede no ser única), y phis, m. El máximo depende del
    vértice óptimo que devuelve HiGHS; la media, no."""
    D = incidence(poly)
    E = poly.E
    cost = np.r_[np.zeros(E), np.ones(E)]
    A_ub = np.block([[np.eye(E), -np.eye(E)], [-np.eye(E), -np.eye(E)]])
    A_eq = np.c_[D, np.zeros_like(D)]
    bounds = [(None, None)] * E + [(0, None)] * E
    best = None
    for m in m_placements(poly.F):
        r = linprog(cost, A_ub=A_ub, b_ub=np.zeros(2 * E), A_eq=A_eq,
                    b_eq=np.asarray(fluxes) - 2 * np.pi * m, bounds=bounds, method="highs")
        if r.status != 0:
            raise RuntimeError(f"{poly.name}: LP L1 no resuelto: {r.message}")
        if best is None or r.fun < best[0] - 1e-12:
            best = (r.fun, r.x[:E].copy(), m)
    fun, phis, m = best
    return dict(min=float(fun), mean=float(fun / E), max=float(np.max(np.abs(phis))),
                phis=phis, m=m)


# ---------------------------------------------------------------------------
# Índice cromático
# ---------------------------------------------------------------------------


def edge_coloring(poly, ncolors):
    """Coloración propia de aristas con ncolors por vuelta atrás, o None si no
    existe (búsqueda exhaustiva salvo permutación de colores)."""
    used = [0] * poly.n
    colors = [0] * poly.E
    edges = poly.edges

    def rec(e, top):
        if e == len(edges):
            return True
        i, j = edges[e]
        for c in range(min(top + 1, ncolors)):
            bit = 1 << c
            if used[i] & bit or used[j] & bit:
                continue
            used[i] |= bit
            used[j] |= bit
            colors[e] = c
            if rec(e + 1, max(top, c + 1)):
                return True
            used[i] ^= bit
            used[j] ^= bit
        return False

    return list(colors) if rec(0, 0) else None


def chromatic_index(poly):
    """Índice cromático exacto: Delta o Delta + 1 (Vizing)."""
    for c in (poly.max_degree, poly.max_degree + 1):
        if edge_coloring(poly, c) is not None:
            return c
    raise RuntimeError("Vizing falla: imposible")


# ---------------------------------------------------------------------------
# Simetría
# ---------------------------------------------------------------------------

Isometry = namedtuple("Isometry", "M perm det")


def symmetry_group(poly):
    """Matrices ortogonales (propias e impropias) que permutan los vértices.

    Cada isometría queda fijada por la imagen de tres vértices independientes;
    se prueban solo las imágenes que conservan los productos escalares."""
    V = poly.verts
    G = V @ V.T
    base = max(itertools.combinations(range(poly.n), 3),
               key=lambda c: abs(np.linalg.det(V[list(c)])))
    a, b, c = base
    inv = np.linalg.inv(V[list(base)])
    group = []
    for x, y, z in itertools.permutations(range(poly.n), 3):
        if not (abs(G[x, x] - G[a, a]) < TOL_GEOM and abs(G[y, y] - G[b, b]) < TOL_GEOM
                and abs(G[z, z] - G[c, c]) < TOL_GEOM and abs(G[x, y] - G[a, b]) < TOL_GEOM
                and abs(G[x, z] - G[a, c]) < TOL_GEOM and abs(G[y, z] - G[b, c]) < TOL_GEOM):
            continue
        M = (inv @ V[[x, y, z]]).T  # M v_a = v_x, etc.
        if np.max(np.abs(M.T @ M - np.eye(3))) > TOL_GEOM:
            continue
        img = V @ M.T
        dist = np.linalg.norm(img[:, None, :] - V[None, :, :], axis=2)
        perm = dist.argmin(axis=1)
        if dist[np.arange(poly.n), perm].max() < TOL_GEOM and len(set(perm.tolist())) == poly.n:
            group.append(Isometry(M, tuple(int(p) for p in perm), int(round(np.linalg.det(M)))))
    return group


def edge_image(g, edge):
    """Imagen (i', j') de la arista con i' < j', y si la isometría la invierte."""
    i, j = g.perm[edge[0]], g.perm[edge[1]]
    return ((i, j), False) if i < j else ((j, i), True)


def b_sign(g, edge):
    """Signo de b al mandar e a g(e): propia sin invertir +, propia que invierte
    -, impropia sin invertir -, impropia que invierte +."""
    _, flipped = edge_image(g, edge)
    return 1 if (g.det > 0) != flipped else -1


Orbit = namedtuple("Orbit", "edges stabilizer b_zero signs")


def edge_orbits(poly, group):
    """Órbitas de aristas. Para cada una: índices de arista, orden del
    estabilizador, si b se anula (algún elemento del estabilizador le cambia el
    signo) y, si no se anula, los signos s_e con b_e = s_e b en la órbita."""
    eidx = {e: m for m, e in enumerate(poly.edges)}
    seen = set()
    orbits = []
    for m0, e0 in enumerate(poly.edges):
        if m0 in seen:
            continue
        sign = {}
        stab = 0
        b_zero = False
        for g in group:
            img, _ = edge_image(g, e0)
            m = eidx[img]
            s = b_sign(g, e0)
            if m == m0:
                stab += 1
                b_zero |= s < 0
            if m in sign and sign[m] != s:
                b_zero = True
            sign.setdefault(m, s)
        members = sorted(sign)
        seen.update(members)
        signs = {m: (0 if b_zero else sign[m]) for m in members}
        orbits.append(Orbit(members, stab, b_zero, signs))
    return orbits


def act_on_witness(poly, g, a, b, c):
    """Testigo transformado por g: (a, b, c) en e pasa a g(e), con b según `b_sign`."""
    eidx = {e: m for m, e in enumerate(poly.edges)}
    a2, b2, c2 = np.empty(poly.E), np.empty(poly.E), np.empty(poly.E)
    for m, e in enumerate(poly.edges):
        img, _ = edge_image(g, e)
        q = eidx[img]
        a2[q], b2[q], c2[q] = a[m], b_sign(g, e) * b[m], c[m]
    return a2, b2, c2


# ---------------------------------------------------------------------------
# Bloques de número de giros
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def sector_basis(n, k):
    """Estados de n bits con k unos, en orden creciente."""
    pop = np.zeros(1, dtype=np.int8)
    for _ in range(n):
        pop = np.concatenate((pop, pop + 1))
    return np.flatnonzero(pop == k)


def hopping_block(n, k, edges, alpha, beta):
    """sum_e (alpha_e A_e + beta_e B_e) en el sector de k giros, disperso."""
    basis = sector_basis(n, k)
    dim = len(basis)
    R, C, D = [], [], []
    for (i, j), al, be in zip(edges, alpha, beta, strict=True):
        bi = (basis >> (n - 1 - i)) & 1
        bj = (basis >> (n - 1 - j)) & 1
        cols = np.flatnonzero(bi != bj)
        rows = np.searchsorted(basis, basis[cols] ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j)))
        # A: intercambio con elemento -2. B = XY - YX = 2i(s+_i s-_j - h.c.), y
        # s+_i s-_j actúa si i está en |1> y j en |0>.
        sgn = np.where(bi[cols] == 1, 1.0, -1.0)
        R.append(rows)
        C.append(cols)
        D.append(-2.0 * al + 2j * be * sgn)
    M = sp.csr_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                      shape=(dim, dim), dtype=complex)
    if M.nnz == 0 or np.all(M.data.imag == 0):
        M = M.real.tocsr()
    return M


def zz_diagonal(n, k, edges, c=None):
    """Diagonal de sum_e c_e C_e = -sum_e c_e Z_iZ_j en el sector de k giros."""
    basis = sector_basis(n, k)
    out = np.zeros(len(basis))
    c = np.ones(len(edges)) if c is None else c
    for (i, j), ce in zip(edges, c, strict=True):
        zi = 1 - 2 * ((basis >> (n - 1 - i)) & 1)
        zj = 1 - 2 * ((basis >> (n - 1 - j)) & 1)
        out -= ce * zi * zj
    return out


def witness_lmax(poly, phis, a, b, c, spins=None, dense_max=2000):
    """lambda_max de un testigo general (a_e, b_e, c_e), por bloques k <= n//2
    (o los de `spins`). Devuelve (lambda_max, espectros densos por k): solo para
    controles con dim <= dense_max."""
    n, E = poly.n, poly.edges
    cp, sp_ = np.cos(phis), np.sin(phis)
    alpha = a * cp - b * sp_
    beta = a * sp_ + b * cp
    spectra = {}
    for k in (range(n // 2 + 1) if spins is None else spins):
        H = hopping_block(n, k, E, alpha, beta).toarray()
        if len(H) > dense_max:
            raise ValueError(f"bloque k={k} de dimensión {len(H)} > dense_max")
        H = H + np.diag(zz_diagonal(n, k, E, c))
        spectra[k] = np.linalg.eigvalsh(H)
    return max(w[-1] for w in spectra.values()), spectra


# ---------------------------------------------------------------------------
# Dual reducido
# ---------------------------------------------------------------------------


def secant_lower_bound(points):
    """Cota inferior del mínimo de una función convexa con valores conocidos:
    en cada intervalo a un lado del mejor punto, el máximo de las dos secantes
    exteriores la acota por debajo. Devuelve -inf si faltan puntos a un lado."""
    xs = sorted(points)
    ys = [points[x] for x in xs]
    i = int(np.argmin(ys))
    if i < 2 or i > len(xs) - 3:
        return -np.inf
    lb = np.inf
    for l, r in ((i - 1, i), (i, i + 1)):
        mL = (ys[l] - ys[l - 1]) / (xs[l] - xs[l - 1])
        mR = (ys[r + 1] - ys[r]) / (xs[r + 1] - xs[r])

        def g(x):
            return max(ys[l] + mL * (x - xs[l]), ys[r] + mR * (x - xs[r]))

        cands = [xs[l], xs[r]]
        if mR > mL:
            xc = (ys[r] - mR * xs[r] - ys[l] + mL * xs[l]) / (mL - mR)
            cands.append(float(np.clip(xc, xs[l], xs[r])))
        lb = min(lb, min(g(x) for x in cands))
    return lb


def minimize_convex(f, x0, h, tol=1e-9, max_eval=200, min_width=1e-13):
    """Mínimo de una función convexa de una variable en toda la recta: acota el
    intervalo ampliando el paso y sigue con búsqueda dorada, hasta que la cota
    por secantes queda a menos de tol. Devuelve (x, f(x), cota inferior, puntos)."""
    pts = {}

    def F(x):
        x = float(x)
        if x not in pts:
            pts[x] = float(f(x))
        return pts[x]

    a, b = x0, x0 + h
    if F(b) > F(a):
        a, b = b, a
    c = b + (b - a) / _GOLD
    while F(c) < F(b):
        if len(pts) > max_eval:
            raise RuntimeError("minimize_convex: no se acota el intervalo")
        a, b, c = b, c, c + (c - b) / _GOLD
    lo, mid, hi = min(a, c), b, max(a, c)

    lower = -np.inf
    while len(pts) < max_eval and hi - lo > min_width * max(1.0, abs(mid)):
        lower = secant_lower_bound(pts)
        if F(mid) - lower < tol:
            break
        if hi - mid > mid - lo:
            y = mid + (1 - _GOLD) * (hi - mid)
            if F(y) <= F(mid):
                lo, mid = mid, y
            else:
                hi = y
        else:
            y = mid - (1 - _GOLD) * (mid - lo)
            if F(y) <= F(mid):
                hi, mid = mid, y
            else:
                lo = y
    lower = secant_lower_bound(pts)
    return mid, F(mid), lower, pts


def optimal_interval(f, x, val, h, tol=1e-8, width=1e-6):
    """Extremos del conjunto de mínimos de una función convexa alrededor de x, a
    tol de val y con precisión width por bisección. Una meseta (bloque cuyo mayor
    autovalor no depende de t) hace que el t óptimo no sea único."""
    edges = []
    for s in (-1, 1):
        inside, step = 0.0, h
        while f(x + s * (inside + step)) <= val + tol:
            inside += step
            step *= 2
        outside = inside + step
        while outside - inside > width:
            y = (inside + outside) / 2
            if f(x + s * y) <= val + tol:
                inside = y
            else:
                outside = y
        edges.append(x + s * inside)
    return tuple(edges)


def _eigsh_max(op, dim, v0, tol, k_eigs, ncv, maxiter):
    """Mayor autovalor con ARPACK. Sin convergencia, repite con más vectores."""
    ncv = max(ncv or 0, 2 * k_eigs + 1, 30)
    for _ in range(3):
        try:
            w, v = eigsh(op, k=k_eigs, which="LA", tol=tol, ncv=min(ncv, dim - 1), v0=v0,
                         maxiter=maxiter)
            q = int(np.argmax(w))
            return float(w[q]), v[:, q]
        except ArpackNoConvergence as err:
            if len(err.eigenvalues):
                v0 = err.eigenvectors[:, int(np.argmax(err.eigenvalues))]
            ncv *= 2
    raise RuntimeError(f"eigsh no converge con ncv = {ncv // 2}, dim = {dim}")


class ReducedDual:
    """W(t, b) = t H_a + b H_b + (1/E - 2t) C con una órbita de aristas: a = t,
    c = 1/E - 2t, b_e = s_e b (b = 0 si b_signs es None) y N = 1.

    `value` evita diagonalizar un bloque cuando la variación de lambda_max entre
    (t, b) y el punto anterior, acotada por Weyl con |dW/dt| <= 4E y |dW/db| <= 2E,
    no le deja alcanzar el máximo actual."""

    def __init__(self, poly, phis, b_signs=None, dense_max=2000, tol=1e-11, k_eigs=3, ncv=None,
                 maxiter=20000, seed=7):
        self.poly = poly
        self.E = poly.E
        self.n = poly.n
        self.dense_max = dense_max
        self.tol, self.k_eigs, self.ncv, self.maxiter = tol, k_eigs, ncv, maxiter
        self.rng = np.random.default_rng(seed)
        self.free_b = b_signs is not None
        phis = np.asarray(phis, dtype=float)
        cp, sn = np.cos(phis), np.sin(phis)
        self.blocks = []
        for k in range(self.n // 2 + 1):
            Ha = hopping_block(self.n, k, poly.edges, cp, sn)
            Hb = hopping_block(self.n, k, poly.edges, -b_signs * sn, b_signs * cp) if self.free_b else None
            self.blocks.append((Ha, Hb, zz_diagonal(self.n, k, poly.edges)))
        self.last = {}  # k -> (t, b, lambda, vector)
        self.n_eval = 0
        self.kmax = None
        self.psi = None
        self.planes = []  # planos de corte (lambda, g_t, g_b, t, b) si free_b

    def diag_coef(self, t):
        return 1.0 / self.E - 2.0 * t

    def block_lmax(self, k, t, b=0.0, k_eigs=None, ncv=None, tol=None, v0=None, warm=True):
        """Mayor autovalor y autovector de W(t, b) en el sector de k giros."""
        Ha, Hb, d = self.blocks[k]
        dim = Ha.shape[0]
        c = self.diag_coef(t)
        if dim <= self.dense_max:
            W = t * Ha.toarray()
            if self.free_b:
                W = W + b * Hb.toarray()
            W[np.arange(dim), np.arange(dim)] += c * d
            w, v = scipy.linalg.eigh(W, subset_by_index=[dim - 1, dim - 1])
            return float(w[0]), v[:, 0]
        cplx = np.iscomplexobj(Ha.data) or self.free_b

        def matvec(x):
            x = x.ravel()
            y = t * (Ha @ x) + c * (d * x)
            if self.free_b:
                y = y + b * (Hb @ x)
            return y

        op = LinearOperator((dim, dim), matvec=matvec, dtype=complex if cplx else float)
        if v0 is None and warm and k in self.last:
            v0 = self.last[k][3]
        if v0 is None:
            v0 = self.rng.standard_normal(dim)
            if cplx:
                v0 = v0 + 1j * self.rng.standard_normal(dim)
        return _eigsh_max(op, dim, v0, self.tol if tol is None else tol,
                          k_eigs or self.k_eigs, ncv or self.ncv, self.maxiter)

    def value(self, t, b=0.0, all_blocks=False):
        """lambda_max(W(t, b)). Guarda en self.kmax y self.psi el bloque máximo."""
        order = sorted(range(len(self.blocks)),
                       key=lambda k: -self.last[k][2] if k in self.last else np.inf)
        vmax, kmax = -np.inf, None
        for k in order:
            if not all_blocks and k in self.last:
                t0, b0, lam0, _ = self.last[k]
                if lam0 + 4 * self.E * abs(t - t0) + 2 * self.E * abs(b - b0) < vmax:
                    continue
            lam, psi = self.block_lmax(k, t, b)
            self.last[k] = (t, b, lam, psi)
            if lam > vmax:
                vmax, kmax = lam, k
        self.n_eval += 1
        self.kmax = kmax
        self.psi = self.last[kmax][3]
        if self.free_b:
            self.planes.append((vmax, *self.subgradient(kmax, self.psi), t, b))
        return vmax

    def subgradient(self, k, psi):
        """Hellmann-Feynman: <psi|dW/dt|psi>, <psi|dW/db|psi> (0 si no hay b).

        Con psi en el autoespacio máximo es un subgradiente de lambda_max, también
        si hay degeneración."""
        Ha, Hb, d = self.blocks[k]
        psic = psi.conj()
        gt = float(np.real(psic @ (Ha @ psi)) - 2 * np.real(psic @ (d * psi)))
        gb = float(np.real(psic @ (Hb @ psi))) if self.free_b else 0.0
        return gt, gb

    def block_values(self, t, b=0.0):
        """Mayor autovalor de cada bloque en (t, b), sin atajos."""
        self.value(t, b, all_blocks=True)
        return np.array([self.last[k][2] for k in range(len(self.blocks))])


@dataclass
class DualResult:
    v: float            # cota superior (testigo factible)
    lower: float        # cota inferior por convexidad (-inf si no hay)
    gap: float          # v - lower
    t: float
    b: float
    c: float
    delta: float        # Delta_xxz = c/a
    kmax: tuple         # bloques a menos de 1e-7 del máximo
    block_values: np.ndarray
    n_eval: int
    free_b: bool
    bound_kind: str     # cómo se obtuvo `lower`


def _plane_lower_bound(planes, tc, bc, R):
    """Mínimo del modelo de planos de corte en la caja |t - tc|, |b - bc| <= R.
    Devuelve (z, t, b, en el interior de la caja)."""
    P = np.array(planes)
    lam, gt, gb, t0, b0 = P.T
    A = np.c_[gt, gb, -np.ones(len(P))]
    rhs = gt * t0 + gb * b0 - lam
    r = linprog([0, 0, 1], A_ub=A, b_ub=rhs,
                bounds=[(tc - R, tc + R), (bc - R, bc + R), (None, None)], method="highs",
                options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
    if r.status != 0:
        raise RuntimeError(f"LP de cortes: {r.message}")
    inside = abs(r.x[0] - tc) < R * (1 - 1e-9) and abs(r.x[1] - bc) < R * (1 - 1e-9)
    return float(r.fun), float(r.x[0]), float(r.x[1]), inside


def minimize_tb(dual, tol_inner=1e-9, tol_outer=1e-8, max_cuts=150, target=1e-8):
    """Minimización anidada convexa en (t, b) y cota inferior por planos de corte
    (subgradientes de Hellmann-Feynman, método de Kelley). Devuelve
    (t, b, v, lower, descripción); lower = -inf si los cortes no acotan."""
    E = dual.E
    best = {"t": 1 / (3 * E)}

    def g(b):
        t, val, _, _ = minimize_convex(lambda x: dual.value(x, b), best["t"], 0.1 / E,
                                       tol=tol_inner)
        best["t"] = t
        return val

    b_opt, _, _, _ = minimize_convex(g, 0.0, 0.05 / E, tol=tol_outer)
    t_opt, v_opt, _, _ = minimize_convex(lambda x: dual.value(x, b_opt), best["t"], 0.02 / E,
                                         tol=tol_inner)
    R = 1.0 / E
    lower, kind = -np.inf, "sin cota"
    t, b, v = t_opt, b_opt, v_opt
    for _ in range(max_cuts):
        z, tc, bc, inside = _plane_lower_bound(dual.planes, t_opt, b_opt, R)
        if inside:
            lower = z
            kind = "planos de corte"
            if v - lower < target:
                break
        val = dual.value(tc, bc)
        if val < v:
            t, b, v = tc, bc, val
    return t, b, v, lower, kind


def reduced_dual(poly, phis, orbits, **kw):
    """Minimiza el dual reducido de una órbita de aristas. Si b se anula en la
    órbita, búsqueda dorada en t y cota por secantes; si b es libre, minimización
    anidada en (t, b) con cota por planos de corte."""
    if len(orbits) != 1:
        raise NotImplementedError(f"{poly.name}: {len(orbits)} órbitas; el dual reducido solo "
                                  "cubre una órbita")
    orb = orbits[0]
    E = poly.E
    if orb.b_zero:
        dual = ReducedDual(poly, phis, **kw)
        t, _, lower, _ = minimize_convex(lambda x: dual.value(x), 1 / (3 * E), 0.1 / E)
        b, kind = 0.0, "secantes"
    else:
        signs = np.array([orb.signs[m] for m in range(E)], dtype=float)
        dual = ReducedDual(poly, phis, b_signs=signs, **kw)
        t, b, _, lower, kind = minimize_tb(dual)
    # Evaluación final de todos los bloques en el óptimo, sin atajos de Weyl.
    vals = dual.block_values(t, b)
    v = float(vals.max())
    kmax = tuple(int(k) for k in np.flatnonzero(vals > v - 1e-7))
    c = dual.diag_coef(t)
    return DualResult(v, lower, v - lower, t, b, c, c / t if t != 0 else np.inf, kmax, vals,
                      dual.n_eval, dual.free_b, kind)


# ---------------------------------------------------------------------------
# Gauge SO(3) ambiente: rotaciones de arco de círculo máximo
# ---------------------------------------------------------------------------


def _rotation(axis, angle):
    """Rotación de ángulo `angle` alrededor del eje unitario `axis` (Rodrigues)."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def unit_vertices(poly):
    """Vértices proyectados a la esfera unidad."""
    return poly.verts / np.linalg.norm(poly.verts, axis=1, keepdims=True)


def ambient_rotations(poly):
    """G_ij y longitud geodésica l_e de cada arista i -> j de poly.edges.

    G_ij es la rotación del arco de círculo máximo que lleva x_j a x_i (ángulo l_e,
    eje x_j x x_i). Ese sentido es el único compatible con la forma pedida: la entrada
    (3, 3) de O_i^T G_ij O_j es x_i . G_ij x_j y R_z exige que valga 1.

    Signo: el producto G_{v0 v1} G_{v1 v2} ... G_{v_{m-1} v0} sobre un ciclo antihorario
    (visto desde fuera) actúa primero con el último factor, de modo que transporta x_0
    por el ciclo al revés: es la rotación de ángulo -f_p alrededor de x_0, o sea +f_p
    alrededor de -x_0. Con los marcos O_i = (e1, e2, x_i) las fases dan por tanto flujo
    -f_p. Se corrige con R_x(pi): los marcos (e1, -e2, -x_i) dan R_z(-phi) y flujo +f_p.
    Es un gauge global, no cambia v_max (`ambient_phases(poly, flip=True)`)."""
    x = unit_vertices(poly)
    Gs, lengths = [], np.empty(poly.E)
    for m, (i, j) in enumerate(poly.edges):
        axis = np.cross(x[j], x[i])
        lengths[m] = np.arctan2(np.linalg.norm(axis), x[i] @ x[j])
        Gs.append(_rotation(axis / np.linalg.norm(axis), lengths[m]))
    return Gs, lengths


def ambient_frames(poly):
    """Marcos O_i = (e1, e2, x_i), con e1 x e2 = x_i y e1, e2 arbitrarios."""
    frames = []
    for x in unit_vertices(poly):
        ref = np.eye(3)[int(np.argmin(np.abs(x)))]
        e1 = np.cross(ref, x)
        e1 /= np.linalg.norm(e1)
        frames.append(np.column_stack((e1, np.cross(x, e1), x)))
    return frames


def ambient_phases(poly, flip=False):
    """Fases phi_ij con O_i^T G_ij O_j = R_z(phi_ij). Sin flip dan flujo -f_p en cada
    cara antihoraria; con flip (marcos girados por R_x(pi)), +f_p."""
    Gs, _ = ambient_rotations(poly)
    O = ambient_frames(poly)
    out = np.empty(poly.E)
    for m, (i, j) in enumerate(poly.edges):
        R = O[i].T @ Gs[m] @ O[j]
        if max(abs(R[2, 2] - 1), np.max(np.abs(R[:2, 2])), np.max(np.abs(R[2, :2]))) > 1e-10:
            raise ValueError(f"{poly.name}: O_i^T G_ij O_j no es R_z en la arista {(i, j)}")
        out[m] = np.arctan2(R[1, 0], R[0, 0])
    return -out if flip else out


# ---------------------------------------------------------------------------
# Índice cromático fraccionario
# ---------------------------------------------------------------------------


def all_matchings(graph):
    """Todos los emparejamientos (también el vacío), como listas de índices de arista."""
    edges = graph.edges
    out = []

    def rec(e, used, cur):
        if e == len(edges):
            out.append(list(cur))
            return
        rec(e + 1, used, cur)
        i, j = edges[e]
        bit = (1 << i) | (1 << j)
        if not used & bit:
            cur.append(e)
            rec(e + 1, used | bit, cur)
            cur.pop()

    rec(0, 0, [])
    return out


def fractional_chromatic_index(graph):
    """chi'_f = 1/c, con c el máximo tal que una combinación convexa de emparejamientos
    cubre cada arista exactamente c (programa lineal sobre todos los emparejamientos;
    sirve cualquier objeto con n y edges)."""
    M = all_matchings(graph)
    rows, cols = [], []
    for q, match in enumerate(M):
        rows.extend(match)
        cols.extend([q] * len(match))
    E, K = len(graph.edges), len(M)
    A = sp.hstack([sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(E, K)),
                   sp.csr_matrix(-np.ones((E, 1)))])
    A = sp.vstack([A, sp.hstack([sp.csr_matrix(np.ones((1, K))), sp.csr_matrix((1, 1))])]).tocsr()
    b = np.r_[np.zeros(E), 1.0]
    cost = np.r_[np.zeros(K), -1.0]
    r = linprog(cost, A_eq=A, b_eq=b, bounds=[(0, None)] * K + [(0, 1)], method="highs")
    if r.status != 0:
        raise RuntimeError(f"LP de chi'_f: {r.message}")
    return 1.0 / float(-r.fun)


# ---------------------------------------------------------------------------
# Subgrafos inducidos y testigo por órbitas
# ---------------------------------------------------------------------------


def induced_subgraph(poly, phis, vertices):
    """Subgrafo inducido con los datos restringidos. La nueva numeración conserva el
    orden, así que conserva la orientación i < j y las fases. Devuelve
    (subgrafo, fases, índices de arista en poly)."""
    keep = sorted(int(v) for v in vertices)
    new = {v: q for q, v in enumerate(keep)}
    ids = [m for m, (i, j) in enumerate(poly.edges) if i in new and j in new]
    edges = tuple((new[poly.edges[m][0]], new[poly.edges[m][1]]) for m in ids)
    sub = Polyhedron(poly.name, poly.verts[keep], edges, ())
    return sub, np.asarray(phis, dtype=float)[ids], ids


def subset_group(poly, group, vertices):
    """Isometrías que conservan el subconjunto, con la permutación de su numeración."""
    keep = sorted(int(v) for v in vertices)
    new = {v: q for q, v in enumerate(keep)}
    out = []
    for g in group:
        if {g.perm[v] for v in keep} == set(keep):
            out.append(Isometry(g.M, tuple(new[g.perm[v]] for v in keep), g.det))
    return out


def identity_group(graph):
    """Grupo trivial: cada arista es su órbita, con b libre (testigo no simétrico)."""
    return [Isometry(np.eye(3), tuple(range(graph.n)), 1)]


@dataclass
class OrbitResult:
    v: float                 # cota superior: lambda_max(y) con N(y) = 1
    lower: float             # cota inferior del LP de cortes
    lower_dual: float        # la misma, desde los multiplicadores del LP (ver `_lp`)
    gap: float               # v - max(lower, lower_dual)
    y: np.ndarray
    params: list
    block_values: np.ndarray
    n_iter: int
    n_cuts: int
    box_active: bool


class OrbitDual:
    """W(y) = sum_O (a_O H_a^O + b_O H_b^O + c_O C^O), un (a, c) por órbita de aristas y
    b solo donde no se anula, con b_e = s_e b_O. N(y) = sum_O |O| (2 a_O + c_O).

    Planos de corte: para cualquier estado unitario psi, lambda_max(W(y)) >= <psi|W(y)|psi>,
    lineal en y, sea o no psi un autovector exacto. El LP de esos cortes con N(y) = 1 es
    una cota inferior; lambda_max(y) es la superior. Se usan k_eigs vectores por bloque y
    puntos alternos (mitad del camino al mejor punto, y el punto del LP) para estabilizar
    Kelley. `lower_dual` rehace la cota con los multiplicadores mu del LP: si
    sum_i mu_i g_i = tau N + r, entonces lambda_max(y) >= tau - |r|_1 R en la caja; es
    el valor de una mezcla explícita de estados."""

    def __init__(self, graph, phis, orbits, dense_max=300, tol=1e-11, k_eigs=4, ncv=None,
                 near=0.02, maxiter=20000, seed=7):
        self.graph = graph
        self.n, self.E = graph.n, len(graph.edges)
        self.orbits = orbits
        self.dense_max, self.tol, self.k_eigs, self.ncv = dense_max, tol, k_eigs, ncv
        self.near, self.maxiter = near, maxiter
        self.rng = np.random.default_rng(seed)
        phis = np.asarray(phis, dtype=float)
        cp_, sn = np.cos(phis), np.sin(phis)
        self.params, self.norm, self.opnorm = [], [], []
        specs = []  # (tipo, aristas, alpha, beta)
        for q, orb in enumerate(orbits):
            m = np.array(orb.edges)
            sub = [graph.edges[e] for e in m]
            self.params.append(("a", q))
            self.norm.append(2.0 * len(m))
            self.opnorm.append(2.0 * len(m))
            specs.append(("h", sub, cp_[m], sn[m]))
            if not orb.b_zero:
                s = np.array([orb.signs[e] for e in m], dtype=float)
                self.params.append(("b", q))
                self.norm.append(0.0)
                self.opnorm.append(2.0 * len(m))
                specs.append(("h", sub, -s * sn[m], s * cp_[m]))
            self.params.append(("c", q))
            self.norm.append(float(len(m)))
            self.opnorm.append(float(len(m)))
            specs.append(("d", sub, None, None))
        self.norm = np.array(self.norm)
        self.opnorm = np.array(self.opnorm)
        self.blocks = []
        for k in range(self.n // 2 + 1):
            ops = [hopping_block(self.n, k, sub, al, be) if tipo == "h"
                   else zz_diagonal(self.n, k, sub) for tipo, sub, al, be in specs]
            self.blocks.append(ops)
        self.last = {}
        self.cuts = []

    def heisenberg(self):
        """a = c = 1/(3E), b = 0: N = 1."""
        return np.array([0.0 if p == "b" else 1 / (3 * self.E) for p, _ in self.params])

    def edge_coefficients(self, y):
        """(a_e, b_e, c_e) por arista."""
        a, b, c = np.zeros(self.E), np.zeros(self.E), np.zeros(self.E)
        for val, (p, q) in zip(y, self.params, strict=True):
            orb = self.orbits[q]
            for e in orb.edges:
                if p == "a":
                    a[e] = val
                elif p == "c":
                    c[e] = val
                else:
                    b[e] = val * orb.signs[e]
        return a, b, c

    def _block(self, k, y):
        """W(y) en el sector k: (parte dispersa, diagonal)."""
        ops = self.blocks[k]
        dim = len(sector_basis(self.n, k))
        H = sp.csr_matrix((dim, dim), dtype=complex)
        d = np.zeros(dim)
        for val, op in zip(y, ops, strict=True):
            if val == 0:
                continue
            if isinstance(op, np.ndarray):
                d = d + val * op
            else:
                H = H + val * op
        return H, d

    def _top(self, k, y):
        """Mayores autovalores y autovectores del bloque k (el mayor, el último)."""
        H, d = self._block(k, y)
        dim = H.shape[0]
        if dim <= self.dense_max:
            W = H.toarray()
            W[np.arange(dim), np.arange(dim)] += d
            w, V = np.linalg.eigh(W)
            m = min(self.k_eigs, dim)
            return w[-m:], V[:, -m:]
        op = LinearOperator((dim, dim), matvec=lambda x: H @ x.ravel() + d * x.ravel(),
                            dtype=complex)
        v0 = self.last[k][2][:, -1] if k in self.last else None
        if v0 is None:
            v0 = self.rng.standard_normal(dim) + 1j * self.rng.standard_normal(dim)
        ncv = max(self.ncv or 0, 2 * self.k_eigs + 1, 30)
        for _ in range(3):
            try:
                w, V = eigsh(op, k=self.k_eigs, which="LA", tol=self.tol, ncv=min(ncv, dim - 1),
                             v0=v0, maxiter=self.maxiter)
                order = np.argsort(w)
                return w[order], V[:, order]
            except ArpackNoConvergence as err:
                if len(err.eigenvalues):
                    v0 = err.eigenvectors[:, int(np.argmax(err.eigenvalues))]
                ncv *= 2
        raise RuntimeError(f"eigsh no converge: dim = {dim}, ncv = {ncv // 2}")

    def _cut(self, k, psi):
        """g con <psi|W(y)|psi> = g . y (psi normalizado)."""
        psi = psi / np.linalg.norm(psi)
        return np.array([np.real(psi.conj() @ (op * psi)) if isinstance(op, np.ndarray)
                         else np.real(psi.conj() @ (op @ psi)) for op in self.blocks[k]])

    def value(self, y, all_blocks=False, add_cuts=True):
        """lambda_max(W(y)) y su bloque. Salta un bloque si la cota de Weyl con la
        evaluación anterior no le deja llegar al máximo actual."""
        order = sorted(range(len(self.blocks)),
                       key=lambda k: -self.last[k][1][-1] if k in self.last else np.inf)
        vmax, kmax = -np.inf, None
        for k in order:
            if not all_blocks and k in self.last:
                if self.last[k][1][-1] + np.abs(y - self.last[k][0]) @ self.opnorm < vmax:
                    continue
            w, V = self._top(k, y)
            self.last[k] = (y.copy(), w, V)
            if w[-1] > vmax:
                vmax, kmax = float(w[-1]), k
        if add_cuts:
            for k, (y0, w, V) in self.last.items():
                if np.array_equal(y0, y):
                    for q in np.flatnonzero(w > vmax - self.near * max(abs(vmax), 1e-3)):
                        self.cuts.append(self._cut(k, V[:, q]))
        return vmax, kmax

    def _lp(self, box):
        """Mínimo del modelo con x = E y en |x_q| <= box. Devuelve (z, cota por
        multiplicadores, y, caja activa)."""
        g = np.array(self.cuts) / self.E
        nvar = g.shape[1]
        n = self.norm / self.E
        A = np.c_[g, -np.ones(len(g))]
        r = linprog(np.r_[np.zeros(nvar), 1.0], A_ub=A, b_ub=np.zeros(len(g)),
                    A_eq=np.r_[n, 0.0][None, :], b_eq=[1.0],
                    bounds=[(-box, box)] * nvar + [(None, None)], method="highs",
                    options=dict(primal_feasibility_tolerance=1e-10,
                                 dual_feasibility_tolerance=1e-10))
        if r.status != 0:
            raise RuntimeError(f"LP de cortes: {r.message}")
        x = r.x[:nvar]
        mu = np.maximum(-r.ineqlin.marginals, 0.0)
        mu = mu / mu.sum()
        gbar = mu @ g
        tau = gbar @ n / (n @ n)
        resid = gbar - tau * n
        lower_dual = tau - np.abs(resid).sum() * box
        active = bool(np.max(np.abs(x)) > box * (1 - 1e-9))
        return float(r.fun), float(lower_dual), x / self.E, active

    def minimize(self, y0=None, target=5e-9, max_iter=3000, box=10.0, log=None):
        """Planos de corte hasta que lambda_max(y) - LP < target."""
        y = self.heisenberg() if y0 is None else np.asarray(y0, dtype=float)
        best, best_y = np.inf, None
        lower, lower_dual, active = -np.inf, -np.inf, False
        it = 0
        for it in range(1, max_iter + 1):
            y = y / (self.norm @ y)
            val, _ = self.value(y)
            if val < best:
                best, best_y = val, y.copy()
            z, zd, y_lp, active = self._lp(box)
            lower, lower_dual = max(lower, z), max(lower_dual, zd)
            if log and it % log == 0:
                print(f"      iter {it}: sup {best:.10f}, inf {lower:.10f}, "
                      f"dif {best - lower:.2e}, {len(self.cuts)} cortes", flush=True)
            if best - max(lower, lower_dual) < target and not active:
                break
            y = y_lp if it % 2 == 0 else best_y + 0.5 * (y_lp - best_y)
        val, _ = self.value(best_y, all_blocks=True, add_cuts=False)
        vals = np.array([self.last[k][1][-1] for k in range(len(self.blocks))])
        v = float(vals.max())
        return OrbitResult(v, lower, lower_dual, v - max(lower, lower_dual), best_y,
                           list(self.params), vals, it, len(self.cuts), active)


def witness_block_top(graph, phis, a, b, c, k, tol=1e-10, k_eigs=2, ncv=40, v0=None, seed=11):
    """Mayor autovalor en el sector k del testigo general (a_e, b_e, c_e), con eigsh si
    la dimensión pasa de 300. Un solo bloque da una cota inferior de lambda_max."""
    phis = np.asarray(phis, dtype=float)
    cp_, sn = np.cos(phis), np.sin(phis)
    H = hopping_block(graph.n, k, graph.edges, a * cp_ - b * sn, a * sn + b * cp_)
    d = zz_diagonal(graph.n, k, graph.edges, c)
    dim = H.shape[0]
    if dim <= 300:
        W = H.toarray().astype(complex)
        W[np.arange(dim), np.arange(dim)] += d
        return float(np.linalg.eigvalsh(W)[-1])
    op = LinearOperator((dim, dim), matvec=lambda x: H @ x.ravel() + d * x.ravel(), dtype=complex)
    if v0 is None:
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return _eigsh_max(op, dim, v0, tol, k_eigs, ncv, 20000)[0]

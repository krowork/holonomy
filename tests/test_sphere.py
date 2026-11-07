"""Pruebas rápidas de holonomy.sphere: geometría, fases, simetría, bloques y dual
reducido frente al primal en tetraedro, octaedro y cubo."""
import csv
import itertools
import pathlib

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp

from holonomy import sdp, sphere
from holonomy.paulis import X, Y, Z, op, rotation_angle

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
POLYS = {nm: sphere.polyhedron(nm) for nm in sphere.NAMES}
VEF = dict(tetraedro=(4, 6, 4), octaedro=(6, 12, 8), cubo=(8, 12, 6), icosaedro=(12, 30, 20),
           dodecaedro=(20, 30, 12), cuboctaedro=(12, 24, 14), rombododecaedro=(14, 24, 12))
PLATONICOS = ("tetraedro", "octaedro", "cubo", "icosaedro", "dodecaedro")
ORDEN_GRUPO = dict(tetraedro=24, octaedro=48, cubo=48, icosaedro=120, dodecaedro=120,
                   cuboctaedro=48, rombododecaedro=48)
# König da 4 en el rombododecaedro, que es bipartito con grado máximo 4.
CHI = dict(tetraedro=3, octaedro=4, cubo=3, icosaedro=5, dodecaedro=3, cuboctaedro=4,
           rombododecaedro=4)


def dense_ops(n, i, j):
    """A, B, C de la arista (i, j) como matrices densas de paulis.op."""
    A = -(op(n, {i: X, j: X}) + op(n, {i: Y, j: Y}))
    B = op(n, {i: X, j: Y}) - op(n, {i: Y, j: X})
    C = -op(n, {i: Z, j: Z})
    return A, B, C


def curved(poly):
    return sphere.phases(poly, sphere.face_fluxes(poly))


@pytest.mark.parametrize("nm", sphere.NAMES)
def test_euler_y_flujos(nm):
    p = POLYS[nm]
    assert (p.n, p.E, p.F) == VEF[nm]
    assert p.n - p.E + p.F == 2
    # Cada arista aparece una vez en cada sentido: ciclos orientados de verdad.
    assert np.all(np.abs(sphere.incidence(p)).sum(axis=0) == 2)
    assert np.all(sphere.incidence(p).sum(axis=0) == 0)
    f = sphere.face_fluxes(p)
    assert f.sum() == pytest.approx(4 * np.pi, abs=1e-10)
    assert np.all(f > 0)


@pytest.mark.parametrize("nm", PLATONICOS)
def test_flujo_platonico(nm):
    p = POLYS[nm]
    assert sphere.face_fluxes(p) == pytest.approx(4 * np.pi / p.F, abs=1e-10)


def test_flujos_con_abanico_de_triangulos():
    """El exceso esférico coincide con van Oosterom-Strackee sobre un abanico."""
    p = POLYS["cuboctaedro"]
    U = p.verts / np.linalg.norm(p.verts, axis=1, keepdims=True)
    for cyc, f in zip(p.faces, sphere.face_fluxes(p), strict=True):
        total = 0.0
        for q in range(1, len(cyc) - 1):
            a, b, c = U[cyc[0]], U[cyc[q]], U[cyc[q + 1]]
            total += 2 * np.arctan2(a @ np.cross(b, c), 1 + a @ b + b @ c + c @ a)
        assert total == pytest.approx(f, abs=1e-10)


@pytest.mark.parametrize("nm", sphere.NAMES)
def test_fases_holonomia_de_cara(nm):
    p = POLYS[nm]
    f = sphere.face_fluxes(p)
    F = p.F
    for m in (None, np.eye(F)[0] + np.eye(F)[F - 1]):
        h = sphere.face_holonomies(p, sphere.phases(p, f, m=m))
        d = np.mod(h - f + np.pi, 2 * np.pi) - np.pi
        assert np.max(np.abs(d)) < 1e-9


def test_suma_m_distinta_de_dos_falla():
    p = POLYS["cubo"]
    with pytest.raises(ValueError):
        sphere.phases(p, sphere.face_fluxes(p), m=np.eye(p.F)[0])


@pytest.mark.parametrize("nm", ["tetraedro", "octaedro", "cubo"])
def test_colocacion_de_m_no_cambia_el_dato(nm):
    """Otra colocación de m da los mismos datos salvo gauge: mismo espectro."""
    p = POLYS[nm]
    f = sphere.face_fluxes(p)
    a = np.full(p.E, 1 / (3 * p.E))
    zero = np.zeros(p.E)
    s1 = sphere.witness_lmax(p, sphere.phases(p, f), a, zero, a)[1]
    s2 = sphere.witness_lmax(p, sphere.phases(p, f, m=np.eye(p.F)[0] + np.eye(p.F)[1]), a, zero, a)[1]
    for k in s1:
        assert s1[k] == pytest.approx(s2[k], abs=1e-9)


@pytest.mark.parametrize("nm", sphere.NAMES)
def test_min_l1_frente_a_milp(nm):
    """Las colocaciones (1, 1) o (2) bastan: enteros m_p libres no mejoran."""
    p = POLYS[nm]
    f = sphere.face_fluxes(p)
    res = sphere.min_l1_phases(p, f)
    D, E, F = sphere.incidence(p), p.E, p.F
    assert np.sum(np.abs(res["phis"])) == pytest.approx(res["min"], abs=1e-9)
    assert res["mean"] == pytest.approx(res["min"] / E, abs=1e-12)
    h = sphere.face_holonomies(p, res["phis"])
    assert np.max(np.abs(np.mod(h - f + np.pi, 2 * np.pi) - np.pi)) < 1e-8
    # Variables: phi_e, |phi_e|, m_p, con m_p entero acotado.
    n_var = 2 * E + F
    cost = np.r_[np.zeros(E), np.ones(E), np.zeros(F)]
    eye = np.eye(E)
    zeros = np.zeros((E, F))
    cons = [
        LinearConstraint(np.block([[eye, -eye, zeros], [-eye, -eye, zeros]]), -np.inf, 0),
        LinearConstraint(np.c_[D, np.zeros((F, E)), 2 * np.pi * np.eye(F)], f, f),
        LinearConstraint(np.r_[np.zeros(2 * E), np.ones(F)][None, :], 2, 2),
    ]
    lo = np.r_[np.full(E, -np.inf), np.zeros(E), np.full(F, -4)]
    hi = np.r_[np.full(2 * E, np.inf), np.full(F, 6)]
    r = milp(cost, constraints=cons, bounds=Bounds(lo, hi),
             integrality=np.r_[np.zeros(2 * E), np.ones(F)], options=dict(time_limit=20))
    assert len(r.x) == n_var
    assert r.fun == pytest.approx(res["min"], abs=1e-6)


@pytest.mark.parametrize("nm", sphere.NAMES)
def test_indice_cromatico(nm):
    p = POLYS[nm]
    chi = sphere.chromatic_index(p)
    assert chi == CHI[nm]
    col = sphere.edge_coloring(p, chi)
    for v in range(p.n):
        vistos = [col[m] for m, e in enumerate(p.edges) if v in e]
        assert len(vistos) == len(set(vistos))
    # Con un color menos que el grado máximo no hay coloración propia.
    assert sphere.edge_coloring(p, p.max_degree - 1) is None


@pytest.mark.parametrize("nm", sphere.NAMES)
def test_grupo_y_orbitas(nm):
    p = POLYS[nm]
    G = sphere.symmetry_group(p)
    assert len(G) == ORDEN_GRUPO[nm]
    assert sum(g.det > 0 for g in G) == ORDEN_GRUPO[nm] // 2
    orbs = sphere.edge_orbits(p, G)
    assert len(orbs) == 1
    assert len(orbs[0].edges) * orbs[0].stabilizer == len(G)
    assert orbs[0].b_zero == (nm != "cuboctaedro")
    # Las isometrías conservan los flujos.
    f = sphere.face_fluxes(p)
    faces = {frozenset(c): q for q, c in enumerate(p.faces)}
    for g in G:
        for q, c in enumerate(p.faces):
            assert f[faces[frozenset(g.perm[v] for v in c)]] == pytest.approx(f[q], abs=1e-10)


@pytest.mark.parametrize("nm", ["tetraedro", "octaedro", "cubo"])
def test_bloques_contra_matrices_densas(nm):
    """Convenio de signos de A, B y C contra paulis.op, en todos los bloques."""
    p = POLYS[nm]
    n = p.n
    rng = np.random.default_rng(1)
    al, be, ce = rng.normal(size=(3, p.E))
    Wfull = np.zeros((2**n, 2**n), dtype=complex)
    for (i, j), a, b, c in zip(p.edges, al, be, ce, strict=True):
        A, B, C = dense_ops(n, i, j)
        Wfull += a * A + b * B + c * C
    for k in range(n + 1):
        idx = sphere.sector_basis(n, k)
        block = sphere.hopping_block(n, k, p.edges, al, be).toarray() + np.diag(
            sphere.zz_diagonal(n, k, p.edges, ce))
        assert np.max(np.abs(block - Wfull[np.ix_(idx, idx)])) < 1e-12
    # Fuera de los sectores de k giros, W no tiene elementos.
    pop = np.array([bin(s).count("1") for s in range(2**n)])
    mask = pop[:, None] != pop[None, :]
    assert np.max(np.abs(Wfull[mask])) == 0


def test_bloques_igual_que_sdp():
    """Mismo convenio que los operadores de sector de holonomy.sdp."""
    p = POLYS["octaedro"]
    for k in range(p.n + 1):
        basis, ops = sdp.u1_sector_operators(p.n, list(p.edges), k)
        assert list(sphere.sector_basis(p.n, k)) == basis
        for e in range(p.E):
            al, be, ce = np.eye(p.E)[e], np.eye(p.E)[e], np.eye(p.E)[e]
            H = sphere.hopping_block(p.n, k, p.edges, al, 0 * be)
            assert abs(H - ops["A", e]).max() < 1e-13
            H = sphere.hopping_block(p.n, k, p.edges, 0 * al, be)
            assert abs(H - ops["B", e]).max() < 1e-13
            assert np.max(np.abs(sphere.zz_diagonal(p.n, k, p.edges, ce)
                                 - ops["C", e].diagonal())) < 1e-13


@pytest.mark.parametrize("nm", ["tetraedro", "cubo"])
def test_k_y_n_menos_k_mismo_espectro(nm):
    p = POLYS[nm]
    rng = np.random.default_rng(2)
    a, b, c = rng.normal(size=(3, p.E))
    _, s = sphere.witness_lmax(p, curved(p), a, b, c, spins=range(p.n + 1))
    for k in range(p.n // 2 + 1):
        assert s[k] == pytest.approx(s[p.n - k], abs=1e-9)


@pytest.mark.parametrize("nm", ["tetraedro", "octaedro", "cubo"])
def test_regla_de_accion_sobre_el_testigo(nm):
    """g con gauge (y R_x(pi) si es impropia) conserva el espectro de cualquier
    testigo con (a, b, c) por arista, incluida la regla de signos de b. Es lo que
    justifica promediar sobre el grupo."""
    p = POLYS[nm]
    phis = curved(p)
    rng = np.random.default_rng(3)
    a, b, c = rng.normal(size=(3, p.E))
    _, s0 = sphere.witness_lmax(p, phis, a, b, c)
    for g in sphere.symmetry_group(p):
        _, s1 = sphere.witness_lmax(p, phis, *sphere.act_on_witness(p, g, a, b, c))
        for k in s0:
            assert s1[k] == pytest.approx(s0[k], abs=1e-9)


def test_regla_de_accion_detecta_un_signo_erroneo():
    """Control positivo: con b sin regla de signos, el espectro sí cambia."""
    p = POLYS["cubo"]
    phis = curved(p)
    rng = np.random.default_rng(3)
    a, b, c = rng.normal(size=(3, p.E))
    _, s0 = sphere.witness_lmax(p, phis, a, b, c)
    g = next(g for g in sphere.symmetry_group(p) if g.det < 0)
    a2, b2, c2 = sphere.act_on_witness(p, g, a, b, c)
    _, s1 = sphere.witness_lmax(p, phis, a2, -b2, c2)
    assert max(np.max(np.abs(s1[k] - s0[k])) for k in s0) > 1e-6


def test_cuboctaedro_regla_y_promedio():
    """En el cuboctaedro: la regla conserva el espectro y el promedio sobre el
    grupo da (a, s_e b, c) con los signos de `edge_orbits`, con lambda_max menor."""
    p = POLYS["cuboctaedro"]
    phis = curved(p)
    G = sphere.symmetry_group(p)
    orb = sphere.edge_orbits(p, G)[0]
    signs = np.array([orb.signs[m] for m in range(p.E)], dtype=float)
    assert np.all(np.abs(signs) == 1)
    rng = np.random.default_rng(4)
    a, b, c = rng.normal(size=(3, p.E))
    lam0, s0 = sphere.witness_lmax(p, phis, a, b, c)
    for g in G[:3] + G[-3:]:
        _, s1 = sphere.witness_lmax(p, phis, *sphere.act_on_witness(p, g, a, b, c))
        for k in s0:
            assert s1[k] == pytest.approx(s0[k], abs=1e-8)
    img = [sphere.act_on_witness(p, g, a, b, c) for g in G]
    am, bm, cm = (np.mean([w[q] for w in img], axis=0) for q in range(3))
    assert am == pytest.approx(np.full(p.E, am[0]), abs=1e-12)
    assert cm == pytest.approx(np.full(p.E, cm[0]), abs=1e-12)
    assert bm == pytest.approx(signs * (bm @ signs) / p.E, abs=1e-12)
    assert abs(bm[0]) > 1e-6
    assert sphere.witness_lmax(p, phis, am, bm, cm)[0] <= lam0 + 1e-9


def primal_ref(nm, caso, phis):
    """Primal del dual reducido. El del cubo (de 74 a 110 s y casi 10 GB por punto) se lee
    de data/sphere_primal.csv, que escribe scripts/sphere_series.py; sin ese
    fichero, se resuelve aquí."""
    path = DATA / "sphere_primal.csv"
    if nm == "cubo" and path.exists():
        for r in csv.DictReader(open(path)):
            if r["nombre"] == nm and r["caso"] == caso:
                return float(r["v_primal"])
    p = POLYS[nm]
    return sdp.primal_u1_graph(p.n, list(p.edges), phis)


@pytest.mark.parametrize("nm,v0,v1", [("tetraedro", 1 / 3, 1 / 3),
                                      ("octaedro", 1 / 3, 0.296296),
                                      ("cubo", 0.535565, 0.424853)])
def test_dual_reducido_contra_primal(nm, v0, v1):
    p = POLYS[nm]
    orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
    for caso, phis, v in (("trivial", np.zeros(p.E), v0), ("curvado", curved(p), v1)):
        res = sphere.reduced_dual(p, phis, orbs)
        assert res.gap < 1e-8
        assert res.lower <= res.v + 1e-10
        assert res.v == pytest.approx(v, abs=1e-5)
        assert res.v == pytest.approx(primal_ref(nm, caso, phis), abs=1e-6)


@pytest.mark.parametrize("nm", ["tetraedro", "octaedro", "cubo"])
def test_trivial_es_heisenberg(nm):
    """SU(2) hace óptimo a t = 1/(3E), aunque en tetraedro y octaedro hay una
    meseta t en [0, 2/(3E)] y el buscador puede pararse en otro punto de ella."""
    p = POLYS[nm]
    orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
    res = sphere.reduced_dual(p, np.zeros(p.E), orbs)
    d = sphere.ReducedDual(p, np.zeros(p.E))
    assert d.value(1 / (3 * p.E)) == pytest.approx(res.v, abs=1e-8)
    f = d.value
    lo, hi = sphere.optimal_interval(f, res.t, res.v, 0.01 / p.E)
    if nm == "cubo":
        assert res.t == pytest.approx(1 / (3 * p.E), abs=1e-4)
        assert hi - lo < 1e-3 / p.E
    else:
        assert lo == pytest.approx(0, abs=1e-5)
        assert hi == pytest.approx(2 / (3 * p.E), abs=1e-5)


def test_testigo_reducido_es_cota_valida_en_cubo():
    """El testigo reducido cumple N = 1 y su lambda_max coincide con la densa."""
    p = POLYS["cubo"]
    orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
    phis = curved(p)
    res = sphere.reduced_dual(p, phis, orbs)
    cp_, sn = np.cos(phis), np.sin(phis)
    alpha, beta = res.t * cp_, res.t * sn
    N = np.sum(2 * alpha * cp_ + 2 * beta * sn + res.c)
    assert N == pytest.approx(1, abs=1e-13)
    W = np.zeros((2**p.n, 2**p.n), dtype=complex)
    for (i, j), al, be in zip(p.edges, alpha, beta, strict=True):
        A, B, C = dense_ops(p.n, i, j)
        W += al * A + be * B + res.c * C
    assert np.max(np.abs(W - W.conj().T)) < 1e-14
    assert np.linalg.eigvalsh(W)[-1] == pytest.approx(res.v, abs=1e-10)


def test_eigsh_igual_a_densa():
    """Con dense_max pequeño (eigsh con todos los ncv), mismo valor que la densa."""
    p = POLYS["cuboctaedro"]
    phis = curved(p)
    t = 0.4 / p.E
    d1 = sphere.ReducedDual(p, phis)
    d2 = sphere.ReducedDual(p, phis, dense_max=50)
    assert d1.value(t) == pytest.approx(d2.value(t), abs=1e-9)


@pytest.mark.parametrize("x0", [0.0, 3.0])
def test_minimizador_convexo(x0):
    """Función con pico (mínimo 1 en x = 1,3) y lisa (mínimo 0 en x = -0,7)."""
    for f, xm, fm in ((lambda x: 1 + abs(x - 1.3), 1.3, 1.0),
                      (lambda x: (x + 0.7) ** 2, -0.7, 0.0)):
        x, val, lower, _ = sphere.minimize_convex(f, x0, 0.1, tol=1e-9)
        assert val == pytest.approx(fm, abs=2e-9)
        assert lower <= fm + 1e-12
        assert val - lower < 1e-9
        assert x == pytest.approx(xm, abs=1e-4)


def test_via_b_libre_con_dualidad_debil():
    """Vía con b libre (minimización anidada y planos de corte) en el octaedro, con
    un patrón de signos cualquiera. Cada testigo es factible, así que no puede bajar
    del primal, y por simetría ahí b = 0 ya es óptimo."""
    p = POLYS["octaedro"]
    orb = sphere.Orbit(list(range(p.E)), 1, False, {m: 1 for m in range(p.E)})
    res = sphere.reduced_dual(p, curved(p), [orb])
    assert res.free_b
    assert res.v == pytest.approx(0.296296, abs=1e-5)
    assert res.bound_kind == "planos de corte"
    assert res.gap < 1e-7
    assert res.lower <= 8 / 27 + 1e-7


def test_orbitas_cubren_las_aristas():
    for nm, p in POLYS.items():
        orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
        assert sorted(itertools.chain.from_iterable(o.edges for o in orbs)) == list(range(p.E))


# ---------------------------------------------------------------------------
# Gauge SO(3) ambiente
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nm", sphere.NAMES)
def test_rotaciones_ambiente(nm):
    p = POLYS[nm]
    Gs, ls = sphere.ambient_rotations(p)
    x = sphere.unit_vertices(p)
    for (i, j), G, le in zip(p.edges, Gs, ls, strict=True):
        assert G @ x[j] == pytest.approx(x[i], abs=1e-12)
        assert G.T @ G == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(G) == pytest.approx(1, abs=1e-12)
        assert le == pytest.approx(np.arccos(x[i] @ x[j]), abs=1e-9)
        assert rotation_angle(G) == pytest.approx(le, abs=1e-7)


@pytest.mark.parametrize("nm", sphere.NAMES)
def test_holonomia_ambiente_de_cara(nm):
    """Cada cara antihoraria: rotación de f_p alrededor de -x_0, -f_p alrededor de x_0."""
    p = POLYS[nm]
    Gs, _ = sphere.ambient_rotations(p)
    G = {e: g for e, g in zip(p.edges, Gs, strict=True)}
    x = sphere.unit_vertices(p)
    for cyc, f in zip(p.faces, sphere.face_fluxes(p), strict=True):
        H = np.eye(3)
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            H = H @ (G[a, b] if a < b else G[b, a].T)
        assert H == pytest.approx(sphere._rotation(-x[cyc[0]], f), abs=1e-10)


@pytest.mark.parametrize("nm", sphere.NAMES)
def test_fases_ambiente(nm):
    """O_i^T G_ij O_j = R_z(phi): flujo -f_p, y +f_p con R_x(pi)."""
    p = POLYS[nm]
    f = sphere.face_fluxes(p)
    h = sphere.face_holonomies(p, sphere.ambient_phases(p))
    assert np.max(np.abs(np.mod(h + f + np.pi, 2 * np.pi) - np.pi)) < 1e-9
    h = sphere.face_holonomies(p, sphere.ambient_phases(p, flip=True))
    assert np.max(np.abs(np.mod(h - f + np.pi, 2 * np.pi) - np.pi)) < 1e-9


@pytest.mark.parametrize("nm", ["octaedro", "cubo"])
def test_gauge_ambiente_mismo_espectro(nm):
    """Las fases ambiente, sin flip y con flip, dan el mismo v que `phases`, y con
    flip el mismo dato salvo gauge."""
    p = POLYS[nm]
    orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
    v = sphere.reduced_dual(p, curved(p), orbs).v
    for flip in (False, True):
        assert sphere.reduced_dual(p, sphere.ambient_phases(p, flip=flip), orbs).v == pytest.approx(v, abs=1e-8)
    a = np.full(p.E, 1 / (3 * p.E))
    s1 = sphere.witness_lmax(p, curved(p), a, 0 * a, a)[1]
    s2 = sphere.witness_lmax(p, sphere.ambient_phases(p, flip=True), a, 0 * a, a)[1]
    for k in s1:
        assert s1[k] == pytest.approx(s2[k], abs=1e-9)


@pytest.mark.slow
def test_primal_so3_octaedro():
    """Primal SO(3) completo con las G_ij del gauge ambiente: 8/27."""
    p = POLYS["octaedro"]
    Gs, _ = sphere.ambient_rotations(p)
    assert sdp.primal(p.n, Gs, edges=list(p.edges)) == pytest.approx(8 / 27, abs=1e-6)


# ---------------------------------------------------------------------------
# Índice cromático fraccionario
# ---------------------------------------------------------------------------


def graph(n, edges):
    return sphere.Polyhedron("grafo", np.zeros((n, 3)), tuple(sorted(tuple(sorted(e)) for e in edges)), ())


def petersen():
    return graph(10, [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
                 + [(5 + i, 5 + (i + 2) % 5) for i in range(5)])


def edmonds(g):
    """max(Delta, max de |E(S)|/floor(|S|/2) en S impar)."""
    deg = np.zeros(g.n, dtype=int)
    for i, j in g.edges:
        deg[i] += 1
        deg[j] += 1
    best = float(deg.max())
    for size in range(3, g.n + 1, 2):
        for S in itertools.combinations(range(g.n), size):
            s = set(S)
            best = max(best, sum(i in s and j in s for i, j in g.edges) / (size // 2))
    return best


def test_chi_f_grafos_de_control():
    c5 = graph(5, [(i, (i + 1) % 5) for i in range(5)])
    assert sphere.fractional_chromatic_index(c5) == pytest.approx(5 / 2, abs=1e-9)
    pet = petersen()
    assert sphere.fractional_chromatic_index(pet) == pytest.approx(3, abs=1e-9)
    assert sphere.chromatic_index(pet) == 4  # Petersen es de clase 2: chi'_f < chi'


@pytest.mark.parametrize("nm,valor", [("cubo", 3), ("dodecaedro", 3), ("tetraedro", 3),
                                      ("octaedro", 4), ("icosaedro", 5), ("cuboctaedro", 4),
                                      ("rombododecaedro", 4)])
def test_chi_f_poliedros(nm, valor):
    assert sphere.fractional_chromatic_index(POLYS[nm]) == pytest.approx(valor, abs=1e-9)


@pytest.mark.parametrize("nm", ["tetraedro", "octaedro", "cubo", "cuboctaedro"])
def test_chi_f_igual_a_edmonds(nm):
    assert sphere.fractional_chromatic_index(POLYS[nm]) == pytest.approx(edmonds(POLYS[nm]), abs=1e-9)


# ---------------------------------------------------------------------------
# Testigo por órbitas
# ---------------------------------------------------------------------------


def cube_cap():
    """Dos caras del cubo con una arista común, de mi numeración: 6 qubits y 7 aristas."""
    p = POLYS["cubo"]
    a, b = next((a, b) for a, b in itertools.combinations(p.faces, 2) if len(set(a) & set(b)) == 2)
    return sorted(set(a) | set(b))


@pytest.mark.parametrize("caso", ["trivial", "curvado"])
def test_orbitdual_casquete_cubo(caso):
    p = POLYS["cubo"]
    vs = cube_cap()
    phis = np.zeros(p.E) if caso == "trivial" else curved(p)
    sub, ph, _ = sphere.induced_subgraph(p, phis, vs)
    assert (sub.n, len(sub.edges)) == (6, 7)
    grupo = sphere.subset_group(p, sphere.symmetry_group(p), vs)
    orbs = sphere.edge_orbits(sub, grupo)
    assert len(grupo) == 4
    assert sorted((len(o.edges), o.stabilizer, o.b_zero) for o in orbs) == [(1, 4, True), (2, 2, False), (4, 1, False)]
    res = sphere.OrbitDual(sub, ph, orbs).minimize()
    assert res.gap < 1e-8
    assert res.lower <= res.v and not res.box_active
    assert res.v == pytest.approx(sdp.primal_u1_graph(sub.n, list(sub.edges), ph), abs=1e-6)
    if caso == "trivial":
        assert res.v == pytest.approx(5 / 9, abs=1e-7)


def test_regla_de_accion_en_el_subgrupo():
    """La regla de signos sirve para el subgrupo que conserva el casquete."""
    p = POLYS["cubo"]
    vs = cube_cap()
    sub, ph, _ = sphere.induced_subgraph(p, curved(p), vs)
    grupo = sphere.subset_group(p, sphere.symmetry_group(p), vs)
    rng = np.random.default_rng(5)
    a, b, c = rng.normal(size=(3, len(sub.edges)))
    _, s0 = sphere.witness_lmax(sub, ph, a, b, c)
    for g in grupo:
        _, s1 = sphere.witness_lmax(sub, ph, *sphere.act_on_witness(sub, g, a, b, c))
        for k in s0:
            assert s1[k] == pytest.approx(s0[k], abs=1e-9)


def test_orbitdual_sin_simetria_octaedro():
    """Cada arista su órbita (36 parámetros), sin usar simetría: vuelve a dar 8/27."""
    p = POLYS["octaedro"]
    orbs = sphere.edge_orbits(p, sphere.identity_group(p))
    res = sphere.OrbitDual(p, curved(p), orbs, dense_max=2000, k_eigs=8, near=0.05).minimize()
    assert res.gap < 1e-8
    assert res.v == pytest.approx(8 / 27, abs=1e-7)


def test_orbitdual_una_orbita_igual_que_reduced_dual():
    p = POLYS["cubo"]
    orbs = sphere.edge_orbits(p, sphere.symmetry_group(p))
    res = sphere.OrbitDual(p, curved(p), orbs).minimize()
    assert res.v == pytest.approx(sphere.reduced_dual(p, curved(p), orbs).v, abs=1e-8)


def test_witness_block_top_eigsh_y_denso():
    p = POLYS["cuboctaedro"]
    rng = np.random.default_rng(6)
    a, b, c = rng.normal(size=(3, p.E)) / p.E
    d = sphere.witness_lmax(p, curved(p), a, b, c, spins=[6])[0]
    assert sphere.witness_block_top(p, curved(p), a, b, c, 6) == pytest.approx(d, abs=1e-8)


def _rotaciones_al_azar(rng, escala, n):
    from scipy.spatial.transform import Rotation
    return [Rotation.from_rotvec(escala * rng.normal(size=3) / np.sqrt(3)).as_matrix() for _ in range(n)]


@pytest.mark.parametrize("n,aristas,chi_f", [
    (4, [(0, 1), (1, 2), (2, 3), (3, 0)], 2),
    (5, [(i, (i + 1) % 5) for i in range(5)], 5 / 2),
    (4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)], 3),
    (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 3),
    pytest.param(6, [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)], 3, marks=pytest.mark.slow),
])
def test_cota_holonomia_pequena_no_abeliana(n, aristas, chi_f):
    """Proposición de holonomía pequeña contra el primal, con datos SO(3) al azar:
    anillos de 4 y 5, diamante, K4 y escalera abierta 2x3."""
    rng = np.random.default_rng(2)
    v0 = sdp.primal(n, [np.eye(3)] * len(aristas), edges=aristas)
    for escala in (0.05, 0.2, 0.5, 1.5):
        Rs = _rotaciones_al_azar(rng, escala, len(aristas))
        th = max(rotation_angle(R) for R in Rs)
        v = sdp.primal(n, Rs, edges=aristas)
        assert v >= v0 / (1 + 3 * chi_f * th) - 1e-6
        den = 1 - (chi_f + 1) * v0 * th**2 / 2
        if den > 0:
            assert v <= v0 / den + 1e-6


def test_petersen_seis_emparejamientos_perfectos():
    """Los 6 emparejamientos perfectos de Petersen cubren cada arista dos veces."""
    pet = petersen()
    perfectos = [m for m in sphere.all_matchings(pet) if len(m) == 5]
    assert len(perfectos) == 6
    cuenta = {e: sum(e in m for m in perfectos) for e in range(len(pet.edges))}
    assert set(cuenta.values()) == {2}


def test_quitar_un_vertice_no_cambia_v_sin_holonomia():
    """El anillo de 4 y el cubo, sin datos, tienen el mismo v que sin un vértice; con
    holonomía, no: es el Teorema del anillo de 4 frente a la cadena de 3."""
    c4 = [(0, 1), (1, 2), (2, 3), (3, 0)]
    p3 = [(0, 1), (1, 2)]
    assert sdp.primal_u1_graph(3, p3, [0.0, 0.0]) == pytest.approx(2 / 3, abs=1e-6)
    assert sdp.primal_u1_graph(4, c4, [0.0] * 4) == pytest.approx(2 / 3, abs=1e-6)
    th = 2.0
    assert sdp.primal_u1_graph(4, c4, [th / 4] * 4) == pytest.approx(2 / (3 + np.sin(th / 2)), abs=1e-6)
    p = sphere.polyhedron("cubo")
    q = next(k for k, v in enumerate(p.verts) if tuple(v) == (1.0, 1.0, 1.0))
    sub, ph, _ = sphere.induced_subgraph(p, np.zeros(p.E), [v for v in range(8) if v != q])
    assert sdp.primal_u1_graph(sub.n, list(sub.edges), ph) == pytest.approx(0.53556549, abs=1e-6)


def test_pentagono_sube_con_el_flujo():
    """El anillo impar está frustrado sin flujo: flujo de cara del dodecaedro lo sube."""
    c5 = [(i, (i + 1) % 5) for i in range(5)]
    v0 = sdp.primal_u1_graph(5, c5, [0.0] * 5)
    v1 = sdp.primal_u1_graph(5, c5, [np.pi / 15] * 5)
    assert v0 == pytest.approx(0.4981424, abs=1e-6)
    assert v1 == pytest.approx(0.5053819, abs=1e-6)

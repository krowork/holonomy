"""Pruebas de la geometría de las cadenas de diamantes."""
import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holonomy import diamond, sdp
from holonomy.paulis import Rz, holonomy

# Aviso de CVXPY al construir los primales (sdp.py, periodic.py); no es un hallazgo.
pytestmark = pytest.mark.filterwarnings("ignore:Initializing a Constant with a nested list")


def test_diamante_simple_aristas():
    assert diamond.chain_edges(1) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    assert diamond.n_sites(1) == 4
    assert len(diamond.chain_edges(3, periodic=True)) == 15
    assert diamond.n_sites(3, True) == 9
    assert diamond.chain_edges(2, periodic=True)[-2:] == [(4, 0), (5, 0)]
    assert diamond.edge_kinds(2) == list(diamond.KINDS) * 2


def test_flujos_ida_y_vuelta():
    fA, fB, psi = [0.3, -1.1, 2.0], [0.7, 0.2, -0.4], 0.9
    phis = diamond.abelian_phases(3, fA, fB, periodic=True, psi=psi)
    a, b, hol = diamond.triangle_fluxes(3, phis, periodic=True)
    s = (sum(fA) + sum(fB)) / 2
    assert a == pytest.approx(fA, abs=1e-12)
    assert b == pytest.approx(fB, abs=1e-12)
    assert hol == pytest.approx((psi + s, psi - s), abs=1e-12)
    phis = diamond.abelian_phases(3, fA, fB)
    a, b, hol = diamond.triangle_fluxes(3, phis)
    assert a == pytest.approx(fA, abs=1e-12)
    assert b == pytest.approx(fB, abs=1e-12)
    assert hol is None


def test_holonomia_no_abeliana():
    rng = np.random.default_rng(7)
    k = 3
    pairs = [tuple(Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
                   for _ in range(2)) for _ in range(k)]
    edges = diamond.chain_edges(k)
    Rs = dict(zip(edges, diamond.chain_rotations(k, pairs)))

    def R(a, b):
        return Rs[(a, b)] if (a, b) in Rs else Rs[(b, a)].T

    for m, (HA, HB) in enumerate(pairs):
        j, u, w, jp = 3 * m, 3 * m + 1, 3 * m + 2, 3 * m + 3
        hA = holonomy([R(j, u), R(u, w), R(w, j)])
        hB = holonomy([R(u, jp), R(jp, w), R(w, u)])
        assert hA == pytest.approx(HA, abs=1e-12)
        assert hB == pytest.approx(HB, abs=1e-12)


def test_abeliano_coincide_con_rotaciones():
    fA, fB = 0.8, -1.3
    edges = diamond.chain_edges(1)
    Rs = diamond.chain_rotations(1, [(Rz(fA), Rz(fB))])
    v = sdp.primal(4, Rs, edges=edges)
    phis = diamond.abelian_phases(1, fA, fB)
    assert v == pytest.approx(sdp.primal_u1_graph(4, edges, phis), abs=1e-7)


def test_pares_relativos_mismos_invariantes():
    Q = Rotation.random(random_state=3).as_matrix()
    (A1, B1), (A2, B2) = diamond.relative_pairs(0.7, 1.9, 0.6, Q)
    assert A2 == pytest.approx(Q @ A1 @ Q.T, abs=1e-12)
    assert B2 == pytest.approx(Q @ B1 @ Q.T, abs=1e-12)
    assert np.trace(A1) == pytest.approx(np.trace(Rz(0.7)), abs=1e-12)
    assert np.trace(B1) == pytest.approx(np.trace(Rz(1.9)), abs=1e-12)


def test_periodica_k1_rechazada():
    with pytest.raises(ValueError):
        diamond.chain_edges(1, periodic=True)


OPTS = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)


# k = 3 no puede ejecutarse aquí: primal_u1_graph pide a Clarabel un bloque
# denso de 31878^2 reales (8,1 GB, sector de 4 giros, dimensión hermítica 126)
# y aborta con ulimit -v 8000000 y con 12000000 (medido el 2026-11-03). Los
# contrastes de k = 3 son test_primal_reducido_k3 y test_testigo_kronecker_k3.
@pytest.mark.parametrize("k", [2, pytest.param(3, marks=pytest.mark.skip(
    reason="primal_u1_graph de 9 qubits: Clarabel pide 8,1 GB de una vez y aborta"))])
@pytest.mark.parametrize("fA,fB,psi", [(np.pi / 2, np.pi / 2, 0.0), (np.pi, -np.pi, 0.0),
                                       (0.0, 0.0, np.pi)])
@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_cadena_cerrada_dual_igual_primal(k, fA, fB, psi):
    # Dual reducido (U(1), traslación, inversión temporal) = primal U(1) sin reducir.
    ph = diamond.abelian_phases(k, fA, fB, periodic=True, psi=psi)
    v, _, st = diamond.DiamondDual(k).v_dual_sdp(ph, **OPTS)
    # Clarabel acaba a menudo 'optimal_inaccurate' a 1e-10; se acepta porque
    # el valor se contrasta con el primal.
    assert st in ("optimal", "optimal_inaccurate")
    vp = sdp.primal_u1_graph(diamond.n_sites(k, True), diamond.chain_edges(k, True), ph, **OPTS)
    assert v == pytest.approx(vp, abs=1e-6)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_cadena_cerrada_cg():
    # v_dual_cg con bloques dispersos (Lanczos) y con solo los bloques k <= n // 2
    # = SDP completo, en un punto genérico (fA != fB, psi != 0).
    ph = diamond.abelian_phases(3, np.pi / 2, np.pi / 3, periodic=True, psi=0.3 * np.pi)
    vs, _, st = diamond.DiamondDual(3).v_dual_sdp(ph, **OPTS)
    # 'optimal_inaccurate' se acepta: el valor se contrasta con el de v_dual_cg.
    assert st in ("optimal", "optimal_inaccurate")
    for D in (diamond.DiamondDual(3, dense_max=4), diamond.DiamondDual(3, half=True)):
        vc, _, st, gap = D.v_dual_cg(ph, **OPTS)
        assert st == "optimal" and gap < 1e-8
        # El peor estado de los SDP restringidos queda registrado.
        assert D.last_solver_status in ("optimal", "optimal_inaccurate")
        assert vc == pytest.approx(vs, abs=1e-8)


def test_diamond_dual_traslacion():
    D = diamond.DiamondDual(3)
    assert D.perm == [(i + 3) % 9 for i in range(9)]
    assert len(D.KEYS) == 15 and all(name in "ABC" for name, _ in D.KEYS)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
@pytest.mark.parametrize("fA,fB,psi", [(np.pi / 2, np.pi / 2, 0.0), (0.0, 0.0, np.pi)])
def test_primal_reducido_k2(fA, fB, psi):
    # El primal reducido por U(1) y traslación = primal U(1) sin reducir.
    ph = diamond.abelian_phases(2, fA, fB, periodic=True, psi=psi)
    v, st = diamond.ring_primal(2, ph, **OPTS)
    assert st in ("optimal", "optimal_inaccurate")
    vp = sdp.primal_u1_graph(6, diamond.chain_edges(2, True), ph, **OPTS)
    assert v == pytest.approx(vp, abs=1e-6)


@pytest.mark.slow  # pico 5,4 GB de RSS; necesita ulimit -v >= 8 GB
@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_primal_reducido_k3():
    # k = 3 sin flujo de triángulo, psi = pi: el primal reducido (cota
    # inferior) = el dual reducido (cota superior) a 1e-7, y v < 1/3.
    ph = diamond.abelian_phases(3, 0.0, 0.0, periodic=True, psi=np.pi)
    vp, _ = diamond.ring_primal(3, ph, **OPTS)
    vd, _, st, _ = diamond.DiamondDual(3, half=True).v_dual_cg(ph, r0=12, **OPTS)
    assert st == "optimal"
    assert vd == pytest.approx(vp, abs=1e-7)
    assert vd < 1 / 3 - 0.02


def _testigo_denso(n, edges, kinds, coef):
    """W = sum_e (a A_e + b B_e + c C_e) con productos de Kronecker de Paulis,
    A = -(XX+YY), B = XY-YX, C = -ZZ (convención de sdp.u1_sector_operators)."""
    from holonomy.paulis import X, Y, Z, op
    W = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for (i, j), kd in zip(edges, kinds):
        a, b, c = (coef.get((name, kd), 0.0) for name in "ABC")
        W += -a * (op(n, {i: X, j: X}) + op(n, {i: Y, j: Y}))
        W += b * (op(n, {i: X, j: Y}) - op(n, {i: Y, j: X}))
        W += -c * op(n, {i: Z, j: Z})
    return W


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_testigo_kronecker_k3():
    # Comprobación independiente de la reducción por bloques: el testigo de
    # v_dual_cg en k = 3, f = 0 construido denso (512 x 512) sin bloques
    # tiene el mismo lambda_max, y N(y) = 1 calculada arista a arista con
    # T = -R_z(phi_e). Así v_max <= lambda_max / N < 1/3 sin depender de
    # symmetric_blocks.
    k, n = 3, 9
    ph = diamond.abelian_phases(k, 0.0, 0.0, periodic=True, psi=0.0)
    D = diamond.DiamondDual(k, half=True)
    v, y, st, _ = D.v_dual_cg(ph, r0=12, **OPTS)
    assert st == "optimal"
    coef = D.witness_terms(y)
    edges, kinds = diamond.chain_edges(k, True), diamond.edge_kinds(k, True)
    W = _testigo_denso(n, edges, kinds, coef)
    assert np.allclose(W, W.conj().T)
    lam = np.linalg.eigvalsh(W)[-1]
    assert lam == pytest.approx(D.lam_max(y), abs=1e-9)
    N = sum(2 * coef[("A", kd)] * np.cos(p) + 2 * coef[("B", kd)] * np.sin(p) + coef[("C", kd)]
            for kd, p in zip(kinds, ph))
    assert N == pytest.approx(1.0, abs=1e-9)
    assert D.norm_grad(ph) @ y == pytest.approx(1.0, abs=1e-9)
    assert lam / N == pytest.approx(v, abs=1e-9)
    assert lam / N < 1 / 3 - 0.02


def test_punto_de_la_orbita_de_gauge(monkeypatch):
    # Con thA = dthB = dalpha = 0, orbit_point cae en la órbita de x*:
    # theta_B = pi y alpha = arccos(n_B . Q n_B) = pi/2; dalpha desplaza alpha.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    m = importlib.import_module("diamond_chain2_min")
    x = m.orbit_point(0.0, 0.0, 0.0)
    assert x == pytest.approx(m.CFG["XSTAR"], abs=1e-12)
    for g in m.CFG["ORBITA_G"]:
        for p1, p2 in m.CFG["ORBITA_PHI"]:
            x = m.orbit_point(g, p1, p2)
            assert x[0] == 0.0 and x[1] == pytest.approx(np.pi, abs=1e-15) and x[2] == g
            assert m.axis_angle(x) == pytest.approx(np.pi / 2, abs=1e-12)
            y = m.orbit_point(g, p1, p2, thA=0.1, dthB=-0.2, dalpha=0.1)
            assert y[0] == 0.1 and y[1] == pytest.approx(np.pi - 0.2, abs=1e-15)
            assert m.axis_angle(y) == pytest.approx(np.pi / 2 + 0.1, abs=1e-12)


@pytest.mark.parametrize("g", [0.0, 0.3 * np.pi, np.pi / 2, np.pi])
def test_reflexion_g(monkeypatch, g):
    # G de la simetría S1 (Lema de reflexión): rotación con G z = -n_B y
    # G n_B = -z, también en gamma = 0 y pi, donde la base (z, n, z x n) es singular.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    m = importlib.import_module("diamond_chain2_min")
    z = np.array([0.0, 0.0, 1.0])
    n = np.array([np.sin(g), 0.0, np.cos(g)])
    G = m.reflexion_g(g)
    assert G @ G.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(G) == pytest.approx(1.0, abs=1e-12)
    assert G @ z == pytest.approx(-n, abs=1e-12)
    assert G @ n == pytest.approx(-z, abs=1e-12)
    assert np.all(np.isfinite(m.swap_q(g, np.array([0.5, 0.3, -0.2]))))

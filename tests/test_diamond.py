"""Pruebas de la geometría de las cadenas de diamantes."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holonomy import diamond, sdp
from holonomy.paulis import Rz, holonomy


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

"""Pruebas del dual general (GraphDual): SDP completo y generación de columnas."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from holonomy import diamond, sdp
from holonomy.gdual import GraphDual
from holonomy.ladder import ladder_edges
from holonomy.paulis import Rx, Rz

OPTS = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)


def f4(theta):
    return 2 / (3 + np.sin(theta / 2))


def tree_gauge_open(L, tops):
    """Como scripts/ladder.py: identidad salvo en la pata superior, que lleva
    las holonomías de las plaquetas (orden de ladder_edges(L))."""
    return [np.eye(3)] * (L - 1) + list(tops) + [np.eye(3)] * L


def diamante_aleatorio(seed):
    pares = [tuple(Rotation.random(random_state=seed + s).as_matrix() for s in (0, 1))]
    return diamond.chain_edges(1), diamond.chain_rotations(1, pares)


@pytest.mark.parametrize("theta", [0.0, 1.0, np.pi])
def test_anillo4(theta):
    v, _, _ = GraphDual(4, sdp.ring_edges(4)).v_dual_sdp([Rz(theta / 4)] * 4, **OPTS)
    assert v == pytest.approx(f4(theta), abs=1e-7)


@pytest.mark.parametrize("seed", [11, 29])
def test_diamante_un_tercio(seed):
    edges, Rs = diamante_aleatorio(seed)
    D = GraphDual(4, edges)
    vs, _, _ = D.v_dual_sdp(Rs, **OPTS)
    vc, _, st, gap = D.v_dual_cg(Rs, **OPTS)
    assert vs == pytest.approx(1 / 3, abs=1e-7)
    assert vc == pytest.approx(1 / 3, abs=1e-7)
    assert st == "optimal" and gap < 1e-7


@pytest.mark.slow
def test_cg_igual_sdp_n6():
    edges = ladder_edges(3)
    Rs = tree_gauge_open(3, [Rz(np.pi), Rx(np.pi)])
    D = GraphDual(6, edges)
    vs, _, _ = D.v_dual_sdp(Rs, **OPTS)
    vc, _, st, gap = D.v_dual_cg(Rs, **OPTS)
    vp = sdp.primal(6, Rs, edges=edges, **OPTS)
    assert st == "optimal" and gap < 1e-7
    assert vc == pytest.approx(vs, abs=1e-7)
    assert vc == pytest.approx(vp, abs=1e-7)


def test_cadena2_abeliana():
    # 7 qubits: contraste con el primal U(1) (el primal completo no cabe con datos
    # no abelianos; aquí sirve el abeliano).
    edges = diamond.chain_edges(2)
    phis = diamond.abelian_phases(2, np.pi / 2, np.pi / 2)
    Rs = [Rz(p) for p in phis]
    vc, _, st, gap = GraphDual(7, edges).v_dual_cg(Rs, **OPTS)
    vp = sdp.primal_u1_graph(7, edges, phis, **OPTS)
    assert st == "optimal" and gap < 1e-7
    assert vc == pytest.approx(vp, abs=1e-7)


def test_caja_reportada():
    # Con una caja en la que N(y) = 1 es imposible, el estado no puede parecer un resultado.
    edges, Rs = diamante_aleatorio(11)
    _, _, st, _ = GraphDual(4, edges).v_dual_cg(Rs, box=1e-3, **OPTS)
    assert st == "en_caja"


def test_rango_comprimido_contiene_el_completo():
    # Con U el conjunto factible es mayor: cada rango contiene al del SDP completo.
    # Se usa un subespacio propio (10 de las 16 columnas finales de v_dual_cg, que
    # en el diamante acaba con el espacio entero) para que la comparación no sea trivial.
    edges, Rs = diamante_aleatorio(11)
    D = GraphDual(4, edges)
    v, _, _, _ = D.v_dual_cg(Rs, **OPTS)
    U = [D.last_U[0][:, :10]]
    full = D.witness_spread(Rs, v, 1e-2, **OPTS)
    comp = D.witness_spread(Rs, v, 1e-2, U=U, **OPTS)
    assert len(full) == len(comp) == 9 * len(edges)
    for (a, b), (c, d) in zip(full, comp):
        assert c <= a + 1e-6 and d >= b - 1e-6
    assert max((d - c) - (b - a) for (a, b), (c, d) in zip(full, comp)) > 1e-2

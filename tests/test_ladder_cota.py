"""Cota de holonomía pequeña (Lema pesos y Proposiciones cotapeq y cotaflujo de escaleras)."""
import numpy as np
import pytest

from holonomy import sdp
from holonomy.ladder import abelian_phases, ladder_edges
from holonomy.paulis import X, Y, Z, op

ANILLO5 = [(i, (i + 1) % 5) for i in range(5)]
ANILLO6 = [(i, (i + 1) % 6) for i in range(6)]
PRISMA = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
DIAMANTE = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
ESCALERA = ladder_edges(3)


def _testigo(n, edges, a, b, c):
    W = np.zeros((2**n, 2**n), complex)
    for k, (i, j) in enumerate(edges):
        p = lambda P, Q: op(n, {i: P, j: Q})
        W += a[k] * (-(p(X, X) + p(Y, Y))) + b[k] * (p(X, Y) - p(Y, X)) + c[k] * (-p(Z, Z))
    return W


@pytest.mark.parametrize("n,edges,chi", [(5, ANILLO5, 3), (6, ANILLO6, 2), (6, PRISMA, 3),
                                         (6, ESCALERA, 3), (4, DIAMANTE, 3)])
def test_pesos_del_testigo(n, edges, chi):
    rng = np.random.default_rng(1)
    for t in range(100):
        s = rng.choice([0.1, 1.0, 5.0], size=3)
        a, b, c = (s[i] * rng.normal(size=len(edges)) for i in range(3))
        if t % 4 == 0:
            c = -5 * np.abs(c)
        lam = np.linalg.eigvalsh(_testigo(n, edges, a, b, c)).max()
        assert 2 * np.hypot(a, b).sum() <= (chi + 1) * lam + 1e-10


@pytest.mark.parametrize("n,edges,chi", [(5, ANILLO5, 3), (6, ANILLO6, 2), (6, ESCALERA, 3)])
@pytest.mark.parametrize("seed,escala", [(0, 0.1), (1, 0.3), (2, 1.0)])
def test_cota_en_el_primal(n, edges, chi, seed, escala):
    v0 = sdp.primal_u1_graph(n, edges, np.zeros(len(edges)))
    phis = escala * np.random.default_rng(seed).uniform(-1, 1, len(edges))
    v = sdp.primal_u1_graph(n, edges, phis)
    pm = np.abs(phis).max()
    assert v >= v0 / (1 + (chi + 1) * pm) - 1e-6
    den = 1 - (chi + 1) * v0 * pm**2 / 2
    if den > 0:
        assert v <= v0 / den + 1e-6


@pytest.mark.parametrize("L", [3, 4, 5, 6, 7])
def test_escalera_tres_emparejamientos(L):
    """El ciclo hamiltoniano da dos emparejamientos, y el resto de aristas, el tercero."""
    edges = {frozenset(e) for e in ladder_edges(L, periodic=True)}
    ciclo = list(range(L)) + list(range(2 * L - 1, L - 1, -1))
    h = [frozenset((ciclo[i], ciclo[(i + 1) % (2 * L)])) for i in range(2 * L)]
    assert set(h) <= edges
    for M in (h[::2], h[1::2], list(edges - set(h))):
        vs = [v for e in M for v in e]
        assert len(vs) == len(set(vs)) == 2 * L
    assert len(edges) == 3 * L


@pytest.mark.parametrize("L,F,psi", [(4, np.pi, 0.0), (5, 2 * np.pi, np.pi / 2)])
def test_gauge_simetrico_acota_las_fases(L, F, psi):
    phis = abelian_phases(L, [F / L] * L, periodic=True, psi=psi)
    assert np.abs(phis).max() == pytest.approx((F / 2 + psi) / L)

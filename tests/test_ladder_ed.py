"""Escalera XXZ con giro por bloques de patas (holonomy.ladder_ed) frente al hamiltoniano denso."""
from itertools import product

import numpy as np
import pytest

from holonomy.ladder_ed import TwistedLadder, dense_hamiltonian


@pytest.mark.parametrize("L, x, D, Dr, Psi", [(L, *p, Psi) for L, p, Psi in
                                              product((3, 4), ((1, 1, 1), (0.8, 0.86, 1.15)), (0.0, np.pi))])
def test_espectro_sz0(L, x, D, Dr, Psi):
    T = TwistedLadder(L, x, D, Dr, Psi)
    M = np.column_stack([T.matvec(e) for e in np.eye(T.dim)])
    assert np.allclose(M, M.T)
    H = dense_hamiltonian(L, x, D, Dr, Psi)
    sel = np.array([bin(k).count("1") for k in range(4**L)]) == L
    assert np.allclose(np.linalg.eigvalsh(M), np.linalg.eigvalsh(H[np.ix_(sel, sel)]), atol=1e-12)


def test_hellmann_feynman():
    T = TwistedLadder(5, 0.7, 0.9, 1.2, np.pi)
    E, u, res = T.ground()
    assert res < 1e-9
    assert E == pytest.approx(T.parts(u) @ [1, T.Delta, T.x, T.c_r], abs=1e-12)

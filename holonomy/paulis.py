"""Operadores de Pauli en n qubits y rotaciones SO(3).

Convención: el qubit 0 es el más significativo y |0> es el autoestado de Z
con valor +1.
"""
from functools import lru_cache

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (X, Y, Z)


def op(n, factors):
    """Producto tensorial con factors[k] en el qubit k e identidad en el resto."""
    m = np.ones((1, 1), dtype=complex)
    for k in range(n):
        m = np.kron(m, factors.get(k, I2))
    return m


@lru_cache(maxsize=None)
def _pauli_cache(n, key):
    return op(n, {k: PAULI[a] for k, a in key})


def pauli_string(n, pairs):
    """Producto de Paulis: pairs = ((sitio, índice 0/1/2), ...). Cacheado."""
    return _pauli_cache(n, tuple(sorted(pairs)))


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_angle(R):
    """Ángulo de rotación en [0, pi] de R en SO(3): tr R = 1 + 2 cos(theta)."""
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def holonomy(Rs):
    """H = R_01 R_12 ... R_{n-1,0} para un anillo con matrices por arista."""
    H = np.eye(3)
    for R in Rs:
        H = H @ R
    return H


def werner_correlations(v, R):
    """Matriz de correlaciones T = -v R de la perspectiva de Werner rotada."""
    return -v * np.asarray(R)


def werner_state(v, R):
    """rho_e(v) = (I + v sum_ab T_ab sigma_a (x) sigma_b)/4 con T = -R."""
    T = -np.asarray(R)
    rho = np.eye(4, dtype=complex)
    for a in range(3):
        for b in range(3):
            rho = rho + v * T[a, b] * np.kron(PAULI[a], PAULI[b])
    return rho / 4


def edge_correlations(rho, n, i, j):
    """Correlaciones <sigma_a^i sigma_b^j> (3x3) y marginales locales de rho."""
    T = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            T[a, b] = np.real(np.trace(rho @ pauli_string(n, ((i, a), (j, b)))))
    mi = np.array([np.real(np.trace(rho @ pauli_string(n, ((i, a),)))) for a in range(3)])
    mj = np.array([np.real(np.trace(rho @ pauli_string(n, ((j, a),)))) for a in range(3)])
    return T, mi, mj

"""Diagonalización exacta del anillo XXZ con flujo.

H(Delta, Phi) = sum_i [ 1/2 (e^{i Phi/n} S+_i S-_{i+1} + h.c.) + Delta Sz_i Sz_{i+1} ],
eps_n(Delta, Phi) = -E0/n.

Se trabaja por sectores de número de espines arriba M. Para Delta > -1 el estado
fundamental está en los sectores de magnetización mínima (M = floor(n/2) y
ceil(n/2)) o, cerca de Delta = -1 y para n pequeño, es el ferromagnético
totalmente polarizado (M = 0 o n, energía n Delta/4, independiente del flujo).
`eps` tiene en cuenta ambos; `eps_full` recorre todos los sectores para
comprobarlo (scripts/sec.py).

Convención de bits: aquí el sitio i es el bit (s >> i) & 1, al revés que en
`holonomy.paulis` (qubit 0 más significativo). Para las energías da igual (es
una reflexión, y el signo del flujo no cambia el espectro); para combinar
autovectores de `ground_state` con operadores de `paulis`, hay que invertir
el orden de los bits.
"""
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DENSE_MAX = 600  # dimensión hasta la que se diagonaliza en denso (después, Lanczos)


@lru_cache(maxsize=None)
def sector_operators(n, M):
    """Devuelve (diag_zz, K) en el sector con M espines arriba (bit = 1).

    diag_zz[s] = sum_i Sz_i Sz_{i+1}; K = sum_i S+_i S-_{i+1} (sparse).
    """
    states = np.array([s for s in range(2 ** n) if bin(s).count("1") == M], dtype=np.int64)
    index = {int(s): k for k, s in enumerate(states)}
    dim = len(states)
    diag = np.zeros(dim)
    rows, cols = [], []
    for k, s in enumerate(states):
        s = int(s)
        for i in range(n):
            j = (i + 1) % n
            bi = (s >> i) & 1
            bj = (s >> j) & 1
            diag[k] += 0.25 if bi == bj else -0.25
            if bi == 0 and bj == 1:
                # S+_i S-_{i+1}: sube i, baja j.
                t = s | (1 << i)
                t &= ~(1 << j)
                rows.append(index[t])
                cols.append(k)
    K = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim), dtype=complex)
    return diag, K


def hamiltonian(n, M, Delta, Phi):
    diag, K = sector_operators(n, M)
    ph = np.exp(1j * Phi / n)
    return sp.diags(Delta * diag) + 0.5 * (ph * K + np.conj(ph) * K.getH())


_lanczos_start = {}  # último autovector por (n, M): arranque en caliente


def sector_spectrum(n, M, Delta, Phi, k=1):
    """Los k autovalores más bajos del sector."""
    H = hamiltonian(n, M, Delta, Phi)
    if H.shape[0] <= DENSE_MAX:
        return np.linalg.eigvalsh(H.toarray())[:k]
    w, V = spla.eigsh(H, k=k, which="SA", tol=1e-13, v0=_lanczos_start.get((n, M)))
    _lanczos_start[(n, M)] = V[:, np.argmin(w)]
    return np.sort(w)


def ground_energy(n, Delta, Phi, sectors=None):
    """Energía fundamental: sectores de magnetización mínima y ferromagnético."""
    if sectors is None:
        sectors = sorted({n // 2, (n + 1) // 2})
        e_min = min(sector_spectrum(n, M, Delta, Phi)[0] for M in sectors)
        return min(e_min, n * Delta / 4)
    return min(sector_spectrum(n, M, Delta, Phi)[0] for M in sectors)


def eps(n, Delta, Phi):
    """eps_n(Delta, Phi) = -E0/n."""
    return -ground_energy(n, Delta, Phi) / n


def eps_full(n, Delta, Phi):
    """eps_n recorriendo todos los sectores (comprobación)."""
    return -ground_energy(n, Delta, Phi, sectors=range(n + 1)) / n


def ground_state(n, Delta, Phi, M=None):
    """Autovector fundamental del sector M (por defecto floor(n/2))."""
    M = n // 2 if M is None else M
    H = hamiltonian(n, M, Delta, Phi)
    if H.shape[0] <= DENSE_MAX:
        w, V = np.linalg.eigh(H.toarray())
        return w[0], V[:, 0]
    w, V = spla.eigsh(H, k=1, which="SA", tol=1e-13)
    return w[0], V[:, 0]


def full_hamiltonian(n, Delta, Phi):
    """H en todo el espacio de 2^n estados (comprobación; n pequeño)."""
    dim = 2 ** n
    diag = np.zeros(dim)
    rows, cols = [], []
    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            bi = (s >> i) & 1
            bj = (s >> j) & 1
            diag[s] += 0.25 if bi == bj else -0.25
            if bi == 0 and bj == 1:
                t = (s | (1 << i)) & ~(1 << j)
                rows.append(t)
                cols.append(s)
    K = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(dim, dim), dtype=complex)
    ph = np.exp(1j * Phi / n)
    return (sp.diags(Delta * diag) + 0.5 * (ph * K + np.conj(ph) * K.getH())).toarray()

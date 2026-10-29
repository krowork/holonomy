"""SDP primal del problema de pegado en un anillo.

maximizar v  sujeto a  rho >= 0, tr rho = 1,
    <sigma_a^i> = 0 en cada sitio,
    <sigma_a^i sigma_b^j> = v T^e_ab en cada arista e = (i -> j), T^e = -R_e.
"""
import itertools

import cvxpy as cp
import numpy as np

from .paulis import pauli_string

DEFAULT_SOLVER = "CLARABEL"


def _tr(P, rho):
    """tr(P rho) = sum_ij P_ij rho_ji, sin formar el producto de matrices
    (con productos densos de 64x64, CVXPY necesita decenas de GB en n = 6)."""
    return cp.real(cp.sum(cp.multiply(P.T, rho)))


def ring_edges(n):
    return [(i, (i + 1) % n) for i in range(n)]


def primal(n, Rs, solver=DEFAULT_SOLVER, edges=None, return_state=False, **kw):
    """v_max del anillo con matrices R_e (una por arista orientada i -> i+1).

    Devuelve v (y rho si return_state).
    """
    edges = ring_edges(n) if edges is None else edges
    d = 2 ** n
    rho = cp.Variable((d, d), hermitian=True)
    v = cp.Variable()
    cons = [rho >> 0, cp.real(cp.trace(rho)) == 1]
    for i in range(n):
        for a in range(3):
            cons.append(_tr(pauli_string(n, ((i, a),)), rho) == 0)
    for (i, j), R in zip(edges, Rs):
        T = -np.asarray(R)
        for a, b in itertools.product(range(3), repeat=2):
            P = pauli_string(n, ((i, a), (j, b)))
            cons.append(_tr(P, rho) == v * T[a, b])
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver=solver, **kw)
    if return_state:
        return float(v.value), np.asarray(rho.value), prob.status
    return float(v.value)


# ---------------------------------------------------------------------------
# Primal reducido por U(1): rho conmuta con la magnetización total.
# ---------------------------------------------------------------------------

def _sector_basis(n, k):
    """Estados de n bits con k unos (qubits en |1>), en orden creciente."""
    return [s for s in range(2 ** n) if bin(s).count("1") == k]


def _bit(s, n, i):
    return (s >> (n - 1 - i)) & 1


def primal_u1(n, Theta, solver=DEFAULT_SOLVER, return_blocks=False, **kw):
    """Primal en el gauge uniforme T^e = -R_z(Theta/n), con rho en bloques de
    número de giros fijo (la optimalidad no se pierde: el problema es invariante
    bajo U(1) alrededor de z y se puede promediar).

    En un estado U(1)-invariante las únicas correlaciones de arista no nulas son
    XX+YY, XY-YX y ZZ, y <Z_i> es la única marginal posible. Se imponen
    tr(rho A_i) = 2v cos(phi), tr(rho B_i) = 2v sin(phi), tr(rho C_i) = v y
    <Z_i> = 0, con A_i = -(XX+YY), B_i = XY-YX, C_i = -ZZ.
    """
    phi = Theta / n
    blocks, cons = [], []
    trace = 0
    lin = {key: 0 for key in itertools.product(range(n), ("A", "B", "C", "Z"))}
    v = cp.Variable()
    for k in range(n + 1):
        basis = _sector_basis(n, k)
        idx = {s: m for m, s in enumerate(basis)}
        dim = len(basis)
        rb = cp.Variable((dim, dim), hermitian=True)
        blocks.append(rb)
        cons.append(rb >> 0)
        trace = trace + cp.real(cp.trace(rb))
        for i in range(n):
            j = (i + 1) % n
            Cd = np.zeros(dim)
            Zd = np.zeros(dim)
            A = np.zeros((dim, dim), dtype=complex)
            B = np.zeros((dim, dim), dtype=complex)
            for m, s in enumerate(basis):
                zi = 1 - 2 * _bit(s, n, i)
                zj = 1 - 2 * _bit(s, n, j)
                Cd[m] = -zi * zj
                Zd[m] = zi
                if zi != zj:
                    t = s ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))
                    mt = idx[t]
                    # XX+YY = 2(s+_i s-_j + h.c.): intercambio con elemento 2.
                    A[mt, m] += -2.0
                    # XY-YX = 2i(s+_i s-_j - h.c.). s+ = |0><1| sube |1> -> |0>.
                    # Si i está en |1> (zi=-1) y j en |0>, s+_i s-_j actúa.
                    sign = 1.0 if zi == -1 else -1.0
                    B[mt, m] += 2j * sign
            lin[(i, "A")] += _tr(A, rb)
            lin[(i, "B")] += _tr(B, rb)
            lin[(i, "C")] += cp.real(cp.sum(cp.multiply(Cd, cp.real(cp.diag(rb)))))
            lin[(i, "Z")] += cp.real(cp.sum(cp.multiply(Zd, cp.real(cp.diag(rb)))))
    cons.append(trace == 1)
    for i in range(n):
        cons += [
            lin[(i, "A")] == 2 * v * np.cos(phi),
            lin[(i, "B")] == 2 * v * np.sin(phi),
            lin[(i, "C")] == v,
            lin[(i, "Z")] == 0,
        ]
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver=solver, **kw)
    if return_blocks:
        return float(v.value), [np.asarray(b.value) for b in blocks], prob.status
    return float(v.value)


def numerical_rank(rho, tol=1e-6):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return int(np.sum(w > tol)), np.sort(w)[::-1]

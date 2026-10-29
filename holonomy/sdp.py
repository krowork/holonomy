"""SDP primal del problema de pegado en un anillo.

maximizar v  sujeto a  rho >= 0, tr rho = 1,
    <sigma_a^i> = 0 en cada sitio,
    <sigma_a^i sigma_b^j> = v T^e_ab en cada arista e = (i -> j), T^e = -R_e.
"""
import itertools

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

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


def u1_sector_operators(n, edges, k):
    """Operadores de las restricciones en el sector de k giros (base de
    `_sector_basis`), como matrices dispersas: por arista (i, j),
    A = -(X_iX_j+Y_iY_j), B = X_iY_j-Y_iX_j y C = -Z_iZ_j; por sitio, Z_i.

    Devuelve (basis, ops), con ops[("A", e)], ops[("B", e)], ops[("C", e)] y
    ops[("Z", i)]; e es el índice de la arista en `edges`.
    """
    basis = _sector_basis(n, k)
    idx = {s: m for m, s in enumerate(basis)}
    dim = len(basis)
    ops = {}
    for i in range(n):
        ops[("Z", i)] = sp.diags([float(1 - 2 * _bit(s, n, i)) for s in basis], shape=(dim, dim))
    for e, (i, j) in enumerate(edges):
        Cd = np.zeros(dim)
        rows, cols, va, vb = [], [], [], []
        for m, s in enumerate(basis):
            zi = 1 - 2 * _bit(s, n, i)
            zj = 1 - 2 * _bit(s, n, j)
            Cd[m] = -zi * zj
            if zi != zj:
                t = s ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))
                rows.append(idx[t])
                cols.append(m)
                # XX+YY = 2(s+_i s-_j + h.c.): intercambio con elemento 2.
                va.append(-2.0)
                # XY-YX = 2i(s+_i s-_j - h.c.). s+ = |0><1| sube |1> -> |0>.
                # Si i está en |1> (zi=-1) y j en |0>, s+_i s-_j actúa.
                vb.append(2j * (1.0 if zi == -1 else -1.0))
        ops[("A", e)] = sp.csr_matrix((va, (rows, cols)), shape=(dim, dim), dtype=complex)
        ops[("B", e)] = sp.csr_matrix((vb, (rows, cols)), shape=(dim, dim), dtype=complex)
        ops[("C", e)] = sp.diags(Cd, shape=(dim, dim))
    return basis, ops


def primal_u1_graph(n, edges, phis, solver=DEFAULT_SOLVER, return_blocks=False, **kw):
    """Primal de un grafo cualquiera con datos abelianos, T^e = -R_z(phi_e) en
    cada arista orientada e = (i -> j), y rho en bloques de número de giros
    fijo (la optimalidad no se pierde: el problema es invariante bajo U(1)
    alrededor de z y se puede promediar).

    En un estado U(1)-invariante las únicas correlaciones de arista no nulas son
    XX+YY, XY-YX y ZZ, y <Z_i> es la única marginal posible. Se imponen
    tr(rho A_e) = 2v cos(phi_e), tr(rho B_e) = 2v sin(phi_e), tr(rho C_e) = v y
    <Z_i> = 0.
    """
    blocks, cons = [], []
    trace = 0
    lin = {}
    v = cp.Variable()
    for k in range(n + 1):
        basis, ops = u1_sector_operators(n, edges, k)
        dim = len(basis)
        rb = cp.Variable((dim, dim), hermitian=True)
        blocks.append(rb)
        cons.append(rb >> 0)
        trace = trace + cp.real(cp.trace(rb))
        for key, M in ops.items():
            if key[0] in ("C", "Z"):
                term = cp.sum(cp.multiply(M.diagonal().real, cp.real(cp.diag(rb))))
            else:
                term = _tr(M.toarray(), rb)
            lin[key] = lin.get(key, 0) + term
    cons.append(trace == 1)
    for e, phi in enumerate(phis):
        cons += [
            lin[("A", e)] == 2 * v * np.cos(phi),
            lin[("B", e)] == 2 * v * np.sin(phi),
            lin[("C", e)] == v,
        ]
    cons += [lin[("Z", i)] == 0 for i in range(n)]
    prob = cp.Problem(cp.Maximize(v), cons)
    prob.solve(solver=solver, **kw)
    if return_blocks:
        return float(v.value), [np.asarray(b.value) for b in blocks], prob.status
    return float(v.value)


def primal_u1(n, Theta, solver=DEFAULT_SOLVER, return_blocks=False, **kw):
    """Primal del anillo en el gauge uniforme T^e = -R_z(Theta/n), reducido
    por U(1) (caso particular de `primal_u1_graph`)."""
    return primal_u1_graph(n, ring_edges(n), [Theta / n] * n, solver=solver,
                           return_blocks=return_blocks, **kw)


def numerical_rank(rho, tol=1e-6):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return int(np.sum(w > tol)), np.sort(w)[::-1]

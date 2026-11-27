"""Piezas ancladas: p(n, Theta) = p_loc para todo n en 3Z a partir de SDP finitos (celda_k_notas.md, 9.17).

Una pieza de longitud l es un estado eta (invariante por U(1)) de l-1 sitios libres entre dos sitios anclados
en estados clásicos puros |v> y |v+l mod 3>. Sus l aristas (las dos que tocan los anclajes son diagonales)
suman una matriz S(eta). Para la familia {eta_v}_{v=0,1,2}:  K_l = (1/3) sum_v S(eta_v).
p_pieza(l) = max p tal que l * rho_p in K_l, rho_p = I/9 + p tau, tau = sigma - I/9  (Theta = 0).
Por los lemas de anclaje y pegado (9.17), si n = sum l_i con n en 3Z, p(n, Theta) >= min_i p_pieza(l_i)
para toda Theta; y p(n, Theta) <= p_loc siempre.
Uso: python scripts/sdp_piezas.py eps l [l ...]   (NREF, SOLVER en el entorno)
"""
import itertools
import os
import sys

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from sdp_sensibilidad import datos

SOLVER = os.environ.get("SOLVER", "CLARABEL")
# entradas independientes de la marginal (U(1)): diagonal salvo (2,2), (2,1), (2,0) [traza y tr1 = tr2],
# y los elementos fuera de la diagonal dentro de cada sector de carga de la arista
DIAG = [(a, b) for a in range(3) for b in range(3) if not (a == 2 and b in (0, 1, 2))]
OFF = [((0, 1), (1, 0)), ((1, 2), (2, 1)), ((0, 2), (1, 1)), ((0, 2), (2, 0)), ((1, 1), (2, 0))]


def pieza(l, v):
    """Bloques de carga de los l-1 sitios libres; para cada bloque, las matrices C[(r,c)] con
    (sum de las l aristas)[r, c] = tr(W C)."""
    w = (v + l) % 3
    m = l - 1
    confs = list(itertools.product(range(3), repeat=m))
    bloques = {}
    for x in confs:
        bloques.setdefault(sum(x), []).append(x)
    out = []
    for q, base in bloques.items():
        idx = {x: i for i, x in enumerate(base)}
        d = len(base)
        acc = {}

        def pon(r, c, i, j):
            acc.setdefault((r, c), []).append((i, j))

        for x in base:
            full = (v,) + x + (w,)
            for e in range(l):  # arista (e, e+1) de la pieza completa
                a, b = full[e], full[e + 1]
                # x' que difiere de x solo en los sitios e, e+1 (que no sean anclajes), misma carga
                libres = [s for s in (e, e + 1) if 1 <= s <= m]
                for vals in itertools.product(range(3), repeat=len(libres)):
                    f2 = list(full)
                    for s, val in zip(libres, vals):
                        f2[s] = val
                    x2 = tuple(f2[1:m + 1])
                    if x2 not in idx:
                        continue
                    a2, b2 = f2[e], f2[e + 1]
                    # tr(W |x><x2|-entrada): marginal[(a,b),(a2,b2)] += W[x, x2]
                    pon(a * 3 + b, a2 * 3 + b2, idx[x], idx[x2])
        Cs = {}
        for (r, c), lst in acc.items():
            ii, jj = zip(*lst)
            M = sp.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(d, d)).tocsr()
            Cs[(r, c)] = M.T  # sum_ij W_ij M_ij = tr(W M^T)
        out.append((q, d, Cs))
    return out


def p_pieza(l, sigma, verbose=False, coh=False):
    """coh=False: max p.  coh=True: diagonal fija en p_loc y max s, coherencias = s * las de rho_{p_loc}."""
    tau = sigma - np.eye(9) / 9
    p = cp.Variable()
    if coh:
        ploc = (1 / 9) / (1 / 9 - np.linalg.eigvalsh(sigma)[0])
        tau = tau * ploc
    cons = []
    tot = {}
    for v in range(3):
        Ws = []
        for (q, d, Cs) in pieza(l, v):
            W = cp.Variable((d, d), hermitian=True)
            cons.append(W >> 0)
            Ws.append(W)
            for rc, C in Cs.items():
                tot.setdefault(rc, []).append(cp.trace(W @ C) / 3)
        cons.append(sum(cp.real(cp.trace(W)) for W in Ws) == 1)

    def S(r, c):
        return sum(tot.get((r, c), [])) if (r, c) in tot else 0

    for (a, b) in DIAG:
        r = a * 3 + b
        cons.append(cp.real(S(r, r)) == l * (1 / 9 + (1 if coh else p) * tau[r, r].real))
    for (x, y) in OFF:
        r, c = x[0] * 3 + x[1], y[0] * 3 + y[1]
        val = S(r, c)
        cons.append(cp.real(val) == l * p * tau[r, c].real)
        cons.append(cp.imag(val) == l * p * tau[r, c].imag)
    prob = cp.Problem(cp.Maximize(p), cons)
    prob.solve(solver=SOLVER, verbose=verbose)
    return p.value, prob.status


if __name__ == "__main__":
    eps = float(sys.argv[1])
    sigma = datos(eps)
    w = np.linalg.eigvalsh(sigma)
    ploc = (1 / 9) / (1 / 9 - w[0])
    print(f"eps={eps}: p_loc = {ploc:.10f}", flush=True)
    for l in [int(x) for x in sys.argv[2:]]:
        pv, st = p_pieza(l, sigma)
        print(f"  l={l}: p_pieza = {pv:.10f} ({st}), p_loc - p_pieza = {ploc - pv:.3e}", flush=True)
